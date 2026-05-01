/*
In this task, participants are asked to develop memory-efficient indexing solutions that will be used to compute an approximation of the k-nearest neighbor graph for k=15. Each solution will be run in a Linux container with limited memory and storage resources.
Container specifications: 8 virtual CPUs, 16 GB of RAM, the dataset will be mounted read-only into the container.
Wall clock time for the entire task: 12 hours.
Minimum average recall to be considered in the final ranking: 0.8.
Dataset: WIkipedia (6.5 million vectors (1024 dimensions) ).
The goal is to compute the k-nearest neighbor graph (without self-references), i.e., find the k-nearest neighbors using all objects in the dataset as queries.
We will measure graph’s quality as the recall against a provided gold standard and the full computation time (i.e., including preprocessing, indexing, and search, and postprocessing)
We provide a development dataset; the evaluation phase will use an undisclosed dataset of similar size computed with the same neural model.
*/

use anyhow::Result;
use bits::container::{BitsContainer, Simd256x4};
use clap::Parser;
use hdf5::File as Hdf5File;
use ndarray::{s, Array1, Array2, ArrayView2};
use r_descent::IntoRDescent;

use bits::EvpBits;
use dao::{Dao, DaoMetaData, Normed};
use std::rc::Rc;
use std::time::Instant;
use utils::arg_sort_big_to_small_2d;

// #[global_allocator]
// static GLOBAL: jemallocator::Jemalloc = jemallocator::Jemalloc;

/// clap parser
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Path to HDF5 source
    source_path: String,
    /// Path to HDF5 target
    output_path: String,
}

fn dao_from_data<C: BitsContainer, const W: usize>(
    data: Vec<EvpBits<C, W>>,
    name: String,
    description: String,
) -> anyhow::Result<Dao<EvpBits<C, W>>> {
    let dao_meta = DaoMetaData {
        name: name,
        description: description,
        data_disk_format: "".to_string(),
        path_to_data: "".to_string(),
        normed: Normed::L2,
        num_records: data.len(),
        dim: 1024, //<<<<<<<<<<<<<<<<<<<<, Hard wired for now.
    };

    let data = Array1::from(data); //<<<<<<<<<<<<<<< copy here

    let dao = Dao {
        meta: dao_meta,
        num_data: data.len(),
        base_addr: 0,
        num_queries: 0,
        embeddings: data,
    };

    Ok(dao)
}

fn main() -> Result<()> {
    pretty_env_logger::formatted_timed_builder()
        .filter_level(log::LevelFilter::Trace)
        .init();

    let args = Args::parse();

    log::info!("Loading Wikipedia data...");
    let start = Instant::now();

    const ALL_RECORDS: usize = 0;
    const NUM_QUERIES: usize = 0;
    const CHUNK_SIZE: usize = 8192;
    const NON_ZEROS: usize = 512;

    let data_f16 = dao::generic_loader::par_load::<_, half::f16, _, _>(
        &args.source_path,
        "train",
        None,
        CHUNK_SIZE,
        |embedding| embedding.mapv(|f| f),
    )
    .unwrap();

    log::info!("First row of Wikipedia data : {}", data_f16[0],); // Note 0.0335, -0.0056 first and second numbers in small and large datasets

    let data = dao::generic_loader::par_load::<_, f32, _, _>(
        &args.source_path,
        "train",
        None,
        CHUNK_SIZE,
        |embedding| EvpBits::<Simd256x4, 1024>::from_embedding(embedding, NON_ZEROS),
    )
    .unwrap();

    let num_data = data.len();

    log::info!("Wikipedia data size: {}", num_data,);

    let start_post_load = Instant::now();

    let num_neighbours = 18;
    let delta = 0.01;
    let reverse_list_size = 64;

    log::info!("Creating NN table");

    log::info!("Creating Dao from data...");
    let dao_bsp: Rc<Dao<EvpBits<Simd256x4, 1024>>> = Rc::new(
        dao_from_data::<Simd256x4, 1024>(data, "Wikipedia".to_string(), "Wikipedia".to_string())
            .unwrap(),
    );
    log::info!("Dao created.");

    log::info!("Starting R-Descent indexing (this may take a while)...");
    let descent = dao_bsp.into_rdescent(num_neighbours, reverse_list_size, delta);
    log::info!("R-Descent indexing finished.");

    let end = Instant::now();

    log::info!(
        "Finished (including load time in {} s",
        (end - start).as_secs()
    );
    log::info!(
        "Finished (post load time) in {} s",
        (end - start_post_load).as_secs()
    );

    let neighbours = &descent.neighbours;
    let sims = &descent.similarities;

    // Add 1 to all elements (preserving shape)
    let neighbours = neighbours.mapv(|x| x + 1);

    log::info!("Sorting similarities...");
    let (ords, _) = arg_sort_big_to_small_2d(&sims.view()); // sort the data
    log::info!("Sorting finished.");

    let selected_neighbours: Vec<usize> = {
        let neighbours_ref = &neighbours; // to avoid capture of neighbours
        let mut count = 0;

        ords.rows()
            .into_iter()
            .enumerate()
            .flat_map(|(row_index, ord_row)| {
                count += 1;
                if count % 100000 == 0 {
                    log::info!("Processing row {}/{}...", count, num_data);
                }
                ord_row
                    .iter()
                    .map(move |&col_index| neighbours_ref[[row_index, col_index]])
                    .collect::<Vec<_>>() // to avoid capture of neighbours.
            })
            .collect()
    };

    let selected_neighbours =
        Array2::from_shape_vec((num_data, num_neighbours), selected_neighbours)
            .expect("Failed to create Array2 - indexing error");

    let selected_neighbours = selected_neighbours.slice(s![.., 1..16]); // get the first 15 columns.

    log::info!("Writing to h5 file {}", &args.output_path);
    save_to_h5(&args.output_path, selected_neighbours)?;

    println!("Data saved to h5 file");

    Ok(())
}

pub fn save_to_h5(f_name: &str, data: ArrayView2<usize>) -> Result<()> {
    let file = Hdf5File::create(f_name)?; // open for writing
    let group = file.create_group("/knns")?; // create a group
                                             // TODO do they need the dists too?
    let builder = group.new_dataset_builder();

    let _ds = builder.with_data(&data.to_owned()).create("results")?;

    file.flush()?;

    Ok(())
}
