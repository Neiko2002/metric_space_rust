/* A copy of challenge1 to test the Hamiltonians */

use anyhow::Result;
use clap::Parser;
use ndarray::{
    Array1, Array2, ArrayView, ArrayView1, ArrayViewMut1, ArrayViewMut2, Axis, Ix1, Ix2,
};

use half::f16;
use hamiltonians::{get_cycle_lengths, get_cycle_lookup_table, get_vertex_number, make_pascal};
use std::time::Instant;
use utils::non_nan::NonNan;
use utils::non_nan_f64::NonNanF64;
use utils::{arg_sort_big_to_small_2d, arg_sort_small_to_big, Nality};

/// clap parser
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Path to HDF5 source
    source_path: String,
    /// Path to HDF5 target
    output_path: String,
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
    const D: usize = 1024;
    const NON_ZEROS: usize = 512;

    let data_f16: Vec<Array1<f16>> = dao::generic_loader::par_load::<_, half::f16, _, _>(
        &args.source_path,
        "train",
        None,
        CHUNK_SIZE,
        |embedding| embedding.mapv(|f| f),
    )
    .unwrap();

    let end = Instant::now();
    let data_size = data_f16.len();

    log::info!(
        "Wikipedia Loaded {} data in {} s",
        data_size,
        (end - start).as_secs()
    );

    //----- Set up Hamiltonian machinery
    log::info!("Building Pascal triangle and lookup tables...");

    // Build Pascal triangle
    let pas_tri: Vec<Vec<f64>> = make_pascal(D);
    // Build cycle lengths and lookup tables
    let cycle_lengths: Vec<usize> = get_cycle_lengths(D);
    let mut all_tables: Vec<Vec<Vec<bool>>> = vec![vec![vec![]]];

    // TODO make these smaller if we are building multiple NN tables with less width than D.

    for this_x in 1..=NON_ZEROS {
        all_tables.push(get_cycle_lookup_table(
            cycle_lengths[NON_ZEROS],
            this_x,
            &pas_tri,
        ));
    }
    log::info!("Pascal triangle and lookup tables ready.");

    const start_index1: usize = 0;
    let code_size = 25; // The number of levels in the simplex rep too big but OK - TODO make this something more reasoned

    log::info!("Starting to build NN table...");
    let nn_table1 = get_nn_table(
        &data_f16,
        NON_ZEROS,
        D,
        &cycle_lengths,
        &all_tables,
        &pas_tri,
    );

    // const start_index2: usize = code_size;
    //
    // // let nn_table2 = get_nn_table(
    // //     &data_f16,
    //
    // //     NON_ZEROS,
    // //     D,
    // //     &cycle_lengths,
    // //     &all_tables,
    // //     &pas_tri,
    // // );

    let end = Instant::now();
    log::info!(
        "Created NN table (including load time in {} s",
        (end - start).as_secs()
    );

    Ok(())
}

fn get_nn_table(
    data_f16: &Vec<Array1<f16>>,
    x: usize,
    D: usize,
    cycles: &Vec<usize>,
    tables: &Vec<Vec<Vec<bool>>>,
    pas_tri: &Vec<Vec<f64>>,
) -> Array2<usize> {
    let data_size = data_f16.len();

    log::info!("Computing Hamiltonian values for all data points...");
    let hamiltonian_values_in_data_order =
        get_hamiltonians_in_data_order(&data_f16, x, D, &cycles, &tables, &pas_tri, data_size);
    log::info!("Hamiltonian values computed.");

    // hamiltonian_order_to_data_order maps from hamiltonian order to data order
    log::info!("Sorting Hamiltonian values...");
    let hamiltonian_order_to_data_order = get_hamiltonian_order(&hamiltonian_values_in_data_order);
    arg_sort_small_to_big_1d_f64(&hamiltonian_values_in_data_order);
    let hamiltonian_order_to_data_order: Array1<usize> =
        Array1::from(hamiltonian_order_to_data_order); // retype the indices so we can use views over the window
    log::info!("Sorting complete.");

    const window_size: usize = 1000;
    const num_neighbours: usize = 15;
    let mut slice_start = 0;
    let mut chunk_count = 0;
    let start_time = Instant::now();

    // create a nn table
    let mut nn_table: Array2<usize> = Array2::<usize>::zeros((data_size, num_neighbours));

    // initialise the nn table a window_size elements at a time
    // takes the smallest distances within this window
    // TODO could deal with the start/end of the cycle better

    // TODO Make this parallel.

    for selector_slice in hamiltonian_order_to_data_order.axis_chunks_iter(Axis(0), window_size) {
        // pull the data values in snake order

        let mut data_values = Array2::<f32>::zeros((window_size, D));
        for (i, &row) in selector_slice.iter().enumerate() {
            let src = data_f16[hamiltonian_order_to_data_order[row]].view(); // lookup data using the data index drawn from the position_of_data_in_snake
            data_values.row_mut(i).assign(&src.mapv(|x| x.to_f32())); // and convert the row to f32s so we can do the dot operation.
        }
        // Find the  distances over data_values
        let chunk_dists = data_values.dot(&data_values.t()); // dists in snake order over slice

        // get the first num_neighbours out of the sorted_ords and assign to the nn table.
        // the ordered_indices are relative to the range from 0..window_size to get to data relative add the slice_start.
        // the nns are in the natural order

        for row in 0..window_size {
            let (row_window_relative_position_in_hamiltonian_order, _) =
                arg_sort_big_to_small_1d_f32(chunk_dists.row(row)); // sorted ords are relative (to the window) indices

            for col in 0..num_neighbours {
                let absolute_position_hamiltonian_order =
                    row_window_relative_position_in_hamiltonian_order[col] + slice_start; // index into the data
                nn_table[[row + slice_start, col]] = absolute_position_hamiltonian_order;
            }
        }

        slice_start += window_size;
        chunk_count += 1;
        if chunk_count % 10 == 0 {
            let elapsed = start_time.elapsed().as_secs_f64();
            let total_chunks = data_size / window_size;
            let remaining = (elapsed / chunk_count as f64) * (total_chunks - chunk_count) as f64;
            log::info!(
                "Processed {}/{} chunks... Elapsed: {:.2}s, Est. Remaining: {:.2}s",
                chunk_count,
                total_chunks,
                elapsed,
                remaining
            );
        }
    }

    print!("First 2 lines of approx NNS: ");
    for row in 0..2 {
        for i in 0..num_neighbours {
            print!("{} ", nn_table[[row, i]]);
        }
        println!();
    }

    print!("First 2000,2001 lines of approx NNS: ");
    for row in 0..2 {
        for i in 0..num_neighbours {
            print!("{} ", nn_table[[row + 2000, i]]);
        }
        println!();
    }

    nn_table
}

fn get_hamiltonian_order(hamiltonian_values_in_data_order: &Vec<f64>) -> Array1<usize> {
    let (hamiltonian_order, _) = arg_sort_small_to_big_1d_f64(&hamiltonian_values_in_data_order);
    let hamiltonian_order: Array1<usize> = Array1::from(hamiltonian_order); // retype the indices so we can use views over the window
    hamiltonian_order
}

fn get_hamiltonians_in_data_order(
    data_f16: &Vec<Array1<f16>>,
    x: usize,
    D: usize,
    cycles: &&Vec<usize>,
    tables: &&Vec<Vec<Vec<bool>>>,
    pas_tri: &&Vec<Vec<f64>>,
    data_size: usize,
) -> Vec<f64> {
    // hamiltonian_values_in_data_order map from original data index to Hamiltonian f64 vertex values
    let mut hamiltonian_values_in_data_order = Vec::with_capacity(data_size);
    let start_time = Instant::now();
    // Turn the f16 data into Hamiltonian f64 vertex numbers and add to snake_positions_from_data data structure
    for (i, vertex) in data_f16.iter().enumerate() {
        let vertex: Vec<bool> = f16_vec_to_bool_vec(&vertex);
        let v_no: f64 = get_vertex_number(x, D, vertex, &cycles, &tables, &pas_tri);
        hamiltonian_values_in_data_order.push(v_no);
        if (i + 1) % 10000 == 0 {
            let elapsed = start_time.elapsed().as_secs_f64();
            let remaining = (elapsed / (i + 1) as f64) * (data_size - (i + 1)) as f64;
            log::info!(
                "Converted {}/{} vertices... Elapsed: {:.2}s, Est. Remaining: {:.2}s",
                i + 1,
                data_size,
                elapsed,
                remaining
            );
        }
    }
    hamiltonian_values_in_data_order
}

fn median(arrai: &Array1<f16>) -> f32 {
    let mut v = arrai.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let len = v.len();

    if len % 2 == 1 {
        v[len / 2].to_f32()
    } else {
        (v[len / 2 - 1] + v[len / 2]).to_f32() / 2.0
    }
}

fn f16_vec_to_bool_vec(arrai: &Array1<f16>) -> Vec<bool> {
    let median: f32 = median(arrai);
    arrai.iter().map(|&x| (x.to_f32()) < median).collect()
}

pub fn arg_sort_small_to_big_2d_f32(vals: &ArrayView<f32, Ix2>) -> (Vec<usize>, Vec<f32>) {
    let mut enumerated = vals.iter().enumerate().collect::<Vec<(usize, &f32)>>(); // Vec of positions (ords) and values (dists)
    enumerated.sort_by(|a, b| NonNan::new(*a.1).partial_cmp(&NonNan::new(*b.1)).unwrap());
    enumerated.into_iter().unzip()
}

pub fn arg_sort_small_to_big_1d_f64(vals: &Vec<f64>) -> (Vec<usize>, Vec<f64>) {
    let mut enumerated = vals.iter().enumerate().collect::<Vec<(usize, &f64)>>(); // Vec of positions (ords) and values (dists)
    enumerated.sort_by(|a, b| {
        NonNanF64::new(*a.1)
            .partial_cmp(&NonNanF64::new(*b.1))
            .unwrap()
    });
    enumerated.into_iter().unzip()
}

pub fn arg_sort_small_to_big_1d_f32(vals: ArrayView1<f32>) -> (Vec<usize>, Vec<f32>) {
    let mut enumerated = vals.iter().enumerate().collect::<Vec<(usize, &f32)>>(); // Vec of positions (ords) and values (dists)
    enumerated.sort_by(|a, b| NonNan::new(*a.1).partial_cmp(&NonNan::new(*b.1)).unwrap());
    enumerated.into_iter().unzip()
}

pub fn arg_sort_big_to_small_1d_f32(vals: ArrayView1<f32>) -> (Vec<usize>, Vec<f32>) {
    let mut enumerated = vals.iter().enumerate().collect::<Vec<(usize, &f32)>>(); // Vec of positions (ords) and values (dists)
    enumerated.sort_by(|a, b| NonNan::new(*b.1).partial_cmp(&NonNan::new(*a.1)).unwrap());
    enumerated.into_iter().unzip()
}
