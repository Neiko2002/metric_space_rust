use anyhow::Result;
use bits::container::Simd256x4;
use bits::EvpBits;
use clap::Parser;
use ndarray::Array1;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Path to HDF5 source
    source_path: String,
    /// Path to HDF5 target (not used but kept for compatibility)
    output_path: String,
}

fn main() -> Result<()> {
    pretty_env_logger::formatted_timed_builder()
        .filter_level(log::LevelFilter::Trace)
        .init();

    let args = Args::parse();
    let program_start = Instant::now();

    log::info!("Loading Wikipedia data...");
    let load_start = Instant::now();

    const CHUNK_SIZE: usize = 8192;
    const NON_ZEROS: usize = 512;

    let f16_start = Instant::now();
    let data_f16 = dao::generic_loader::par_load::<_, half::f16, _, _>(
        &args.source_path,
        "train",
        None,
        CHUNK_SIZE,
        |embedding| embedding.mapv(|f| f),
    )
    .unwrap();
    log::info!(
        "f16 data loaded in {} s",
        (f16_start.elapsed()).as_secs_f32()
    );
    log::info!("First row of Wikipedia data : {}", data_f16[0],);

    let f32_start = Instant::now();
    let data = dao::generic_loader::par_load::<_, f32, _, _>(
        &args.source_path,
        "train",
        None,
        CHUNK_SIZE,
        |embedding| EvpBits::<Simd256x4, 1024>::from_embedding(embedding, NON_ZEROS),
    )
    .unwrap();
    log::info!(
        "f32 (EvpBits) data loaded in {} s",
        (f32_start.elapsed()).as_secs_f32()
    );

    let num_data = data.len();
    log::info!(
        "Wikipedia data size: {}. Total load time: {} s",
        num_data,
        (load_start.elapsed()).as_secs_f32()
    );

    // Compare distances between element 0 and the first 10 elements
    if num_data >= 10 {
        log::info!("Comparing distances (f16 Inner Product vs EvpBits Similarity) for element 0 vs first 10:");

        let mut f16_inner_products = Vec::new();
        let mut evp_similarities = Vec::new();

        for i in 0..10 {
            // Inner product for f16
            let a = &data_f16[0];
            let b = &data_f16[i];
            let inner_product: f32 = a
                .iter()
                .zip(b.iter())
                .map(|(&x, &y)| x.to_f32() * y.to_f32())
                .sum::<f32>();
            f16_inner_products.push(inner_product);

            // BSP Similarity for EvpBits
            let sim = bits::evp::similarity_as_f32(&data[0], &data[i]);
            evp_similarities.push(sim);
        }

        log::info!("f16 Inner Products: {:?}", f16_inner_products);
        log::info!("EvpBits Similarities: {:?}", evp_similarities);
        let max_sim_embedding = Array1::ones(1024);
        let max_sim_evp = EvpBits::<Simd256x4, 1024>::from_embedding(max_sim_embedding, NON_ZEROS);
        let max_sim = bits::evp::similarity_as_f32(&max_sim_evp, &max_sim_evp);
        let normed_sims: Vec<f32> = evp_similarities.iter().map(|&s| s / max_sim).collect();
        log::info!(
            "EvpBits Normalized Similarities (max={:.1}): {:?}",
            max_sim,
            normed_sims
        );
    }

    log::info!(
        "Total program execution time: {:.2} s",
        (program_start.elapsed()).as_secs_f32()
    );

    Ok(())
}
