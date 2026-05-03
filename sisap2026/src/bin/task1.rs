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

    // Load Ground Truth
    let gt_knns = dao::generic_loader::par_load::<_, i32, _, _>(
        &args.source_path,
        "allknn/knns",
        None,
        CHUNK_SIZE,
        |row| row.to_vec(),
    )
    .unwrap();

    // 2. Similarity Computation
    const K_TOP: usize = 15;
    const K_RECALL: usize = 15;

    log::info!(
        "Computing all-pairs similarities (N={}) to find Top {}...",
        num_data,
        K_TOP
    );
    let sim_start = Instant::now();

    use rayon::prelude::*;
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let progress = AtomicUsize::new(0);

    let top_k_indices: Vec<Vec<usize>> = (0..num_data)
        .into_par_iter()
        .map(|i| {
            // Min-heap to keep track of the largest K_TOP elements
            // We use Reverse to turn the max-heap into a min-heap based on similarity
            let mut heap = BinaryHeap::with_capacity(K_TOP + 1);

            for j in 0..num_data {
                let sim = bits::evp::similarity(&data[i], &data[j]);
                if heap.len() < K_TOP {
                    heap.push(Reverse((sim, j)));
                } else if sim > heap.peek().unwrap().0.0 {
                    heap.push(Reverse((sim, j)));
                    heap.pop();
                }
            }

            // Extract from heap and sort descending
            let mut top_k: Vec<_> = heap.into_iter().map(|Reverse((sim, idx))| (sim, idx)).collect();
            top_k.sort_unstable_by(|a, b| b.0.cmp(&a.0));
            
            let count = progress.fetch_add(1, Ordering::Relaxed) + 1;
            if count % 10000 == 0 {
                log::info!("Processed {}/{} queries... ({:.2} s)", count, num_data, sim_start.elapsed().as_secs_f32());
            }

            top_k.into_iter().map(|(_, idx)| idx).collect()
        })
        .collect();

    let sim_time = sim_start.elapsed().as_secs_f32();
    log::info!("Similarity computation completed in {:.2} s", sim_time);

    // 3. Recall Calculation
    log::info!("Computing Recall...");
    let recall_start = Instant::now();
    let mut total_recall = 0.0;

    for i in 0..num_data {
        // Python code: gt_top = set(gt_knns[i, :K_RECALL] - 1)
        // my_top = set(top_100_indices[i, :K_RECALL])
        let gt_top: std::collections::HashSet<usize> = gt_knns[i]
            .iter()
            .take(K_RECALL)
            .map(|&idx| (idx - 1) as usize)
            .collect();
        let my_top: std::collections::HashSet<usize> = top_k_indices[i]
            .iter()
            .take(K_RECALL)
            .copied()
            .collect();

        let overlap = gt_top.intersection(&my_top).count();
        total_recall += overlap as f64 / K_RECALL as f64;
    }
    let avg_recall = total_recall / num_data as f64;
    let recall_calc_time = recall_start.elapsed().as_secs_f32();

    // Final Summary Output
    let load_and_convert_time = load_start.elapsed().as_secs_f32() - sim_time - recall_calc_time;

    println!("\n=============================================");
    println!("{:^45}", "FINAL EVALUATION SUMMARY");
    println!("=============================================");
    println!("{:<25} {:>10.2} s", "Loading & Conversion:", load_start.elapsed().as_secs_f32() - sim_start.elapsed().as_secs_f32());
    println!("{:<25} {:>10.2} s", "Similarity Computation:", sim_time);
    println!("{:<25} {:>10.2} s", "Recall Calculation:", recall_calc_time);
    println!("---------------------------------------------");
    println!("{:<25} {:>10.4}", format!("AVERAGE RECALL@{}:", K_RECALL), avg_recall);
    println!("=============================================");

    log::info!(
        "Total program execution time: {:.2} s",
        (program_start.elapsed()).as_secs_f32()
    );

    Ok(())
}
