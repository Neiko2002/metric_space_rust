use anyhow::Result;
use clap::Parser;
use half::f16;
use ndarray::{s, Array2, Axis};
use std::time::Instant;
use std::sync::atomic::{AtomicUsize, Ordering};
use rayon::prelude::*;

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

    log::info!("Loading and converting Wikipedia data (f16 -> f32)...");
    let load_start = Instant::now();

    const CHUNK_SIZE: usize = 8192;
    const DIM: usize = 1024;

    // Load and convert on the fly to save memory
    let data_vec = dao::generic_loader::par_load::<_, f16, _, _>(
        &args.source_path,
        "train",
        None,
        CHUNK_SIZE,
        |embedding| {
            embedding.iter().map(|&x| x.to_f32()).collect::<Vec<f32>>()
        },
    )
    .unwrap();

    let num_data = data_vec.len();
    log::info!(
        "Wikipedia data size: {}. Load + Conversion time: {:.2} s",
        num_data,
        load_start.elapsed().as_secs_f32()
    );

    log::info!("Creating flat f32 matrix...");
    let data_f32 = Array2::from_shape_vec(
        (num_data, DIM),
        data_vec.into_iter().flatten().collect()
    ).expect("Failed to create Array2 from loaded data");
    
    // data_vec is now dropped, freeing up its memory

    // Load Ground Truth
    let gt_knns = dao::generic_loader::par_load::<_, i32, _, _>(
        &args.source_path,
        "allknn/knns",
        None,
        CHUNK_SIZE,
        |row| row.to_vec(),
    )
    .unwrap();

    // 2. Similarity Computation (Batched)
    const K_TOP: usize = 15;
    const K_RECALL: usize = 15;
    const BATCH_SIZE: usize = 100;

    let num_to_process = num_data; // Process all queries now
    log::info!(
        "Computing similarities for {} queries against {} items (Batch Size: {})...",
        num_to_process,
        num_data,
        BATCH_SIZE
    );

    let sim_start = Instant::now();
    let progress = AtomicUsize::new(0);

    let top_k_indices: Vec<Vec<usize>> = (0..num_to_process)
        .step_by(BATCH_SIZE)
        .collect::<Vec<_>>()
        .into_par_iter()
        .flat_map(|start| {
            let end = (start + BATCH_SIZE).min(num_to_process);
            let queries_batch = data_f32.slice(s![start..end, ..]);
            
            // Optimized matrix dot product (uses SIMD/matrixmultiply)
            let sims = queries_batch.dot(&data_f32.t());
            
            let mut results = Vec::with_capacity(end - start);
            for row in sims.axis_iter(Axis(0)) {
                // Find top K in this query's results
                let mut enumerated: Vec<_> = row.iter().enumerate().collect();
                // Partial sort would be faster but for K=15 a full sort is negligible
                enumerated.sort_unstable_by(|a, b| b.1.partial_cmp(a.1).unwrap());
                
                let top_k: Vec<usize> = enumerated.iter().take(K_TOP).map(|&(idx, _)| idx).collect();
                results.push(top_k);
                
                let count = progress.fetch_add(1, Ordering::Relaxed) + 1;
                if count % 1000 == 0 {
                    log::info!("Processed {}/{} queries... ({:.2} s)", count, num_to_process, sim_start.elapsed().as_secs_f32());
                }
            }
            results
        })
        .collect();

    let sim_time = sim_start.elapsed().as_secs_f32();
    log::info!("Similarity computation completed in {:.2} s", sim_time);

    // 3. Recall Calculation
    log::info!("Computing Recall for {} queries...", num_to_process);
    let mut total_recall = 0.0;

    for i in 0..num_to_process {
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
    let avg_recall = total_recall / num_to_process as f64;

    println!("\n=============================================");
    println!("{:^45}", format!("FINAL EVALUATION SUMMARY (f16 -> f32, N={})", num_to_process));
    println!("=============================================");
    println!("{:<25} {:>10.2} s", "Loading + Conversion:", load_start.elapsed().as_secs_f32() - sim_time);
    println!("{:<25} {:>10.2} s", "Similarity Computation:", sim_time);
    println!("---------------------------------------------");
    println!("{:<25} {:>10.4}", format!("AVERAGE RECALL@{}:", K_RECALL), avg_recall);
    println!("=============================================");

    log::info!(
        "Total program execution time: {:.2} s",
        (program_start.elapsed()).as_secs_f32()
    );

    Ok(())
}
