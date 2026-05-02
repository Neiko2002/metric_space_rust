"""
Performance benchmark for optimized EvpBits similarity computation.

Compares two approaches using the optimized integer-based EvpBits:
1. Multi-threaded nested loop using evp_similarity() (fast integer bit_count).
2. Batched matrix multiplication (fast memory assembly + ternary dot product).

Usage:
    uv run python tests/test_speed_optimized.py
"""
import time
import numpy as np
import h5py
import sys
from pathlib import Path

# Add the project root to sys.path to allow importing the sisap package
sys.path.append(str(Path(__file__).parent.parent))

from sisap import get_h5_file
from sisap.evp_optimized import EvpBits, evp_similarity, evp_similarity_batch

NON_ZEROS = 512
CHUNK_SIZE = 10000

def test_speed():
    BATCH_SIZE = 1000
    
    print(f"Loading first {CHUNK_SIZE} EvpBits objects from the H5 dataset...")
    source_path = get_h5_file(is_small=True)
    with h5py.File(source_path, 'r') as f:
        # Only load the first CHUNK_SIZE elements to keep the test fast
        dataset = f['train'][:CHUNK_SIZE]
        N, dim = dataset.shape
        print(f"Loaded {N} vectors (dim={dim}). Converting to optimized EvpBits...")
        evp_list = EvpBits.from_embeddings(dataset, NON_ZEROS, chunk_size=CHUNK_SIZE)
    
    # Take the first BATCH_SIZE items to compare against all N items
    queries = evp_list[:BATCH_SIZE]
    
    print(f"\n--- Method 1: Nested Loop with evp_similarity (Single-threaded) ---")
    start_time = time.time()
    results_loop = np.zeros((BATCH_SIZE, N), dtype=np.float32)    
    for i in range(BATCH_SIZE):
        q = queries[i]
        for j in range(N):
            results_loop[i, j] = evp_similarity(q, evp_list[j])            
    loop_time = time.time() - start_time
    print(f"Time taken: {loop_time:.4f} seconds")
    
    
    print(f"\n--- Method 2: Matrix Multiplication (evp_similarity_batch) ---")
    start_time = time.time()
    results_matrix = evp_similarity_batch(queries, evp_list)    
    matrix_time = time.time() - start_time
    print(f"Time taken (including matrix setup): {matrix_time:.4f} seconds")
    

    print(f"\n--- Comparison ---")
    print(f"Matrix approach is {loop_time / matrix_time:.2f}x faster!")
    
    # Check correctness
    diff = np.abs(results_loop - results_matrix).max()
    print(f"Max difference between results: {diff}")
    if diff == 0:
        print("Mathematical equivalence: CONFIRMED")
    else:
        print("Mathematical equivalence: FAILED")
        
    print("\n=== Batched Similarity vs Old Implementation ===")
    from sisap.evp import EvpBits as OldEvpBits, evp_similarity_batch as old_batch
    
    print("Loading old EvpBits for comparison...")
    with h5py.File(source_path, 'r') as f:
        dataset_old = f['train'][:CHUNK_SIZE]
        old_evp_list = OldEvpBits.from_embeddings(dataset_old, NON_ZEROS, chunk_size=CHUNK_SIZE)
    queries_old = old_evp_list[:BATCH_SIZE]
    
    print("Running old evp_similarity_batch...")
    start_old = time.time()
    res_old = old_batch(queries_old, old_evp_list)
    time_old = time.time() - start_old
    
    print(f"Old Batch Time: {time_old:.4f} s")
    print(f"New Batch Time: {matrix_time:.4f} s")
    print(f"Optimized batch is {time_old / matrix_time:.2f}x faster than old batch!")

if __name__ == '__main__':
    test_speed()
