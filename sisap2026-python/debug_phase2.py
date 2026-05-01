import time
from huggingface_hub import hf_hub_download
import h5py
import numpy as np

FILENAME = "benchmark-dev-wikipedia-bge-m3-small.h5"
CHUNK_SIZE = 8192
NON_ZEROS = 512

class EvpBits:
    def __init__(self, ones, negative_ones, W):
        self.ones = ones
        self.negative_ones = negative_ones
        self.W = W

    @classmethod
    def from_embedding(cls, embedding, non_zeros):
        W = len(embedding)
        abs_emb = np.abs(embedding)
        
        # Get indices of top `non_zeros` largest absolute values
        # using argsort to mimic Rust's arg_sort exactly
        if non_zeros < W:
            # argsort returns ascending order, so we take the last `non_zeros`
            top_indices = np.argsort(abs_emb)[-non_zeros:]
        else:
            top_indices = np.arange(W)
        
        # We need to construct binary masks for ones and negative_ones
        ones = np.zeros(W, dtype=bool)
        negative_ones = np.zeros(W, dtype=bool)
        
        for idx in top_indices:
            val = embedding[idx]
            if val > 0:
                ones[idx] = True
            elif val < 0:
                negative_ones[idx] = True
                
        return cls(ones, negative_ones, W)

    @classmethod
    def from_embeddings(cls, dataset, non_zeros, chunk_size=8192):
        num_rows, W = dataset.shape
        data_evp = []
        
        for i in range(0, num_rows, chunk_size):
            chunk = dataset[i:i+chunk_size]
            abs_chunk = np.abs(chunk)
            
            if non_zeros < W:
                top_indices = np.argpartition(abs_chunk, -non_zeros, axis=1)[:, -non_zeros:]
            else:
                top_indices = np.tile(np.arange(W), (chunk.shape[0], 1))
                
            rows = np.arange(chunk.shape[0])[:, np.newaxis]
            top_vals = chunk[rows, top_indices]
            
            ones_mat = np.zeros(chunk.shape, dtype=bool)
            negative_ones_mat = np.zeros(chunk.shape, dtype=bool)
            
            ones_mat[rows, top_indices] = top_vals > 0
            negative_ones_mat[rows, top_indices] = top_vals < 0
            
            for j in range(chunk.shape[0]):
                data_evp.append(cls(ones_mat[j], negative_ones_mat[j], W))
                
        return data_evp

def evp_similarity(a, b):
    # aa = a.ones & b.ones
    # bb = a.negative_ones & b.negative_ones
    # cc = a.ones & b.negative_ones
    # dd = b.ones & a.negative_ones
    
    aa = np.logical_and(a.ones, b.ones).sum()
    bb = np.logical_and(a.negative_ones, b.negative_ones).sum()
    cc = np.logical_and(a.ones, b.negative_ones).sum()
    dd = np.logical_and(b.ones, a.negative_ones).sum()
    
    W = a.W
    return float((aa + bb + W * 2) - (cc + dd))

def main():
    print("Downloading/Locating Wikipedia data from Hugging Face Hub...")
    source_path = hf_hub_download(
        repo_id="SISAP-Challenges/SISAP2026",
        filename=FILENAME,
        repo_type="dataset",
    )
    
    program_start = time.time()
    
    print("Loading Wikipedia data...")
    load_start = time.time()
    
    f16_start = time.time()
    
    # Load f16 data
    with h5py.File(source_path, 'r') as f:
        # Load the whole train dataset as f16
        data_f16 = f['train'][:].astype(np.float16)
        
    print(f"f16 data loaded in {time.time() - f16_start:.3f} s")
    print(f"First row of Wikipedia data : {data_f16[0]}")
    
    f32_start = time.time()
    
    # In Python, we load as float32 and convert to EvpBits
    # Using the optimized from_embeddings which processes in chunks
    with h5py.File(source_path, 'r') as f:
        dataset = f['train']
        data_evp = EvpBits.from_embeddings(dataset, NON_ZEROS, chunk_size=CHUNK_SIZE)
        
    print(f"f32 (EvpBits) data loaded in {time.time() - f32_start:.3f} s")
    
    num_data = len(data_evp)
    print(f"Wikipedia data size: {num_data}. Total load time: {time.time() - load_start:.3f} s")
    
    if num_data >= 10:
        print("Comparing distances (f16 Inner Product vs EvpBits Similarity) for element 0 vs first 10:")
        
        f16_inner_products = []
        evp_similarities = []
        
        for i in range(10):
            # inner product in float32
            a_f32 = data_f16[0].astype(np.float32)
            b_f32 = data_f16[i].astype(np.float32)
            inner_product = np.sum(a_f32 * b_f32)
            f16_inner_products.append(float(inner_product))
            
            sim = evp_similarity(data_evp[0], data_evp[i])
            evp_similarities.append(sim)
            
        print(f"f16 Inner Products: {f16_inner_products}")
        print(f"EvpBits Similarities: {evp_similarities}")
        
        max_sim_embedding = np.ones(1024, dtype=np.float32)
        max_sim_evp = EvpBits.from_embedding(max_sim_embedding, NON_ZEROS)
        max_sim = evp_similarity(max_sim_evp, max_sim_evp)
        
        normed_sims = [s / max_sim for s in evp_similarities]
        print(f"EvpBits Normalized Similarities (max={max_sim:.1f}): {normed_sims}")
        
    print(f"Total program execution time: {time.time() - program_start:.2f} s")

if __name__ == '__main__':
    main()
