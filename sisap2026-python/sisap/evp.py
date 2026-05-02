import time
import numpy as np

class EvpBits:
    def __init__(self, ones, negative_ones, dim):
        self.ones = ones
        self.negative_ones = negative_ones
        self.dim = dim

    @classmethod
    def from_embedding(cls, embedding, non_zeros):
        dim = len(embedding)
        if non_zeros >= dim:
            raise ValueError(f"non_zeros ({non_zeros}) must be strictly less than embedding dimension dim ({dim})")
            
        abs_emb = np.abs(embedding)
        
        # using argsort to mimic Rust's arg_sort exactly
        top_indices = np.argsort(abs_emb)[-non_zeros:]
        
        ones = np.zeros(dim, dtype=bool)
        negative_ones = np.zeros(dim, dtype=bool)
        
        for idx in top_indices:
            val = embedding[idx]
            if val > 0:
                ones[idx] = True
            elif val < 0:
                negative_ones[idx] = True
                
        return cls(ones, negative_ones, dim)

    @classmethod
    def from_embeddings(cls, dataset, non_zeros, chunk_size=8192):
        num_rows, dim = dataset.shape
        if non_zeros >= dim:
            raise ValueError(f"non_zeros ({non_zeros}) must be strictly less than embedding dimension dim ({dim})")
            
        data_evp = []
        
        for i in range(0, num_rows, chunk_size):
            chunk = dataset[i:i+chunk_size]
            abs_chunk = np.abs(chunk)
            
            top_indices = np.argpartition(abs_chunk, -non_zeros, axis=1)[:, -non_zeros:]
                
            rows = np.arange(chunk.shape[0])[:, np.newaxis]
            top_vals = chunk[rows, top_indices]
            
            ones_mat = np.zeros(chunk.shape, dtype=bool)
            negative_ones_mat = np.zeros(chunk.shape, dtype=bool)
            
            ones_mat[rows, top_indices] = top_vals > 0
            negative_ones_mat[rows, top_indices] = top_vals < 0
            
            for j in range(chunk.shape[0]):
                data_evp.append(cls(ones_mat[j], negative_ones_mat[j], dim))
                
        return data_evp

        
def get_max_similarity(dim, non_zeros):
    """
    Returns the maximum possible EvpBits similarity for a given dimension dim
    and number of non-zero elements. This corresponds to the similarity of
    a vector with itself.
    """
    return float(non_zeros + 2 * dim)


def evp_similarity(a, b):
    """
    Computes the EvpBits similarity between two EvpBits objects.
    """
    aa = np.logical_and(a.ones, b.ones).sum()
    bb = np.logical_and(a.negative_ones, b.negative_ones).sum()
    cc = np.logical_and(a.ones, b.negative_ones).sum()
    dd = np.logical_and(b.ones, a.negative_ones).sum()
    
    dim = a.dim
    return float((aa + bb + dim * 2) - (cc + dd))

def evp_similarity_batch(queries, targets):
    """
    Computes similarities between two lists/arrays of EvpBits objects using 
    optimized matrix multiplication.
    Returns a numpy array of shape (len(queries), len(targets)).
    """
    if not queries or not targets:
        return np.array([[]])
        
    dim = queries[0].dim
    
    # Convert to ternary matrices (-1, 0, 1) and use float32 for BLAS speed
    Q = (np.array([q.ones for q in queries], dtype=np.int8) - 
         np.array([q.negative_ones for q in queries], dtype=np.int8)).astype(np.float32)
    T_T = (np.array([t.ones for t in targets], dtype=np.int8) - 
           np.array([t.negative_ones for t in targets], dtype=np.int8)).astype(np.float32).T
    
    return np.dot(Q, T_T) + 2 * dim

def compute_all_similarities_batch(evp_list, k_top=100, batch_size=1000):
    """
    Computes all-pairs EvpBits similarities directly from the underlying 
    `ones` and `negative_ones` boolean arrays of the `EvpBits` objects.
    Returns an array of shape (N, K) containing the indices of the Top-K elements for each vector.
    """
    N = len(evp_list)
    if N == 0:
        return np.array([])
    dim = evp_list[0].dim      
    
    # Convert all EvpBits objects to a single ternary matrix (-1, 0, 1) in float32
    print("Re-assembling matrices from EvpBits objects for fast batched computation...", flush=True)
    convert_start = time.time()  
    T = (np.array([e.ones for e in evp_list], dtype=np.int8) - 
         np.array([e.negative_ones for e in evp_list], dtype=np.int8)).astype(np.float32)
    T_T = T.T
    print(f"Matrix re-assembly took {time.time() - convert_start:.2f} s")
    
    # Computing all-pairs similarity
    sim_start = time.time()    
    top_100_indices = np.zeros((N, k_top), dtype=np.int32)    
    for i in range(0, N, batch_size):
        end = min(i + batch_size, N)
        B_T = T[i:end]
        
        # Calculate dot product
        sim = np.dot(B_T, T_T)
        sim += 2 * dim
        
        # Extract top 100 indices
        top_k = np.argpartition(sim, -k_top, axis=1)[:, -k_top:]
        
        # Sort the top K elements properly by similarity (descending)
        top_k_vals = np.take_along_axis(sim, top_k, axis=1)
        sort_order = np.argsort(-top_k_vals, axis=1)
        top_100_indices[i:end] = np.take_along_axis(top_k, sort_order, axis=1)
        
        # Progress reporting
        if (i // 20000) > (max(0, i - batch_size) // 20000):
            print(f"\rProcessed {i}/{N} vectors... ({time.time() - sim_start:.2f} s elapsed)", end="", flush=True)
            
    print("\n", end="")
    print(f"Similarity computation and top {k_top} extraction took {time.time() - sim_start:.2f} s", flush=True)
    
    return top_100_indices