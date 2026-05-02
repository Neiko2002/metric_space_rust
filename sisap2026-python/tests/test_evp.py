import numpy as np
import sys
from pathlib import Path

# Add the project root to sys.path to allow importing the sisap package
sys.path.append(str(Path(__file__).parent.parent))

from sisap import EvpBits, evp_similarity

def test_evp_similarity_simple():
    # Create simple embeddings
    emb1 = np.array([1.0, -1.0, 0.0, 0.5], dtype=np.float32)
    emb2 = np.array([1.0, 0.0, -1.0, 0.5], dtype=np.float32)
    
    # NON_ZEROS = 2
    evp1 = EvpBits.from_embedding(emb1, non_zeros=2)
    evp2 = EvpBits.from_embedding(emb2, non_zeros=2)
    
    # evp1 should have ones at [0], negative_ones at [1]
    # evp2 should have ones at [0], negative_ones at [2]
    
    sim = evp_similarity(evp1, evp2)
    
    # Formula: (aa + bb + dim * 2) - (cc + dd)
    # aa = intersection of ones = {0} -> count = 1
    # bb = intersection of negative_ones = {} -> count = 0
    # cc = intersection of ones(1) and neg(2) = {} -> count = 0
    # dd = intersection of ones(2) and neg(1) = {} -> count = 0
    # dim = 4
    # Expected Sim = (1 + 0 + 4*2) - (0 + 0) = 9.0
    
    assert sim == 9.0
    print("test_evp_similarity_simple passed!")

def test_evp_bits_conversion():
    emb = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
    evp = EvpBits.from_embedding(emb, non_zeros=2)
    
    # top 2 absolute are 0.3 (idx 2) and -0.4 (idx 3)
    assert evp.ones[2] == True
    assert evp.negative_ones[3] == True
    assert evp.ones[0] == False
    assert evp.negative_ones[1] == False
    print("test_evp_bits_conversion passed!")

def test_from_embeddings_consistency():
    # Create a small dataset
    np.random.seed(42)
    N = 100
    dim = 128
    non_zeros = 32
    dataset = np.random.randn(N, dim).astype(np.float32)
    
    # 1. Convert using from_embedding (single)
    evp_single = [EvpBits.from_embedding(row, non_zeros) for row in dataset]
    
    # 2. Convert using from_embeddings (batch)
    evp_batch = EvpBits.from_embeddings(dataset, non_zeros, chunk_size=30) # odd chunk size to test boundary
    
    assert len(evp_single) == len(evp_batch)
    
    for i in range(N):
        np.testing.assert_array_equal(evp_single[i].ones, evp_batch[i].ones, err_msg=f"Mismatch in ones at index {i}")
        np.testing.assert_array_equal(evp_single[i].negative_ones, evp_batch[i].negative_ones, err_msg=f"Mismatch in negative_ones at index {i}")
    
    print("test_from_embeddings_consistency passed!")

def test_from_embeddings_with_ties():
    # Elements with same absolute value
    # ties can lead to different indices being picked by argsort vs argpartition
    emb = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    non_zeros = 2
    
    # single (uses argsort)
    evp_s = EvpBits.from_embedding(emb, non_zeros)
    
    # batch (uses argpartition)
    dataset = emb.reshape(1, -1)
    evp_b = EvpBits.from_embeddings(dataset, non_zeros)[0]
    
    # Both should have exactly non_zeros ones set.
    assert evp_s.ones.sum() == non_zeros
    assert evp_b.ones.sum() == non_zeros
    
    # Note: If this fails, it's expected due to different tie-breaking.
    # But it's good to know if they happen to match.
    match = np.array_equal(evp_s.ones, evp_b.ones)
    print(f"test_from_embeddings_with_ties: ones match? {match}")

def test_invalid_non_zeros():
    emb = np.array([1.0, 2.0], dtype=np.float32)
    # non_zeros = 2 is now invalid for dim = 2 (must be strictly less)
    try:
        EvpBits.from_embedding(emb, non_zeros=2)
    except ValueError as e:
        assert "must be strictly less than" in str(e)
        print("test_invalid_non_zeros passed!")
        return
    
    assert False, "Should have raised ValueError for non_zeros == dim"

if __name__ == "__main__":
    test_evp_similarity_simple()
    test_evp_bits_conversion()
    test_from_embeddings_consistency()
    test_from_embeddings_with_ties()
    test_invalid_non_zeros()
    print("All tests passed!")
