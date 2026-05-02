# SISAP 2026 Python

Python environment for processing and evaluating the SISAP 2026 HDF5 dataset using the EvpBits (sparsified ternary) approximation.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) installed
- HuggingFace token (optional, for private repo access)

## Setup

```bash
cd sisap2026-python
uv sync
```

## Project Structure

- `sisap/`: Core library containing `EvpBits` logic and data loading utilities.
- `task1.py`: Main evaluation script (Recall@15).
- `tests/`: Unit tests and verification scripts.

## Usage

### 1. Evaluate Task 1 (Recall)
Computes the average Recall@15 for the dataset by comparing EvpBits results with the provided ground-truth KNNs.

```bash
uv run python task1.py
```

### 2. Verify Similarity Approximation
Compares the exact float16 inner products against the normalized EvpBits similarity scores.

```bash
uv run python tests/test_similarity.py
```

### 3. Performance Benchmark
Compares the execution speed of a single-threaded Python loop vs. the optimized matrix multiplication approach.

```bash
uv run python tests/test_speed.py
```

## Testing

To run the unit tests:

```bash
uv run python tests/test_evp.py
```
