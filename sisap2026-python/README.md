# SISAP 2026 Python

Python environment for inspecting the SISAP 2026 HDF5 dataset.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) installed
- HuggingFace token (for private repo access, optional)

## Setup

```bash
cd sisap2026-python
uv sync
```

## Usage

To inspect the HDF5 file structure:

```bash
uv run python inspect_h5.py
```

If not already cached, `hf_hub_download` will automatically download
`benchmark-dev-wikipedia-bge-m3-small.h5` from the `SISAP-Challenges/SISAP2026` repository.

To run the `debug_phase2` python equivalent (calculates EvpBits similarities and inner products like the Rust implementation):

```bash
uv run python debug_phase2.py
```
This will automatically download/locate the same HDF5 dataset. You can configure `FILENAME`, `CHUNK_SIZE`, and `NON_ZEROS` directly at the top of the script if needed.
