from pathlib import Path

import h5py
from huggingface_hub import hf_hub_download

H5_FILE = hf_hub_download(
    repo_id="SISAP-Challenges/SISAP2026",
    filename="benchmark-dev-wikipedia-bge-m3-small.h5",
    repo_type="dataset",
)

path = Path(H5_FILE)

with h5py.File(path, "r") as f:

    def print_name(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"  Dataset: {name}")
            print(f"    Shape: {obj.shape}")
            print(f"    Dtype: {obj.dtype}")
        elif isinstance(obj, h5py.Group):
            print(f"  Group: {name}")

    print("=== HDF5 File Structure ===")
    print(f"File: {path}")
    f.visititems(print_name)
