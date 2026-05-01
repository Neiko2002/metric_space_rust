# SISAP 2026

This directory contains the implementation for the SISAP 2026 challenge.

## Prerequisites

To compile and run this code on Windows 11, you need:
- **Rust Toolchain**: Install via [rustup.rs](https://rustup.rs/).
- **C++ Build Tools**: Microsoft Visual C++ Build Tools.
- **HDF5 Libraries**: The project depends on HDF5 for data loading and saving. Ensure HDF5 is installed and available on your system.

## Preparation

For Windows 11, `jemallocator` is disabled to avoid compilation errors. If you are working on another platform, you may need to adjust the `Cargo.toml` and source files accordingly.

## Compilation

Navigate to the workspace root (`C:\Lang\Rust\metric_space_rust`) and run:

```bash
# Debug build
cargo build -p sisap2026

# Release build (significantly faster)
cargo build --release -p sisap2026
```

## Running the Binaries

All commands should be executed from the workspace root. **It is highly recommended to use the `--release` flag for significantly better performance.**

### 1. Hamiltonian Test
Used to test the Hamiltonian machinery.
```powershell
cargo run --release -p sisap2026 --bin hamiltonian_test <INPUT_H5_PATH> <OUTPUT_H5_PATH>
```
**Example:**
```powershell
cargo run --release -p sisap2026 --bin hamiltonian_test "C:\Users\Neiko\.cache\huggingface\hub\datasets--sisap-challenges--SISAP2026\snapshots\67a012fdc69f52b1974e97053dcf47a41ad5eec4\benchmark-dev-wikipedia-bge-m3-small.h5" "C:\Lang\Rust\metric_space_rust\output_hamiltonian.h5"
```

### 2. Challenge 1
The main implementation for the first challenge.
```powershell
cargo run --release -p sisap2026 --bin challenge1 <INPUT_H5_PATH> <OUTPUT_H5_PATH>
```
**Example:**
```powershell
cargo run --release -p sisap2026 --bin challenge1 "C:\Users\Neiko\.cache\huggingface\hub\datasets--sisap-challenges--SISAP2026\snapshots\67a012fdc69f52b1974e97053dcf47a41ad5eec4\benchmark-dev-wikipedia-bge-m3-small.h5" "C:\Lang\Rust\metric_space_rust\output_challenge1.h5"
```

Note: These processes are computationally intensive and may take several minutes to complete.


