# SISAP 2026 - Hamiltonian Test

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
cargo build -p sisap2026
```

## Running the Hamiltonian Test

To run the `hamiltonian_test` binary, use the following command from the workspace root:

```powershell
cargo run --bin hamiltonian_test <INPUT_H5_PATH> <OUTPUT_H5_PATH>
```

### Example with local paths:
```powershell
cargo run --bin hamiltonian_test "C:\Users\Neiko\.cache\huggingface\hub\datasets--sisap-challenges--SISAP2026\snapshots\67a012fdc69f52b1974e97053dcf47a41ad5eec4\benchmark-dev-wikipedia-bge-m3-small.h5" "C:\Lang\Rust\metric_space_rust\output.h5"
```

Note: This process is computationally intensive and may take several minutes to complete.
