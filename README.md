# PHAZE Framework Proof-of-Concept

This repository contains a proof-of-concept implementation of the PHAZE framework for low latency inference using ML-based triggers at the Large Hadron Collider. The focus is on integrating cryptographic and zero-knowledge machine learning (ZKML) components.

## Project Structure

- `pyproject.toml`: Project configuration and dependency management using `uv`.
- `src/phaze/`:
    - `early_exit_models.py`: Contains toy implementations of early-exit ML models (`M_early`).
    - `crypto_primitives.py`: Implements basic cryptographic primitives like Lagrange interpolation and SHA256 hashing.
    - `zkml_integration.py`: Provides a modular interface for integrating different ZKML systems (currently `ezkl` and a placeholder Rust backend).
    - `rust_zkml_backend.py`: A Python wrapper for Rust-based ZKML functionalities (currently SHA256 hashing via PyO3).
    - `benchmarking.py`: A suite for performance testing and benchmarking of different components.
- `rust_zkml_bindings/`: Contains the Rust project for Python bindings using PyO3.
- `docs/`: Sphinx documentation.
- `tests/`: Unit tests for various modules.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your_username/phaze.git
    cd phaze
    ```

2.  **Install Python dependencies using `uv`:**
    ```bash
    pip install -e .
    ```

3.  **Install Rust and Cargo (if not already installed):**
    ```bash
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o rustup-init.sh
    chmod +x rustup-init.sh
    ./rustup-init.sh -y --no-modify-path
    source $HOME/.cargo/env
    ```

4.  **Install `maturin` (for building Rust-Python bindings):**
    ```bash
    sudo pip install maturin
    ```

5.  **Build the Rust-Python bindings:**
    ```bash
    cd rust_zkml_bindings
    source $HOME/.cargo/env
    /usr/local/lib/python3.11/dist-packages/maturin-1.9.4.data/scripts/maturin build --release
    pip install ./target/wheels/rust_zkml_bindings-0.1.0-cp311-cp311-manylinux_2_34_x86_64.whl --force-reinstall
    cd ..
    ```

## Usage

### Running Benchmarks

To run the comprehensive benchmarking suite for early-exit models, cryptographic primitives, and ZKML systems (ezkl and Rust placeholder):

```bash
python3 src/phaze/benchmarking.py
```

### Running Tests

To run unit tests for the project:

```bash
pytest tests/
```

## Documentation

To build the Sphinx documentation:

```bash
cd docs
make html
```

The generated HTML documentation will be available in `docs/build/html`.

## Modularity

The system is designed to be modular, allowing easy switching between different ZKML implementations. The `ZKMLProverVerifier` class in `src/phaze/zkml_integration.py` can be configured to use different backends.

## Future Work

-   Integrate a full-fledged Rust-based ZKML library (e.g., `ddkang/zkml` or `arkworks-rs/snark`) via PyO3.
-   Implement more sophisticated polynomial interpolation and hashing techniques.
-   Expand the benchmarking suite with more metrics and detailed analysis.
-   Develop more diverse and realistic `M_early` models.


