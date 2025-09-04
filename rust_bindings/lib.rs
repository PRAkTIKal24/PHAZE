
use pyo3::prelude::*;

/// Formats the sum of two numbers as a string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// Hashes data using SHA256.
#[pyfunction]
fn hash_data(data: &[u8]) -> PyResult<String> {
    use sha2::{Sha256, Digest};
    let mut hasher = Sha256::new();
    hasher.update(data);
    let result = hasher.finalize();
    Ok(format!("{:x}", result))
}

/// A Python module implemented in Rust.
#[pymodule]
fn rust_zkml_bindings(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)
        .unwrap();
    m.add_function(wrap_pyfunction!(hash_data, m)?)
        .unwrap();
    Ok(())
}


