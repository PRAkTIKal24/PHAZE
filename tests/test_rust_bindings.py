import pytest
from phaze.rust_zkml_backend import RustZKMLBackend

def test_sha256_hash_data():
    backend = RustZKMLBackend()
    data = b"hello world"
    expected_hash = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    assert backend.sha256_hash(data) == expected_hash

def test_sha256_hash_empty_data():
    backend = RustZKMLBackend()
    data = b""
    expected_hash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    assert backend.sha256_hash(data) == expected_hash

def test_keccak256_hash_data():
    backend = RustZKMLBackend()
    data = b"hello world"
    expected_hash = "47173285a8d7341e5fe08d5cfa03f2c3c88fcd85403403403403403403403403"
    assert backend.keccak256_hash(data) == expected_hash

def test_keccak256_hash_empty_data():
    backend = RustZKMLBackend()
    data = b""
    expected_hash = "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"
    assert backend.keccak256_hash(data) == expected_hash


