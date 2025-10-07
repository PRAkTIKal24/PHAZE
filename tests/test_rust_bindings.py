from phaze import RustZKMLBackend


def test_sha256_hash_data():
    backend = RustZKMLBackend()
    test_data = b"hello world"
    result = backend.sha256_hash(test_data)
    # Test that it returns a valid 64-character hex string
    assert isinstance(result, str)
    assert len(result) == 64
    assert all(c in "0123456789abcdef" for c in result)


def test_sha256_hash_empty_data():
    backend = RustZKMLBackend()
    data = b""
    expected_hash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    assert backend.sha256_hash(data) == expected_hash


def test_keccak256_hash_data():
    backend = RustZKMLBackend()
    test_data = b"hello world"
    result = backend.keccak256_hash(test_data)
    # Test that it returns a valid 64-character hex string
    assert isinstance(result, str)
    assert len(result) == 64
    assert all(c in "0123456789abcdef" for c in result)


def test_keccak256_hash_empty_data():
    backend = RustZKMLBackend()
    data = b""
    expected_hash = "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"
    assert backend.keccak256_hash(data) == expected_hash
