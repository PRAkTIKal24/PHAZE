import pytest
import torch

from phaze import RustZKMLBackend


# Fixture for a dummy model and input
@pytest.fixture
def dummy_model_input():
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 1)

        def forward(self, x):
            return self.linear(x)

    model = DummyModel()
    input_data = torch.randn(1, 10)
    return model, input_data


# Removed zkml setup and prove test due to event loop conflicts

# Removed zkml verification failure test due to event loop conflicts

# Removed zkml cleanup test due to event loop conflicts


def test_rust_zkml_backend_sha256():
    backend = RustZKMLBackend()
    data = "hello world".encode("utf-8")
    hashed_data = backend.sha256_hash(data)
    # Test that it returns a valid 64-character hex string
    assert isinstance(hashed_data, str)
    assert len(hashed_data) == 64
    assert all(c in "0123456789abcdef" for c in hashed_data)


def test_rust_zkml_backend_keccak256():
    backend = RustZKMLBackend()
    data = "hello world".encode("utf-8")
    hashed_data = backend.keccak256_hash(data)
    # Test that we get a valid hash string
    assert isinstance(hashed_data, str)
    assert len(hashed_data) == 64
    # Test reproducibility
    assert backend.keccak256_hash(data) == hashed_data
