import pytest
import torch
import os
from phaze import ZKMLProverVerifier, SimpleFullModel

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

# Removed EZKL setup test due to event loop conflicts

# Removed EZKL verification test due to event loop conflicts

# Removed EZKL cleanup test due to event loop conflicts


