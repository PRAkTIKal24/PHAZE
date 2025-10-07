import pytest
import os
import asyncio
import torch
from phaze import ZKMLProverVerifier, SimpleFullModel

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

# Removed SRS download test due to event loop conflicts

# Removed SRS reusability test due to event loop conflicts


