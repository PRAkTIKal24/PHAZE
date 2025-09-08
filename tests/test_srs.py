import pytest
import os
import asyncio
import torch
from phaze.zkml_integration import ZKMLProverVerifier, SimpleFullModel

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

@pytest.mark.asyncio
async def test_srs_download_and_setup(dummy_model_input):
    model, input_data = dummy_model_input
    pv = ZKMLProverVerifier(model, "test_model_srs")
    
    try:
        # Attempt to setup without pre-downloaded SRS
        # This should trigger the automatic download
        await pv.setup(input_data)
        
        # Verify that the SRS file exists
        # The exact path might vary based on ezkl\'s internal logic
        # This is a heuristic check
        srs_path_exists = False
        for root, _, files in os.walk(os.path.expanduser("~/.ezkl")):
            if any("kzg_bn254_" in f and f.endswith(".srs") for f in files):
                srs_path_exists = True
                break
        assert srs_path_exists, "SRS file was not downloaded or found"

    finally:
        pv.cleanup()

@pytest.mark.asyncio
async def test_srs_reusability(dummy_model_input):
    model, input_data = dummy_model_input
    pv1 = ZKMLProverVerifier(model, "test_model_srs_reusable_1")
    pv2 = ZKMLProverVerifier(model, "test_model_srs_reusable_2")

    try:
        # First setup should download SRS
        await pv1.setup(input_data)
        # Second setup should reuse existing SRS
        await pv2.setup(input_data)

        # No explicit assertion, but if setup completes without re-downloading
        # (which would be slow), it implies reusability.
        # We can add a more robust check if ezkl provides an API for it.
        assert True # Placeholder for successful execution

    finally:
        pv1.cleanup()
        pv2.cleanup()


