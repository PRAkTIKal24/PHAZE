"""
Test module for RISC Zero guest and Python data serialization.
"""

import unittest
import torch
import numpy as np
import json
from phaze.src.rust_zkml_backend import RustRiscZeroBackend
from phaze.src.model_architectures import PHAZEModelFactory


class TestRiscZeroSerialization(unittest.TestCase):
    """Test the serialization/deserialization pipeline for RISC Zero."""

    def test_tensor_serialization(self):
        """Test that PyTorch tensors can be serialized and deserialized properly."""
        # Create a simple test tensor
        test_tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        
        # Serialize to list
        serialized = test_tensor.flatten().tolist()
        
        # Deserialize back to tensor
        deserialized = torch.tensor(serialized).reshape(test_tensor.shape)
        
        # Check that they match
        self.assertTrue(torch.allclose(test_tensor, deserialized))
    
    def test_model_weights_serialization(self):
        """Test that model weights can be serialized and deserialized properly."""
        # Create a simple model
        model = PHAZEModelFactory.create_early_exit_model("simple", complexity="light")
        
        # Get the state dict
        state_dict = model.state_dict()
        
        # Simulate serialization process for RISC Zero
        serialized_weights = []
        for weight_tensor in state_dict.values():
            serialized_weights.extend(weight_tensor.flatten().tolist())
        
        # Verify the serialized weights are a flat list of floats
        self.assertIsInstance(serialized_weights, list)
        self.assertTrue(all(isinstance(x, float) for x in serialized_weights))
        
        # In a real implementation, we would deserialize and rebuild the model,
        # but for this test we just verify the serialization format


class TestRiscZeroBackend(unittest.TestCase):
    """Test the RustRiscZeroBackend Python wrapper."""
    
    def setUp(self):
        """Set up the test environment."""
        self.backend = RustRiscZeroBackend()
    
    def test_setup(self):
        """Test the setup method."""
        params = {
            "model_type": "simple",
            "input_size": "10",
            "output_size": "5",
        }
        
        result = self.backend.setup(params)
        self.assertTrue(isinstance(result, str))
        self.assertTrue("RISC Zero setup completed" in result)
        self.assertTrue(self.backend.is_setup)
    
    def test_prove_verify_workflow(self):
        """Test the basic prove/verify workflow."""
        # Setup the backend
        self.backend.setup()
        
        # Create dummy input tensor and model weights
        input_tensor = torch.randn(1, 10)
        model_weights = {
            "layer1.weight": torch.randn(50, 10),
            "layer1.bias": torch.randn(50),
            "layer2.weight": torch.randn(5, 50),
            "layer2.bias": torch.randn(5),
        }
        
        # Generate proof
        proof = self.backend.prove(input_tensor, model_weights)
        
        # Verify the proof structure
        self.assertIn("proof_data", proof)
        self.assertIn("public_inputs", proof)
        self.assertIn("framework", proof)
        self.assertEqual(proof["framework"], "RISC0")
        
        # Verify the proof
        result = self.backend.verify(proof)
        self.assertTrue(result)
    
    def test_config(self):
        """Test getting the backend configuration."""
        config = self.backend.get_config()
        self.assertIsInstance(config, dict)
        self.assertIn("proof_system", config)
        self.assertEqual(config["proof_system"], "STARK")


if __name__ == "__main__":
    unittest.main()