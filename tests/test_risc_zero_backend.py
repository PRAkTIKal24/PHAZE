"""
Tests for RISC Zero backend implementation.
"""

import asyncio
import json
import os
import tempfile
import pytest
import torch
import torch.nn as nn

from phaze.src.zkml_backends import RiscZeroBackend
from phaze.src.zkml_framework_interface import ZKMLFramework


class TestRiscZeroBackend:
    """Test RISC Zero zkML backend."""

    def setup_method(self):
        """Setup for each test method."""
        self.input_size = 10
        self.output_size = 1
        self.model = nn.Linear(self.input_size, self.output_size)
        self.backend = RiscZeroBackend(self.model, "test_risc_zero")
        self.input_data = torch.randn(1, self.input_size)

    def teardown_method(self):
        """Cleanup after each test method."""
        if hasattr(self, 'backend'):
            self.backend.cleanup()

    def test_initialization(self):
        """Test backend initialization."""
        assert self.backend.framework == ZKMLFramework.RISC_ZERO
        assert self.backend.model == self.model
        assert self.backend.name == "test_risc_zero"
        assert not self.backend.is_setup

    def test_framework_info(self):
        """Test framework information."""
        info = self.backend.get_framework_info()
        
        assert info["framework"] == "RISC Zero"
        assert info["version"] == "0.18.0"
        assert info["backend"] == "RISC-V zkVM"
        assert info["proof_system"] == "STARK"
        assert info["guest_support"] == "Rust"
        assert info["curve"] == "RISC-V ISA"
        assert info["status"] == "active_implementation"

    @pytest.mark.asyncio
    async def test_setup(self):
        """Test backend setup."""
        await self.backend.setup(self.input_data)
        
        assert self.backend.is_setup
        assert self.backend.guest_id is not None
        assert self.backend.zkvm_binary_path is not None
        assert os.path.exists(self.backend.model_artifacts_path)
        assert os.path.exists(self.backend.zkvm_binary_path)

    @pytest.mark.asyncio
    async def test_setup_model_artifacts(self):
        """Test model artifacts creation during setup."""
        await self.backend.setup(self.input_data)
        
        # Check model artifacts file
        assert os.path.exists(self.backend.model_artifacts_path)
        
        with open(self.backend.model_artifacts_path, "r") as f:
            model_state = json.load(f)
        
        assert "weights" in model_state
        assert "architecture" in model_state
        assert "input_shape" in model_state
        assert model_state["input_shape"] == list(self.input_data.shape)

    @pytest.mark.asyncio
    async def test_generate_proof(self):
        """Test proof generation."""
        await self.backend.setup(self.input_data)
        
        proof, output = await self.backend.generate_proof(self.input_data)
        
        # Verify proof structure
        assert isinstance(proof, dict)
        assert "receipt" in proof
        assert "proof_system" in proof
        assert proof["proof_system"] == "STARK"
        
        # Verify receipt structure
        receipt = proof["receipt"]
        assert "seal" in receipt
        assert "guest_id" in receipt
        assert "journal_digest" in receipt
        assert receipt["guest_id"] == self.backend.guest_id
        
        # Verify output
        assert isinstance(output, list)
        assert len(output) == self.output_size

    @pytest.mark.asyncio
    async def test_verify_proof_valid(self):
        """Test verification of valid proof."""
        await self.backend.setup(self.input_data)
        proof, _ = await self.backend.generate_proof(self.input_data)
        
        verified = await self.backend.verify_proof(proof, self.input_data)
        assert verified is True

    @pytest.mark.asyncio
    async def test_verify_proof_invalid_structure(self):
        """Test verification of invalid proof structure."""
        await self.backend.setup(self.input_data)
        
        # Test with invalid proof structure
        invalid_proof = {"invalid": "proof"}
        verified = await self.backend.verify_proof(invalid_proof, self.input_data)
        assert verified is False

    @pytest.mark.asyncio
    async def test_verify_proof_wrong_guest_id(self):
        """Test verification with wrong guest ID."""
        await self.backend.setup(self.input_data)
        proof, _ = await self.backend.generate_proof(self.input_data)
        
        # Modify guest_id in proof
        proof["receipt"]["guest_id"] = "wrong_guest_id"
        verified = await self.backend.verify_proof(proof, self.input_data)
        assert verified is False

    @pytest.mark.asyncio
    async def test_verify_proof_invalid_seal(self):
        """Test verification with invalid seal."""
        await self.backend.setup(self.input_data)
        proof, _ = await self.backend.generate_proof(self.input_data)
        
        # Modify seal format
        proof["receipt"]["seal"] = "invalid_seal_format"
        verified = await self.backend.verify_proof(proof, self.input_data)
        assert verified is False

    @pytest.mark.asyncio
    async def test_setup_required_for_operations(self):
        """Test that setup is required before operations."""
        # Test generate_proof without setup
        with pytest.raises(RuntimeError, match="Backend not setup"):
            await self.backend.generate_proof(self.input_data)
        
        # Test verify_proof without setup
        with pytest.raises(RuntimeError, match="Backend not setup"):
            await self.backend.verify_proof({}, self.input_data)

    def test_cleanup(self):
        """Test cleanup functionality."""
        # Create some dummy files to test cleanup
        test_files = [
            self.backend.model_artifacts_path,
            self.backend.proof_path,
            self.backend.journal_path
        ]
        
        for file_path in test_files:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w") as f:
                f.write("test content")
            assert os.path.exists(file_path)
        
        # Call cleanup
        self.backend.cleanup()
        
        # Verify files are removed
        for file_path in test_files:
            assert not os.path.exists(file_path)

    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test complete workflow: setup -> prove -> verify -> cleanup."""
        # Setup
        await self.backend.setup(self.input_data)
        assert self.backend.is_setup
        
        # Generate proof
        proof, output = await self.backend.generate_proof(self.input_data)
        assert proof is not None
        assert output is not None
        
        # Verify proof
        verified = await self.backend.verify_proof(proof, self.input_data)
        assert verified is True
        
        # Cleanup
        self.backend.cleanup()
        assert not os.path.exists(self.backend.model_artifacts_path)

    @pytest.mark.asyncio
    async def test_multiple_inputs(self):
        """Test with different input sizes and batch sizes."""
        # Test with batch size > 1
        batch_input = torch.randn(3, self.input_size)
        await self.backend.setup(batch_input)
        
        proof, output = await self.backend.generate_proof(batch_input)
        verified = await self.backend.verify_proof(proof, batch_input)
        
        assert verified is True
        assert len(output) == 3 * self.output_size  # Flattened output

    @pytest.mark.asyncio
    async def test_model_output_consistency(self):
        """Test that zkVM output matches regular model output."""
        await self.backend.setup(self.input_data)
        
        # Get zkVM output
        _, zkvm_output = await self.backend.generate_proof(self.input_data)
        
        # Get regular model output
        with torch.no_grad():
            regular_output = self.model(self.input_data)
        
        # Compare outputs (allowing for small numerical differences)
        zkvm_tensor = torch.tensor(zkvm_output).reshape(regular_output.shape)
        assert torch.allclose(zkvm_tensor, regular_output, atol=1e-6)

    def test_concurrent_backends(self):
        """Test multiple backend instances don't interfere."""
        backend1 = RiscZeroBackend(self.model, "backend1")
        backend2 = RiscZeroBackend(self.model, "backend2")
        
        # Verify they have different paths
        assert backend1.base_path != backend2.base_path
        assert backend1.guest_id != backend2.guest_id
        
        # Cleanup
        backend1.cleanup()
        backend2.cleanup()

    @pytest.mark.asyncio
    async def test_proof_artifacts_created(self):
        """Test that all expected proof artifacts are created."""
        await self.backend.setup(self.input_data)
        await self.backend.generate_proof(self.input_data)
        
        # Check all proof artifacts exist
        assert os.path.exists(self.backend.proof_path)
        assert os.path.exists(self.backend.journal_path)
        assert os.path.exists(self.backend.receipt_path)
        assert os.path.exists(self.backend.input_data_path)
        
        # Verify file contents are valid JSON
        with open(self.backend.proof_path, "r") as f:
            proof = json.load(f)
        assert isinstance(proof, dict)
        
        with open(self.backend.journal_path, "r") as f:
            journal = json.load(f)
        assert isinstance(journal, dict)
        assert "model_output" in journal