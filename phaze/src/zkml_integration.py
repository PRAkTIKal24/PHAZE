import asyncio
import json
import os

import numpy as np
import onnx  # noqa: F401
import onnxruntime as ort  # noqa: F401
import torch
import torch.nn as nn

try:
    import ezkl

    EZKL_AVAILABLE = True
except ImportError:
    print("ezkl not found. Please install ezkl to enable full ZKML functionality.")
    EZKL_AVAILABLE = False


class ZKMLProverVerifier:
    """
    A class for integrating and testing ZK-SNARK based ZKML systems using ezkl.

    This class provides a high-level interface for:
    1. Compiling a PyTorch model to a zk-SNARK circuit.
    2. Generating a zero-knowledge proof of a model inference.
    3. Verifying the proof.
    """

    def __init__(self, model: nn.Module, zkml_system_name: str = "ezkl"):
        if not EZKL_AVAILABLE:
            raise ImportError(
                "ezkl is not installed. Please install it to use this class."
            )

        self.model = model
        self.zkml_system_name = zkml_system_name
        self.model_name = model.__class__.__name__
        self.onnx_path = f"/tmp/{self.model_name}.onnx"
        self.compiled_model_path = f"/tmp/{self.model_name}.compiled"
        self.pk_path = f"/tmp/{self.model_name}.pk"
        self.vk_path = f"/tmp/{self.model_name}.vk"
        self.settings_path = f"/tmp/{self.model_name}_settings.json"
        self.witness_path = f"/tmp/{self.model_name}.witness.json"
        self.proof_path = f"/tmp/{self.model_name}.proof"
        self.input_json_path = f"/tmp/{self.model_name}_input.json"
        self.output_json_path = f"/tmp/{self.model_name}_output.json"
        self.srs_path = os.path.join(
            os.path.expanduser("~"), ".ezkl", "srs", "kzg17.srs"
        )

    def _export_to_onnx(self, input_data: torch.Tensor):
        """Exports the PyTorch model to ONNX format."""
        torch.onnx.export(
            self.model,
            input_data,
            self.onnx_path,
            opset_version=11,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )

    async def _async_setup(self, input_data: torch.Tensor):
        """Asynchronous part of the setup process."""
        # Generate settings
        ezkl.gen_settings(self.onnx_path, self.settings_path)

        # Download SRS if not exists
        if not os.path.exists(self.srs_path):
            print(f"Downloading SRS to {self.srs_path}...")
            os.makedirs(os.path.dirname(self.srs_path), exist_ok=True)

            # The ezkl.get_srs function can take settings_path directly.
            # Ensure only keyword arguments are used for get_srs
            await ezkl.get_srs(srs_path=self.srs_path, settings_path=self.settings_path)

        # Compile the model
        ezkl.compile_circuit(
            self.onnx_path, self.compiled_model_path, self.settings_path
        )

        # Generate proving and verification keys
        ezkl.setup(
            self.compiled_model_path, self.vk_path, self.pk_path, srs_path=self.srs_path
        )

    def setup(self, input_data: torch.Tensor):
        """Sets up the ZKML system for the model."""
        self._export_to_onnx(input_data)
        # Run the async setup function in a new event loop
        asyncio.run(self._async_setup(input_data))

    def generate_proof(self, input_data: torch.Tensor):
        """Generates a zero-knowledge proof for the model inference."""
        # Prepare input data for ezkl
        input_array = (input_data.detach().numpy() * 2**15).astype(np.int64)
        data = dict(input_data=[input_array.flatten().tolist()])
        with open(self.input_json_path, "w") as f:
            json.dump(data, f)

        # Generate witness
        ezkl.gen_witness(
            self.input_json_path, self.compiled_model_path, self.witness_path
        )

        # Generate proof
        proof = ezkl.prove(
            self.witness_path,
            self.compiled_model_path,
            self.pk_path,
            self.proof_path,
            "single",
        )

        # Get the model output from the witness
        with open(self.witness_path, "r") as f:
            witness = json.load(f)

            # Prioritize 'pretty_elements' if available, as it contains rescaled outputs
            if (
                "pretty_elements" in witness
                and "rescaled_outputs" in witness["pretty_elements"]
            ):
                rescaled_outputs_list = witness["pretty_elements"]["rescaled_outputs"][
                    0
                ]
                # Ensure all elements are floats before creating the tensor
                rescaled_outputs_float = [float(x) for x in rescaled_outputs_list]
                model_output = torch.tensor(rescaled_outputs_float, dtype=torch.float64)
            elif "outputs" in witness:
                # Fallback to converting hexadecimal strings to integers and
                # then to float64
                outputs_hex = witness["outputs"][0]
                outputs_float = []
                for hex_str in outputs_hex:
                    # Convert hex string to integer
                    val_int = int(hex_str, 16)
                    # Convert to float, as these are likely large field elements
                    outputs_float.append(float(val_int))
                model_output = (
                    torch.tensor(outputs_float, dtype=torch.float64) / 2**15
                )  # Scale back if fixed-point was used
            elif "output_data" in witness:
                model_output = (
                    torch.tensor(witness["output_data"], dtype=torch.float64) / 2**15
                )  # Scale back if fixed-point was used
            else:
                raise KeyError(
                    "Neither 'outputs', 'output_data' nor 'pretty_elements.rescaled_outputs' found in witness JSON."  # noqa: E501
                )

        return {"model_output": model_output, "proof": proof}

    def verify_proof(self, proof: dict) -> bool:
        """Verifies a zero-knowledge proof."""
        return ezkl.verify(
            self.proof_path, self.settings_path, self.vk_path, srs_path=self.srs_path
        )

    def cleanup(self):
        """Cleans up generated files."""
        for path in [
            self.onnx_path,
            self.compiled_model_path,
            self.pk_path,
            self.vk_path,
            self.settings_path,
            self.witness_path,
            self.proof_path,
            self.input_json_path,
            self.output_json_path,
        ]:
            if os.path.exists(path):
                os.remove(path)


class SimpleFullModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)
