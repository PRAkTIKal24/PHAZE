import torch
import torch.nn as nn
import os
import json
import asyncio
import ezkl
import numpy as np

class MinimalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

async def test_ezkl_functions():
    model = MinimalModel()
    input_data = torch.randn(1, 10)
    model_name = model.__class__.__name__

    onnx_path = f"/tmp/{model_name}.onnx"
    compiled_model_path = f"/tmp/{model_name}.compiled"
    pk_path = f"/tmp/{model_name}.pk"
    vk_path = f"/tmp/{model_name}.vk"
    settings_path = f"/tmp/{model_name}_settings.json"
    witness_path = f"/tmp/{model_name}.witness.json"
    proof_path = f"/tmp/{model_name}.proof"
    input_json_path = f"/tmp/{model_name}_input.json"
    output_json_path = f"/tmp/{model_name}_output.json"
    srs_path = os.path.join(os.path.expanduser("~"), ".ezkl", "srs", "kzg17.srs")

    # Cleanup previous runs
    for path in [onnx_path, compiled_model_path, pk_path, vk_path, settings_path, witness_path, proof_path, input_json_path, output_json_path]:
        if os.path.exists(path):
            os.remove(path)

    print("--- Testing _export_to_onnx ---")
    torch.onnx.export(model, input_data, onnx_path,
                       opset_version=11,
                       do_constant_folding=True,
                       input_names=["input"],
                       output_names=["output"],
                       dynamic_axes={
                           "input": {0: "batch_size"},
                           "output": {0: "batch_size"}
                       })
    print(f"ONNX model exported to {onnx_path}")

    print("--- Testing gen_settings ---")
    ezkl.gen_settings(onnx_path, settings_path)
    print(f"Settings generated at {settings_path}")

    print("--- Testing get_srs ---")
    if not os.path.exists(srs_path):
        os.makedirs(os.path.dirname(srs_path), exist_ok=True)
        await ezkl.get_srs(srs_path=srs_path, settings_path=settings_path)
        print(f"SRS downloaded to {srs_path}")
    else:
        print(f"SRS already exists at {srs_path}")

    print("--- Testing compile_circuit ---")
    ezkl.compile_circuit(onnx_path, compiled_model_path, settings_path)
    print(f"Circuit compiled to {compiled_model_path}")

    print("--- Testing setup ---")
    ezkl.setup(compiled_model_path, vk_path, pk_path, srs_path=srs_path)
    print(f"Proving and verification keys generated at {pk_path} and {vk_path}")

    print("--- Testing gen_witness ---")
    input_array = (input_data.detach().numpy() * 2**15).astype(np.int64)
    data = dict(input_data = [input_array.flatten().tolist()])
    with open(input_json_path, "w") as f:
        json.dump(data, f)
    ezkl.gen_witness(input_json_path, compiled_model_path, witness_path)
    print(f"Witness generated at {witness_path}")

    print("--- Testing prove ---")
    proof = ezkl.prove(witness_path, compiled_model_path, pk_path, proof_path, "single")
    print(f"Proof generated at {proof_path}")
    print(f"Proof: {proof}")

    print("--- Testing verify ---")
    is_valid = ezkl.verify(proof_path, settings_path, vk_path, srs_path=srs_path)
    print(f"Proof verification result: {is_valid}")

    print("--- Cleanup ---")
    for path in [onnx_path, compiled_model_path, pk_path, vk_path, settings_path, witness_path, proof_path, input_json_path, output_json_path]:
        if os.path.exists(path):
            os.remove(path)
    print("Cleanup complete.")

if __name__ == "__main__":
    asyncio.run(test_ezkl_functions())

