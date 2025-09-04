import torch
from phaze.early_exit_models import SimpleEarlyExitModel

def test_simple_early_exit_model_output_shape():
    model = SimpleEarlyExitModel(input_dim=10, hidden_dim=5, output_dim=2)
    input_data = torch.randn(1, 10)
    output = model(input_data)
    assert output.shape == (1, 2)

def test_simple_early_exit_model_multiple_inputs():
    model = SimpleEarlyExitModel(input_dim=10, hidden_dim=5, output_dim=2)
    input_data = torch.randn(5, 10) # Batch of 5 inputs
    output = model(input_data)
    assert output.shape == (5, 2)

def test_simple_early_exit_model_has_relu():
    model = SimpleEarlyExitModel()
    assert isinstance(model.relu, torch.nn.ReLU)


