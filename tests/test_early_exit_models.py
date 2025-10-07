import torch

from phaze import SimpleEarlyExitModel


def test_simple_early_exit_model_init():
    model = SimpleEarlyExitModel()
    assert isinstance(model, torch.nn.Module)


def test_simple_early_exit_model_forward():
    model = SimpleEarlyExitModel()
    input_data = torch.randn(1, 10)
    output = model(input_data)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 5)
