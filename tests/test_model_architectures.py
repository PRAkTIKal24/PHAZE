import pytest
import torch
import torch.nn as nn
from phaze import (
    SimpleEarlyExitModel,
    ConvolutionalEarlyExitModel,
    TransformerEarlyExitModel,
    MultiExitModel,
    PHAZEModelFactory,
    ModelComplexity,
    create_simple_early_exit_model,
    create_simple_full_model
)


class TestSimpleEarlyExitModel:
    """Test the SimpleEarlyExitModel."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = SimpleEarlyExitModel()
        assert model.input_size == 10
        assert model.output_size == 5
        assert model.complexity == ModelComplexity.MINIMAL
    
    def test_forward_pass(self):
        """Test forward pass."""
        model = SimpleEarlyExitModel()
        input_data = torch.randn(2, 10)
        
        output = model(input_data)
        assert output.shape == (2, 5)
    
    def test_confidence_estimation(self):
        """Test confidence estimation."""
        model = SimpleEarlyExitModel()
        input_data = torch.randn(2, 10)
        
        confidence = model.get_confidence(input_data)
        assert confidence.shape == (2, 1)
        assert torch.all(confidence >= 0) and torch.all(confidence <= 1)
    
    def test_model_info(self):
        """Test model information."""
        model = SimpleEarlyExitModel()
        info = model.get_model_info()
        
        assert info["architecture"] == "SimpleEarlyExit"
        assert info["complexity"] == "minimal"
        assert info["input_size"] == 10
        assert info["output_size"] == 5
        assert "parameters" in info
        assert "layers" in info
    
    def test_parameter_count(self):
        """Test parameter counting."""
        model = SimpleEarlyExitModel()
        param_count = model.count_parameters()
        
        # Should have parameters from fc1, fc2, and confidence_head
        expected_params = (10 * 20 + 20) + (20 * 5 + 5) + (20 * 1 + 1)  # weights + biases
        assert param_count == expected_params


class TestConvolutionalEarlyExitModel:
    """Test the ConvolutionalEarlyExitModel."""
    
    def test_initialization_minimal(self):
        """Test minimal complexity initialization."""
        model = ConvolutionalEarlyExitModel(complexity=ModelComplexity.MINIMAL)
        assert model.complexity == ModelComplexity.MINIMAL
        assert model.feature_size == 16
    
    def test_initialization_light(self):
        """Test light complexity initialization."""
        model = ConvolutionalEarlyExitModel(complexity=ModelComplexity.LIGHT)
        assert model.complexity == ModelComplexity.LIGHT
        assert model.feature_size == 64
    
    def test_initialization_medium(self):
        """Test medium complexity initialization."""
        model = ConvolutionalEarlyExitModel(complexity=ModelComplexity.MEDIUM)
        assert model.complexity == ModelComplexity.MEDIUM
        assert model.feature_size == 256
    
    def test_forward_pass_image_input(self):
        """Test forward pass with image input."""
        model = ConvolutionalEarlyExitModel(input_channels=3, input_size=32, output_size=10)
        input_data = torch.randn(2, 3, 32, 32)
        
        output = model(input_data)
        assert output.shape == (2, 10)
    
    def test_forward_pass_flattened_input(self):
        """Test forward pass with flattened input."""
        model = ConvolutionalEarlyExitModel(input_channels=3, input_size=32, output_size=10)
        input_data = torch.randn(2, 3 * 32 * 32)  # Flattened
        
        output = model(input_data)
        assert output.shape == (2, 10)
    
    def test_confidence_estimation(self):
        """Test confidence estimation."""
        model = ConvolutionalEarlyExitModel()
        input_data = torch.randn(2, 3, 32, 32)
        
        confidence = model.get_confidence(input_data)
        assert confidence.shape == (2, 1)
        assert torch.all(confidence >= 0) and torch.all(confidence <= 1)


class TestTransformerEarlyExitModel:
    """Test the TransformerEarlyExitModel."""
    
    def test_initialization_different_complexities(self):
        """Test initialization with different complexities."""
        complexities = [
            (ModelComplexity.MINIMAL, 64, 2, 1),
            (ModelComplexity.LIGHT, 128, 4, 2),
            (ModelComplexity.MEDIUM, 256, 8, 4),
            (ModelComplexity.HEAVY, 512, 8, 6)
        ]
        
        for complexity, expected_d_model, expected_nhead, expected_layers in complexities:
            model = TransformerEarlyExitModel(complexity=complexity)
            assert model.d_model == expected_d_model
            assert model.nhead == expected_nhead
            assert model.num_layers == expected_layers
    
    def test_forward_pass_2d_input(self):
        """Test forward pass with 2D input."""
        model = TransformerEarlyExitModel(input_size=128, output_size=10)
        input_data = torch.randn(2, 128)
        
        output = model(input_data)
        assert output.shape == (2, 10)
    
    def test_forward_pass_3d_input(self):
        """Test forward pass with 3D input (sequence)."""
        model = TransformerEarlyExitModel(input_size=128, output_size=10)
        input_data = torch.randn(2, 5, 128)  # batch, seq_len, features
        
        output = model(input_data)
        assert output.shape == (2, 10)
    
    def test_confidence_estimation(self):
        """Test confidence estimation."""
        model = TransformerEarlyExitModel(input_size=128, output_size=10)
        input_data = torch.randn(2, 128)
        
        confidence = model.get_confidence(input_data)
        assert confidence.shape == (2, 1)
        assert torch.all(confidence >= 0) and torch.all(confidence <= 1)


class TestMultiExitModel:
    """Test the MultiExitModel."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = MultiExitModel(input_size=10, output_size=5, complexity=ModelComplexity.LIGHT)
        assert model.num_exits == 3  # Based on light complexity layer structure
    
    def test_forward_pass_default_exit(self):
        """Test forward pass with default exit."""
        model = MultiExitModel()
        input_data = torch.randn(2, 10)
        
        output = model(input_data)
        assert output.shape == (2, 5)
    
    def test_forward_pass_specific_exit(self):
        """Test forward pass with specific exit layer."""
        model = MultiExitModel()
        input_data = torch.randn(2, 10)
        
        # Test different exit layers
        for exit_layer in range(model.num_exits):
            output = model(input_data, exit_layer=exit_layer)
            assert output.shape == (2, 5)
    
    def test_confidence_estimation(self):
        """Test confidence estimation for different exits."""
        model = MultiExitModel()
        input_data = torch.randn(2, 10)
        
        for exit_layer in range(model.num_exits):
            confidence = model.get_confidence(input_data, exit_layer=exit_layer)
            assert confidence.shape == (2, 1)
            assert torch.all(confidence >= 0) and torch.all(confidence <= 1)
    
    def test_get_all_exits(self):
        """Test getting outputs from all exits."""
        model = MultiExitModel()
        input_data = torch.randn(2, 10)
        
        results = model.get_all_exits(input_data)
        assert len(results) == model.num_exits
        
        for output, confidence in results:
            assert output.shape == (2, 5)
            assert confidence.shape == (2, 1)
            assert torch.all(confidence >= 0) and torch.all(confidence <= 1)
    
    def test_adaptive_forward(self):
        """Test adaptive forward pass."""
        model = MultiExitModel()
        input_data = torch.randn(2, 10)
        
        output, exit_used = model.adaptive_forward(input_data, confidence_threshold=0.5)
        assert output.shape == (2, 5)
        assert 0 <= exit_used < model.num_exits
    
    def test_should_exit_early(self):
        """Test early exit decision."""
        model = MultiExitModel()
        input_data = torch.randn(2, 10)
        
        # Test with different thresholds
        model.confidence_threshold = 0.1  # Very low threshold
        should_exit = model.should_exit_early(input_data)
        assert isinstance(should_exit, (bool, torch.Tensor))  # Can be tensor with single bool value
        
        model.confidence_threshold = 0.99  # Very high threshold
        should_exit = model.should_exit_early(input_data)
        assert isinstance(should_exit, (bool, torch.Tensor))


class TestPHAZEModelFactory:
    """Test the PHAZEModelFactory."""
    
    def test_create_early_exit_model_simple(self):
        """Test creating simple early exit model."""
        model = PHAZEModelFactory.create_early_exit_model("simple")
        assert isinstance(model, SimpleEarlyExitModel)
    
    def test_create_early_exit_model_conv(self):
        """Test creating convolutional early exit model."""
        model = PHAZEModelFactory.create_early_exit_model(
            "conv", 
            complexity=ModelComplexity.LIGHT,
            input_channels=3,
            spatial_size=32,
            output_size=10
        )
        assert isinstance(model, ConvolutionalEarlyExitModel)
        assert model.input_channels == 3
    
    def test_create_early_exit_model_transformer(self):
        """Test creating transformer early exit model."""
        model = PHAZEModelFactory.create_early_exit_model(
            "transformer",
            complexity=ModelComplexity.MEDIUM,
            input_size=256,
            output_size=10
        )
        assert isinstance(model, TransformerEarlyExitModel)
        assert model.input_size == 256
    
    def test_create_early_exit_model_multi_exit(self):
        """Test creating multi-exit model."""
        model = PHAZEModelFactory.create_early_exit_model(
            "multi_exit",
            complexity=ModelComplexity.LIGHT
        )
        assert isinstance(model, MultiExitModel)
    
    def test_create_full_model(self):
        """Test creating full model (should be more complex)."""
        early_model = PHAZEModelFactory.create_early_exit_model(
            "simple", 
            complexity=ModelComplexity.LIGHT
        )
        full_model = PHAZEModelFactory.create_full_model(
            "simple", 
            complexity=ModelComplexity.LIGHT
        )
        
        # Full model should have more parameters than early model
        early_params = early_model.count_parameters()
        full_params = full_model.count_parameters()
        assert full_params >= early_params
    
    def test_unknown_architecture(self):
        """Test creating model with unknown architecture."""
        with pytest.raises(ValueError, match="Unknown architecture"):
            PHAZEModelFactory.create_early_exit_model("unknown_arch")
    
    def test_get_available_architectures(self):
        """Test getting available architectures."""
        architectures = PHAZEModelFactory.get_available_architectures()
        expected = ["simple", "conv", "transformer", "multi_exit"]
        assert set(architectures) == set(expected)
    
    def test_get_complexity_levels(self):
        """Test getting complexity levels."""
        complexities = PHAZEModelFactory.get_complexity_levels()
        expected = list(ModelComplexity)
        assert complexities == expected


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_create_simple_early_exit_model(self):
        """Test convenience function for simple early exit model."""
        model = create_simple_early_exit_model(input_size=20, output_size=10)
        assert isinstance(model, SimpleEarlyExitModel)
        assert model.input_size == 20
        assert model.output_size == 10
    
    def test_create_simple_full_model(self):
        """Test convenience function for simple full model."""
        model = create_simple_full_model(input_size=20, output_size=10)
        assert hasattr(model, 'forward')
        
        # Test forward pass
        input_data = torch.randn(2, 20)
        output = model(input_data)
        assert output.shape == (2, 10)


class TestModelComplexity:
    """Test ModelComplexity enum."""
    
    def test_enum_values(self):
        """Test enum values."""
        assert ModelComplexity.MINIMAL.value == "minimal"
        assert ModelComplexity.LIGHT.value == "light"
        assert ModelComplexity.MEDIUM.value == "medium"
        assert ModelComplexity.HEAVY.value == "heavy"
        assert ModelComplexity.EXTREME.value == "extreme"
    
    def test_enum_ordering(self):
        """Test that we can iterate over complexity levels."""
        complexities = list(ModelComplexity)
        assert len(complexities) == 5
        assert ModelComplexity.MINIMAL in complexities
        assert ModelComplexity.EXTREME in complexities

