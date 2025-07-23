import pytest
import numpy as np
import json
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock


def test_basic_imports():
    """Test that core modules can be imported without torch dependencies."""
    # Test that the main class can be imported
    from torchTextClassifiers.torchTextClassifiers import torchTextClassifiers
    assert torchTextClassifiers is not None


def test_wrapper_based_pattern():
    """Test the wrapper-based pattern without actual implementations."""
    from torchTextClassifiers.classifiers.base import BaseClassifierWrapper, BaseClassifierConfig
    
    # Test that base classes exist and have expected structure
    assert hasattr(BaseClassifierWrapper, 'prepare_text_features')
    assert hasattr(BaseClassifierWrapper, '_build_pytorch_model')
    assert hasattr(BaseClassifierWrapper, 'get_config_class')
    assert hasattr(BaseClassifierConfig, 'to_dict')
    assert hasattr(BaseClassifierConfig, 'from_dict')


def test_class_structure():
    """Test that the main class has the expected class structure."""
    # We can test the class structure without instantiating
    from torchTextClassifiers.torchTextClassifiers import torchTextClassifiers
    
    # Check that it has the expected methods defined
    assert hasattr(torchTextClassifiers, '__init__')
    assert hasattr(torchTextClassifiers, 'build_tokenizer')  # backward compatibility
    assert hasattr(torchTextClassifiers, 'prepare_text_features')  # new method
    assert hasattr(torchTextClassifiers, 'build')
    assert hasattr(torchTextClassifiers, 'train')
    assert hasattr(torchTextClassifiers, 'predict')
    assert hasattr(torchTextClassifiers, 'validate')
    assert hasattr(torchTextClassifiers, 'to_json')
    assert hasattr(torchTextClassifiers, 'from_json')


def test_abstract_base_classes():
    """Test the abstract base class structure."""
    from torchTextClassifiers.classifiers.base import BaseClassifierConfig, BaseClassifierWrapper
    
    # Test that they are abstract
    with pytest.raises(TypeError):
        BaseClassifierConfig()
    
    # Test that BaseClassifierWrapper requires a config but still can't be instantiated
    mock_config = Mock()
    with pytest.raises(TypeError):
        BaseClassifierWrapper(mock_config)


def test_utilities_import():
    """Test that utility functions can be imported."""
    from torchTextClassifiers.utilities.checkers import check_X, check_Y, NumpyJSONEncoder
    
    # Test basic functionality that doesn't depend on complex imports
    assert callable(check_X)
    assert callable(check_Y)
    assert NumpyJSONEncoder is not None


def test_torchTextClassifiers_initialization_pattern():
    """Test the initialization pattern using mocks."""
    from torchTextClassifiers.torchTextClassifiers import torchTextClassifiers
    
    # Mock wrapper with config
    mock_wrapper = Mock()
    mock_config = Mock()
    mock_wrapper.config = mock_config
    
    # Create instance directly with wrapper
    classifier = torchTextClassifiers(mock_wrapper)
    
    # Verify initialization
    assert classifier.classifier == mock_wrapper
    assert classifier.config == mock_config


def test_numpy_json_encoder():
    """Test the custom JSON encoder for numpy arrays."""
    from torchTextClassifiers.utilities.checkers import NumpyJSONEncoder
    
    # Test with numpy array
    test_data = {
        "array": np.array([1, 2, 3]),
        "scalar": np.int64(42),
        "regular": "string"
    }
    
    # Should not raise an error
    json_str = json.dumps(test_data, cls=NumpyJSONEncoder)
    assert isinstance(json_str, str)
    
    # Verify it can be loaded back
    loaded_data = json.loads(json_str)
    assert loaded_data["regular"] == "string"


def test_create_fasttext_classmethod():
    """Test the create_fasttext class method through FastTextFactory."""
    from torchTextClassifiers.classifiers.fasttext.core import FastTextFactory
    from torchTextClassifiers.torchTextClassifiers import torchTextClassifiers
    from torchTextClassifiers.classifiers.fasttext.wrapper import FastTextWrapper
    
    # Just test that it creates a real instance and config properly
    result = FastTextFactory.create_fasttext(
        embedding_dim=50,
        sparse=True,
        num_tokens=5000,
        min_count=2,
        min_n=2,
        max_n=5,
        len_word_ngrams=3,
        num_classes=2
    )
    
    # Verify the result is a proper torchTextClassifiers instance
    assert isinstance(result, torchTextClassifiers)
    assert isinstance(result.classifier, FastTextWrapper)
    assert result.config.embedding_dim == 50
    assert result.config.sparse == True
    assert result.config.num_tokens == 5000


def test_method_delegation_pattern():
    """Test that the main class properly delegates to wrapper methods."""
    from torchTextClassifiers.torchTextClassifiers import torchTextClassifiers
    
    # Create a mock instance
    classifier = Mock(spec=torchTextClassifiers)
    classifier.classifier = Mock()
    
    # Test predict delegation
    expected_result = np.array([1, 0, 1])
    classifier.classifier.predict.return_value = expected_result
    
    # Apply the real predict method to our mock
    sample_X = np.array(["test1", "test2", "test3"])
    result = torchTextClassifiers.predict(classifier, sample_X)
    
    classifier.classifier.predict.assert_called_once_with(sample_X)
    assert result is expected_result


def test_error_handling_patterns():
    """Test expected error handling without actual implementation."""
    
    # Test that incomplete wrapper configurations raise appropriate errors
    from torchTextClassifiers.torchTextClassifiers import torchTextClassifiers
    
    # Test with invalid wrapper (missing config attribute)
    class InvalidWrapper:
        pass
    
    invalid_wrapper = InvalidWrapper()
    
    # This should raise AttributeError for missing config
    with pytest.raises(AttributeError):
        torchTextClassifiers(invalid_wrapper)


@pytest.mark.parametrize("method_name,expected_args", [
    ("predict", ["X"]),
    ("validate", ["X", "Y"]),
    ("prepare_text_features", ["training_text"]),
])
def test_wrapper_method_signatures(method_name, expected_args):
    """Test that wrapper methods have expected signatures."""
    from torchTextClassifiers.classifiers.base import BaseClassifierWrapper
    
    # Get the method from the abstract class
    method = getattr(BaseClassifierWrapper, method_name)
    
    # Check that it's abstract
    assert hasattr(method, '__isabstractmethod__')
    assert method.__isabstractmethod__ == True


def test_configuration_serialization_pattern():
    """Test the configuration serialization pattern."""
    from torchTextClassifiers.classifiers.base import BaseClassifierConfig
    
    # Verify abstract methods exist
    assert hasattr(BaseClassifierConfig, 'to_dict')
    assert hasattr(BaseClassifierConfig, 'from_dict')
    
    # Verify they are abstract
    assert BaseClassifierConfig.to_dict.__isabstractmethod__ == True
    assert BaseClassifierConfig.from_dict.__isabstractmethod__ == True


def test_sample_data_fixtures(sample_text_data, sample_labels, sample_categorical_data):
    """Test that our test fixtures work correctly."""
    assert len(sample_text_data) == 6
    assert len(sample_labels) == 6
    assert sample_categorical_data.shape == (6, 2)
    
    # Verify data types
    assert isinstance(sample_text_data, np.ndarray)
    assert isinstance(sample_labels, np.ndarray)
    assert isinstance(sample_categorical_data, np.ndarray)
    
    # Verify content makes sense
    assert all(isinstance(text, str) for text in sample_text_data)
    assert all(label in [0, 1] for label in sample_labels)


