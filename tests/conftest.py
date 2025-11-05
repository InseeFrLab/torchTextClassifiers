from unittest.mock import Mock

import numpy as np
import pytest


@pytest.fixture
def sample_text_data():
    """Sample text data for testing."""
    return np.array(
        [
            "This is a positive example",
            "This is a negative example",
            "Another positive case",
            "Another negative case",
            "Good example here",
            "Bad example here",
        ]
    )


@pytest.fixture
def sample_labels():
    """Sample labels for testing."""
    return np.array([1, 0, 1, 0, 1, 0])


@pytest.fixture
def sample_categorical_data():
    """Sample categorical data for testing."""
    return np.array([[1, 2], [2, 1], [1, 3], [3, 1], [2, 2], [3, 3]])


@pytest.fixture
def sample_X_with_categorical(sample_text_data, sample_categorical_data):
    """Sample X data with categorical variables."""
    return np.column_stack([sample_text_data, sample_categorical_data])


@pytest.fixture
def sample_X_text_only(sample_text_data):
    """Sample X data with text only."""
    return sample_text_data.reshape(-1, 1)


@pytest.fixture
def model_config():
    """Mock model configuration."""
    from torchTextClassifiers import ModelConfig

    config = ModelConfig(
        embedding_dim=10,
        categorical_vocabulary_sizes=[4, 5],
        categorical_embedding_dims=[3, 4],
        num_classes=10,
    )
    return config


@pytest.fixture
def mock_tokenizer():
    """Mock BaseTokenizer for testing."""
    tokenizer = Mock()
    tokenizer.vocab_size = 1000
    tokenizer.padding_idx = 1
    tokenizer.tokenize = Mock(
        return_value={
            "input_ids": np.array([[1, 2, 3], [4, 5, 6]]),
            "attention_mask": np.array([[1, 1, 1], [1, 1, 1]]),
        }
    )
    tokenizer.output_dim = 50
    return tokenizer


@pytest.fixture
def mock_pytorch_model():
    """Mock PyTorch model for testing."""
    model = Mock()
    model.eval = Mock()
    model.to = Mock(return_value=model)
    model.predict = Mock(return_value=np.array([1, 0, 1]))
    model.predict_and_explain = Mock(return_value=(np.array([1, 0, 1]), np.array([0.8, 0.2, 0.9])))
    return model


@pytest.fixture
def mock_lightning_module():
    """Mock Lightning module for testing."""
    module = Mock()
    module.model = Mock()
    return module


@pytest.fixture
def mock_dataset():
    """Mock dataset for testing."""
    dataset = Mock()
    dataset.create_dataloader = Mock()
    return dataset


@pytest.fixture
def mock_dataloader():
    """Mock dataloader for testing."""
    return Mock()
