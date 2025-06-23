import pytest
import numpy as np
from abc import ABC
from unittest.mock import Mock

from torchTextClassifiers.classifiers.base import BaseClassifierConfig, BaseClassifierWrapper


class TestBaseClassifierConfig:
    """Test the BaseClassifierConfig abstract base class."""
    
    def test_cannot_instantiate_abstract_class(self):
        """Test that BaseClassifierConfig cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseClassifierConfig()
    
    def test_concrete_implementation_required_methods(self):
        """Test that concrete implementations must provide required methods."""
        
        class ConcreteConfig(BaseClassifierConfig):
            def __init__(self, value):
                self.value = value
            
            def to_dict(self):
                return {"value": self.value}
            
            @classmethod
            def from_dict(cls, data):
                return cls(data["value"])
        
        # Should work with all methods implemented
        config = ConcreteConfig(42)
        assert config.value == 42
        
        # Test serialization
        config_dict = config.to_dict()
        assert config_dict == {"value": 42}
        
        # Test deserialization
        restored_config = ConcreteConfig.from_dict(config_dict)
        assert restored_config.value == 42
    
    def test_incomplete_implementation_fails(self):
        """Test that incomplete implementations cannot be instantiated."""
        
        class IncompleteConfig(BaseClassifierConfig):
            def to_dict(self):
                return {}
            # Missing from_dict method
        
        with pytest.raises(TypeError):
            IncompleteConfig()


class TestBaseClassifierWrapper:
    """Test the BaseClassifierWrapper abstract base class."""
    
    def test_cannot_instantiate_abstract_class(self):
        """Test that BaseClassifierWrapper cannot be instantiated directly."""
        mock_config = Mock()
        with pytest.raises(TypeError):
            BaseClassifierWrapper(mock_config)
    
    def test_concrete_implementation_initialization(self):
        """Test that concrete implementations can be initialized."""
        
        class ConcreteWrapper(BaseClassifierWrapper):
            def build_tokenizer(self, training_text):
                self.tokenizer = "mock_tokenizer"
            
            def _build_pytorch_model(self):
                self.pytorch_model = "mock_model"
            
            def _check_and_init_lightning(self, **kwargs):
                self.lightning_module = "mock_lightning"
            
            def predict(self, X, **kwargs):
                return np.array([1, 0, 1])
            
            def validate(self, X, Y, **kwargs):
                return 0.85
            
            def create_dataset(self, texts, labels, categorical_variables=None):
                return "mock_dataset"
            
            def create_dataloader(self, dataset, batch_size, num_workers=0, shuffle=True):
                return "mock_dataloader"
            
            def load_best_model(self, checkpoint_path):
                self.trained = True
        
        mock_config = Mock()
        wrapper = ConcreteWrapper(mock_config)
        
        # Test initialization
        assert wrapper.config == mock_config
        assert wrapper.tokenizer is None
        assert wrapper.pytorch_model is None
        assert wrapper.lightning_module is None
        assert wrapper.trained == False
        assert wrapper.device is None
    
    def test_concrete_implementation_methods(self):
        """Test that concrete implementations can use all methods."""
        
        class ConcreteWrapper(BaseClassifierWrapper):
            def build_tokenizer(self, training_text):
                self.tokenizer = f"tokenizer_for_{len(training_text)}_samples"
            
            def _build_pytorch_model(self):
                self.pytorch_model = "pytorch_model"
            
            def _check_and_init_lightning(self, **kwargs):
                self.lightning_module = f"lightning_with_{kwargs}"
            
            def predict(self, X, **kwargs):
                return np.array([1] * len(X))
            
            def validate(self, X, Y, **kwargs):
                return float(np.mean(Y))
            
            def create_dataset(self, texts, labels, categorical_variables=None):
                return {
                    "texts": texts,
                    "labels": labels, 
                    "categorical": categorical_variables
                }
            
            def create_dataloader(self, dataset, batch_size, num_workers=0, shuffle=True):
                return {
                    "dataset": dataset,
                    "batch_size": batch_size,
                    "num_workers": num_workers,
                    "shuffle": shuffle
                }
            
            def load_best_model(self, checkpoint_path):
                self.trained = True
                self.pytorch_model = f"model_from_{checkpoint_path}"
        
        mock_config = Mock()
        wrapper = ConcreteWrapper(mock_config)
        
        # Test build_tokenizer
        sample_texts = np.array(["text1", "text2", "text3"])
        wrapper.build_tokenizer(sample_texts)
        assert wrapper.tokenizer == "tokenizer_for_3_samples"
        
        # Test _build_pytorch_model
        wrapper._build_pytorch_model()
        assert wrapper.pytorch_model == "pytorch_model"
        
        # Test _check_and_init_lightning
        wrapper._check_and_init_lightning(learning_rate=0.01)
        assert "learning_rate" in str(wrapper.lightning_module)
        
        # Test predict
        X = np.array(["test1", "test2"])
        predictions = wrapper.predict(X)
        np.testing.assert_array_equal(predictions, np.array([1, 1]))
        
        # Test validate
        Y = np.array([0, 1, 1, 0])
        accuracy = wrapper.validate(X, Y)
        assert accuracy == 0.5
        
        # Test create_dataset
        labels = np.array([1, 0])
        dataset = wrapper.create_dataset(sample_texts[:2], labels)
        np.testing.assert_array_equal(dataset["texts"], sample_texts[:2])
        np.testing.assert_array_equal(dataset["labels"], labels)
        assert dataset["categorical"] is None
        
        # Test create_dataset with categorical
        categorical = np.array([[1, 2], [3, 4]])
        dataset_with_cat = wrapper.create_dataset(sample_texts[:2], labels, categorical)
        np.testing.assert_array_equal(dataset_with_cat["categorical"], categorical)
        
        # Test create_dataloader
        dataloader = wrapper.create_dataloader(dataset, batch_size=32, num_workers=4, shuffle=False)
        assert dataloader["dataset"] == dataset
        assert dataloader["batch_size"] == 32
        assert dataloader["num_workers"] == 4
        assert dataloader["shuffle"] == False
        
        # Test load_best_model
        checkpoint_path = "/path/to/checkpoint"
        wrapper.load_best_model(checkpoint_path)
        assert wrapper.trained == True
        assert wrapper.pytorch_model == f"model_from_{checkpoint_path}"
    
    def test_incomplete_implementation_fails(self):
        """Test that incomplete implementations cannot be instantiated."""
        
        class IncompleteWrapper(BaseClassifierWrapper):
            def build_tokenizer(self, training_text):
                pass
            
            def _build_pytorch_model(self):
                pass
            
            def _check_and_init_lightning(self, **kwargs):
                pass
            
            def predict(self, X, **kwargs):
                return np.array([])
            
            # Missing: validate, create_dataset, create_dataloader, load_best_model
        
        mock_config = Mock()
        with pytest.raises(TypeError):
            IncompleteWrapper(mock_config)
    
    def test_method_signatures(self):
        """Test that abstract methods have correct signatures."""
        
        class ConcreteWrapper(BaseClassifierWrapper):
            def build_tokenizer(self, training_text: np.ndarray) -> None:
                pass
            
            def _build_pytorch_model(self) -> None:
                pass
            
            def _check_and_init_lightning(self, **kwargs) -> None:
                pass
            
            def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
                return np.array([])
            
            def validate(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> float:
                return 0.0
            
            def create_dataset(self, texts: np.ndarray, labels: np.ndarray, categorical_variables=None):
                return None
            
            def create_dataloader(self, dataset, batch_size: int, num_workers: int = 0, shuffle: bool = True):
                return None
            
            def load_best_model(self, checkpoint_path: str) -> None:
                pass
        
        # Should be able to instantiate with all methods implemented
        mock_config = Mock()
        wrapper = ConcreteWrapper(mock_config)
        assert wrapper is not None