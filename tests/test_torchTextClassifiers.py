import pytest
import numpy as np
import json
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from torchTextClassifiers.torchTextClassifiers import (
    torchTextClassifiers, 
    ClassifierType, 
    ClassifierFactory
)
from torchTextClassifiers.classifiers.fasttext.core import FastTextConfig, FastTextFactory
from torchTextClassifiers.classifiers.fasttext.wrapper import FastTextWrapper


class TestClassifierType:
    """Test the ClassifierType enum."""
    
    def test_fasttext_type_exists(self):
        """Test that FASTTEXT type is available."""
        assert ClassifierType.FASTTEXT.value == "fasttext"


class TestClassifierFactory:
    """Test the ClassifierFactory class."""
    
    def test_create_fasttext_classifier(self, fasttext_config):
        """Test creating FastText classifier through factory."""
        classifier = ClassifierFactory.create_classifier(
            ClassifierType.FASTTEXT, fasttext_config
        )
        assert isinstance(classifier, FastTextWrapper)
        assert classifier.config == fasttext_config
    
    def test_create_unsupported_classifier(self, fasttext_config):
        """Test creating unsupported classifier type raises error."""
        with pytest.raises(ValueError, match="Unsupported classifier type"):
            # Create a mock classifier type that doesn't exist
            mock_type = Mock()
            mock_type.name = "UNSUPPORTED"
            ClassifierFactory.create_classifier(mock_type, fasttext_config)
    
    def test_register_new_classifier(self, fasttext_config):
        """Test registering a new classifier type."""
        # Create a mock classifier type and wrapper
        mock_type = Mock()
        mock_wrapper_class = Mock()
        mock_wrapper_instance = Mock()
        mock_wrapper_class.return_value = mock_wrapper_instance
        
        # Register the new classifier
        ClassifierFactory.register_classifier(mock_type, mock_wrapper_class)
        
        # Test that it can be created
        result = ClassifierFactory.create_classifier(mock_type, fasttext_config)
        assert result == mock_wrapper_instance
        mock_wrapper_class.assert_called_once_with(fasttext_config)


class TestTorchTextClassifiers:
    """Test the main torchTextClassifiers class."""
    
    def test_initialization(self, fasttext_config):
        """Test basic initialization."""
        classifier = torchTextClassifiers(ClassifierType.FASTTEXT, fasttext_config)
        
        assert classifier.classifier_type == ClassifierType.FASTTEXT
        assert classifier.config == fasttext_config
        assert isinstance(classifier.classifier_wrapper, FastTextWrapper)
    
    def test_create_fasttext_classmethod(self):
        """Test the create_fasttext class method."""
        classifier = FastTextFactory.create_fasttext(
            embedding_dim=50,
            sparse=True,
            num_tokens=5000,
            min_count=2,
            min_n=2,
            max_n=5,
            len_word_ngrams=3,
            num_classes=3
        )
        
        assert classifier.classifier_type == ClassifierType.FASTTEXT
        assert classifier.config.embedding_dim == 50
        assert classifier.config.sparse == True
        assert classifier.config.num_tokens == 5000
        assert classifier.config.num_classes == 3
    
    def test_build_from_tokenizer(self, mock_tokenizer):
        """Test building classifier from existing tokenizer."""
        classifier = FastTextFactory.build_from_tokenizer(
            tokenizer=mock_tokenizer,
            embedding_dim=100,
            num_classes=2,
            sparse=False
        )
        
        assert classifier.classifier_type == ClassifierType.FASTTEXT
        assert classifier.config.embedding_dim == 100
        assert classifier.config.num_classes == 2
        assert classifier.classifier_wrapper.tokenizer == mock_tokenizer
    
    def test_build_from_tokenizer_missing_attributes(self):
        """Test build_from_tokenizer with tokenizer missing attributes."""
        class IncompleteTokenizer:
            def __init__(self):
                self.min_count = 1
                # Missing: min_n, max_n, num_tokens, word_ngrams
        
        incomplete_tokenizer = IncompleteTokenizer()
        
        with pytest.raises(ValueError, match="Missing attributes in tokenizer"):
            FastTextFactory.build_from_tokenizer(
                tokenizer=incomplete_tokenizer,
                embedding_dim=100,
                num_classes=2
            )
    
    
    @patch('torchTextClassifiers.torchTextClassifiers.check_X')
    def test_build_tokenizer(self, mock_check_X, fasttext_config, sample_text_data):
        """Test build_tokenizer method."""
        mock_check_X.return_value = (sample_text_data, None, True)
        
        classifier = torchTextClassifiers(ClassifierType.FASTTEXT, fasttext_config)
        classifier.classifier_wrapper.build_tokenizer = Mock()
        
        classifier.build_tokenizer(sample_text_data)
        
        classifier.classifier_wrapper.build_tokenizer.assert_called_once_with(sample_text_data)
    
    @patch('torchTextClassifiers.torchTextClassifiers.check_X')
    @patch('torchTextClassifiers.torchTextClassifiers.check_Y')
    def test_build_method_with_labels(self, mock_check_Y, mock_check_X, fasttext_config, 
                                     sample_text_data, sample_labels):
        """Test build method with training labels."""
        mock_check_X.return_value = (sample_text_data, None, True)
        mock_check_Y.return_value = sample_labels
        
        classifier = torchTextClassifiers(ClassifierType.FASTTEXT, fasttext_config)
        classifier.classifier_wrapper.build_tokenizer = Mock()
        classifier.classifier_wrapper._build_pytorch_model = Mock()
        classifier.classifier_wrapper._check_and_init_lightning = Mock()
        
        classifier.build(sample_text_data, sample_labels)
        
        # Verify methods were called
        classifier.classifier_wrapper.build_tokenizer.assert_called_once()
        classifier.classifier_wrapper._build_pytorch_model.assert_called_once()
        classifier.classifier_wrapper._check_and_init_lightning.assert_called_once()
        
        # Verify num_classes was updated
        assert classifier.config.num_classes == len(np.unique(sample_labels))
    
    @patch('torchTextClassifiers.torchTextClassifiers.check_X')
    def test_build_method_without_labels(self, mock_check_X, sample_text_data):
        """Test build method without training labels."""
        mock_check_X.return_value = (sample_text_data, None, True)
        
        # Config with pre-set num_classes
        config = FastTextConfig(
            embedding_dim=10, sparse=False, num_tokens=1000,
            min_count=1, min_n=3, max_n=6, len_word_ngrams=2,
            num_classes=3  # Pre-set
        )
        
        classifier = torchTextClassifiers(ClassifierType.FASTTEXT, config)
        classifier.classifier_wrapper.build_tokenizer = Mock()
        classifier.classifier_wrapper._build_pytorch_model = Mock()
        classifier.classifier_wrapper._check_and_init_lightning = Mock()
        
        classifier.build(sample_text_data, y_train=None)
        
        # Should not raise error since num_classes is pre-set
        assert classifier.config.num_classes == 3
    
    @patch('torchTextClassifiers.torchTextClassifiers.check_X')
    def test_build_method_no_labels_no_num_classes(self, mock_check_X, fasttext_config, sample_text_data):
        """Test build method fails when no labels and no num_classes."""
        mock_check_X.return_value = (sample_text_data, None, True)
        
        # Config without num_classes
        fasttext_config.num_classes = None
        
        classifier = torchTextClassifiers(ClassifierType.FASTTEXT, fasttext_config)
        
        with pytest.raises(ValueError, match="Either num_classes must be provided"):
            classifier.build(sample_text_data, y_train=None)
    
    @patch('torchTextClassifiers.torchTextClassifiers.check_X')
    @patch('torchTextClassifiers.torchTextClassifiers.check_Y')
    def test_build_invalid_labels_range(self, mock_check_Y, mock_check_X, fasttext_config,
                                       sample_text_data):
        """Test build method with invalid label range."""
        mock_check_X.return_value = (sample_text_data, None, True)
        # Labels with values that don't start from 0 or have gaps (invalid)
        invalid_labels = np.array([0, 1, 5])  # Max value 5 but only 3 unique values, so num_classes=3 but max=5
        mock_check_Y.return_value = invalid_labels
        
        classifier = torchTextClassifiers(ClassifierType.FASTTEXT, fasttext_config)
        
        with pytest.raises(ValueError, match="y_train must contain values between 0 and num_classes-1"):
            classifier.build(sample_text_data, invalid_labels)
    
    @patch('torchTextClassifiers.torchTextClassifiers.check_X')
    @patch('torchTextClassifiers.torchTextClassifiers.check_Y')
    @patch('torch.cuda.is_available')
    @patch('pytorch_lightning.Trainer')
    def test_train_method_basic(self, mock_trainer_class, mock_cuda, mock_check_Y, mock_check_X,
                               fasttext_config, sample_text_data, sample_labels, mock_dataset, mock_dataloader):
        """Test basic train method functionality."""
        # Setup mocks
        mock_check_X.return_value = (sample_text_data, None, True)
        mock_check_Y.return_value = sample_labels
        mock_cuda.return_value = True
        
        mock_trainer = Mock()
        mock_trainer.checkpoint_callback.best_model_path = "/fake/path"
        mock_trainer_class.return_value = mock_trainer
        
        classifier = torchTextClassifiers(ClassifierType.FASTTEXT, fasttext_config)
        
        # Mock wrapper methods
        classifier.classifier_wrapper.create_dataset = Mock(return_value=mock_dataset)
        classifier.classifier_wrapper.create_dataloader = Mock(return_value=mock_dataloader)
        classifier.classifier_wrapper.load_best_model = Mock()
        classifier.classifier_wrapper.tokenizer = Mock()  # Pretend it's built
        classifier.classifier_wrapper.pytorch_model = Mock()
        classifier.classifier_wrapper.pytorch_model.to = Mock(return_value=classifier.classifier_wrapper.pytorch_model)
        classifier.classifier_wrapper.lightning_module = Mock()
        
        # Call train
        classifier.train(
            X_train=sample_text_data,
            y_train=sample_labels,
            X_val=sample_text_data[:3],
            y_val=sample_labels[:3],
            num_epochs=1,
            batch_size=2
        )
        
        # Verify dataset creation
        assert classifier.classifier_wrapper.create_dataset.call_count == 2  # train + val
        assert classifier.classifier_wrapper.create_dataloader.call_count == 2  # train + val
        
        # Verify trainer was called
        mock_trainer.fit.assert_called_once()
        classifier.classifier_wrapper.load_best_model.assert_called_once()
    
    def test_predict_method(self, fasttext_config, sample_text_data):
        """Test predict method."""
        classifier = torchTextClassifiers(ClassifierType.FASTTEXT, fasttext_config)
        classifier.classifier_wrapper.predict = Mock(return_value=np.array([1, 0, 1]))
        
        result = classifier.predict(sample_text_data)
        
        classifier.classifier_wrapper.predict.assert_called_once_with(sample_text_data)
        np.testing.assert_array_equal(result, np.array([1, 0, 1]))
    
    def test_validate_method(self, fasttext_config, sample_text_data, sample_labels):
        """Test validate method."""
        classifier = torchTextClassifiers(ClassifierType.FASTTEXT, fasttext_config)
        classifier.classifier_wrapper.validate = Mock(return_value=0.85)
        
        result = classifier.validate(sample_text_data, sample_labels)
        
        classifier.classifier_wrapper.validate.assert_called_once_with(sample_text_data, sample_labels)
        assert result == 0.85
    
    def test_predict_and_explain_method(self, fasttext_config, sample_text_data):
        """Test predict_and_explain method."""
        classifier = torchTextClassifiers(ClassifierType.FASTTEXT, fasttext_config)
        expected_predictions = np.array([1, 0, 1])
        expected_explanations = np.array([0.8, 0.2, 0.9])
        classifier.classifier_wrapper.predict_and_explain = Mock(
            return_value=(expected_predictions, expected_explanations)
        )
        
        predictions, explanations = classifier.predict_and_explain(sample_text_data)
        
        classifier.classifier_wrapper.predict_and_explain.assert_called_once_with(sample_text_data)
        np.testing.assert_array_equal(predictions, expected_predictions)
        np.testing.assert_array_equal(explanations, expected_explanations)
    
    def test_predict_and_explain_not_supported(self, fasttext_config, sample_text_data):
        """Test predict_and_explain when not supported by wrapper."""
        
        # Create a mock wrapper class that doesn't have predict_and_explain
        class MockWrapperWithoutExplain:
            pass
        
        classifier = torchTextClassifiers(ClassifierType.FASTTEXT, fasttext_config)
        classifier.classifier_wrapper = MockWrapperWithoutExplain()
        
        with pytest.raises(NotImplementedError, match="Explanation not supported"):
            classifier.predict_and_explain(sample_text_data)
    
    def test_to_json_method(self, fasttext_config):
        """Test to_json serialization method."""
        classifier = torchTextClassifiers(ClassifierType.FASTTEXT, fasttext_config)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            classifier.to_json(temp_path)
            
            # Verify file was created and has correct content
            assert os.path.exists(temp_path)
            
            with open(temp_path, 'r') as f:
                data = json.load(f)
            
            assert data['classifier_type'] == 'fasttext'
            assert 'config' in data
            assert data['config']['embedding_dim'] == fasttext_config.embedding_dim
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_from_json_method(self, fasttext_config):
        """Test from_json deserialization method."""
        # First create a JSON file
        original_classifier = torchTextClassifiers(ClassifierType.FASTTEXT, fasttext_config)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            original_classifier.to_json(temp_path)
            
            # Load from JSON
            loaded_classifier = torchTextClassifiers.from_json(temp_path)
            
            assert loaded_classifier.classifier_type == ClassifierType.FASTTEXT
            assert loaded_classifier.config.embedding_dim == fasttext_config.embedding_dim
            assert loaded_classifier.config.sparse == fasttext_config.sparse
            assert loaded_classifier.config.num_tokens == fasttext_config.num_tokens
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_from_json_unsupported_type(self):
        """Test from_json with unsupported classifier type."""
        # Create a JSON with unsupported type
        fake_data = {
            "classifier_type": "unsupported_type",
            "config": {"some": "config"}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(fake_data, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Unsupported classifier type"):
                torchTextClassifiers.from_json(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)