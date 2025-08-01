import pytest
import numpy as np
import json
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from torchTextClassifiers.torchTextClassifiers import torchTextClassifiers
from torchTextClassifiers.classifiers.fasttext.core import FastTextConfig, FastTextFactory
from torchTextClassifiers.classifiers.fasttext.wrapper import FastTextWrapper




class TestTorchTextClassifiers:
    """Test the main torchTextClassifiers class."""
    
    def test_initialization(self, fasttext_config):
        """Test basic initialization."""
        wrapper = FastTextWrapper(fasttext_config)
        classifier = torchTextClassifiers(wrapper)
        
        assert classifier.config == fasttext_config
        assert isinstance(classifier.classifier, FastTextWrapper)
        assert classifier.classifier is wrapper
    
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
        
        assert isinstance(classifier.classifier, FastTextWrapper)
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
        
        assert isinstance(classifier.classifier, FastTextWrapper)
        assert classifier.config.embedding_dim == 100
        assert classifier.config.num_classes == 2
        assert classifier.classifier.tokenizer == mock_tokenizer
    
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
        
        wrapper = FastTextWrapper(fasttext_config)
        classifier = torchTextClassifiers(wrapper)
        classifier.classifier.prepare_text_features = Mock()
        
        classifier.build_tokenizer(sample_text_data)
        
        classifier.classifier.prepare_text_features.assert_called_once_with(sample_text_data)
    
    @patch('torchTextClassifiers.torchTextClassifiers.check_X')
    @patch('torchTextClassifiers.torchTextClassifiers.check_Y')
    def test_build_method_with_labels(self, mock_check_Y, mock_check_X, fasttext_config, 
                                     sample_text_data, sample_labels):
        """Test build method with training labels."""
        mock_check_X.return_value = (sample_text_data, None, True)
        mock_check_Y.return_value = sample_labels
        
        wrapper = FastTextWrapper(fasttext_config)
        classifier = torchTextClassifiers(wrapper)
        classifier.classifier.prepare_text_features = Mock()
        classifier.classifier._build_pytorch_model = Mock()
        classifier.classifier._check_and_init_lightning = Mock()
        
        classifier.build(sample_text_data, sample_labels)
        
        # Verify methods were called
        classifier.classifier.prepare_text_features.assert_called_once()
        classifier.classifier._build_pytorch_model.assert_called_once()
        classifier.classifier._check_and_init_lightning.assert_called_once()
        
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
        
        wrapper = FastTextWrapper(config)
        classifier = torchTextClassifiers(wrapper)
        classifier.classifier.prepare_text_features = Mock()
        classifier.classifier._build_pytorch_model = Mock()
        classifier.classifier._check_and_init_lightning = Mock()
        
        classifier.build(sample_text_data, y_train=None)
        
        # Should not raise error since num_classes is pre-set
        assert classifier.config.num_classes == 3
    
    @patch('torchTextClassifiers.torchTextClassifiers.check_X')
    def test_build_method_no_labels_no_num_classes(self, mock_check_X, fasttext_config, sample_text_data):
        """Test build method fails when no labels and no num_classes."""
        mock_check_X.return_value = (sample_text_data, None, True)
        
        # Config without num_classes
        fasttext_config.num_classes = None
        
        wrapper = FastTextWrapper(fasttext_config)
        classifier = torchTextClassifiers(wrapper)
        
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
        
        wrapper = FastTextWrapper(fasttext_config)
        classifier = torchTextClassifiers(wrapper)
        
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
        
        wrapper = FastTextWrapper(fasttext_config)
        classifier = torchTextClassifiers(wrapper)
        
        # Mock wrapper methods
        classifier.classifier.create_dataset = Mock(return_value=mock_dataset)
        classifier.classifier.create_dataloader = Mock(return_value=mock_dataloader)
        classifier.classifier.load_best_model = Mock()
        classifier.classifier.tokenizer = Mock()  # Pretend it's built
        classifier.classifier.pytorch_model = Mock()
        classifier.classifier.pytorch_model.to = Mock(return_value=classifier.classifier.pytorch_model)
        classifier.classifier.lightning_module = Mock()
        
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
        assert classifier.classifier.create_dataset.call_count == 2  # train + val
        assert classifier.classifier.create_dataloader.call_count == 2  # train + val
        
        # Verify trainer was called
        mock_trainer.fit.assert_called_once()
        classifier.classifier.load_best_model.assert_called_once()
    
    def test_predict_method(self, fasttext_config, sample_text_data):
        """Test predict method."""
        wrapper = FastTextWrapper(fasttext_config)
        classifier = torchTextClassifiers(wrapper)
        classifier.classifier.predict = Mock(return_value=np.array([1, 0, 1]))
        
        result = classifier.predict(sample_text_data)
        
        classifier.classifier.predict.assert_called_once_with(sample_text_data)
        np.testing.assert_array_equal(result, np.array([1, 0, 1]))
    
    def test_validate_method(self, fasttext_config, sample_text_data, sample_labels):
        """Test validate method."""
        wrapper = FastTextWrapper(fasttext_config)
        classifier = torchTextClassifiers(wrapper)
        classifier.classifier.validate = Mock(return_value=0.85)
        
        result = classifier.validate(sample_text_data, sample_labels)
        
        classifier.classifier.validate.assert_called_once_with(sample_text_data, sample_labels)
        assert result == 0.85
    
    def test_predict_and_explain_method(self, fasttext_config, sample_text_data):
        """Test predict_and_explain method."""
        wrapper = FastTextWrapper(fasttext_config)
        classifier = torchTextClassifiers(wrapper)
        expected_predictions = np.array([1, 0, 1])
        expected_explanations = np.array([0.8, 0.2, 0.9])
        classifier.classifier.predict_and_explain = Mock(
            return_value=(expected_predictions, expected_explanations)
        )
        
        predictions, explanations = classifier.predict_and_explain(sample_text_data)
        
        classifier.classifier.predict_and_explain.assert_called_once_with(sample_text_data)
        np.testing.assert_array_equal(predictions, expected_predictions)
        np.testing.assert_array_equal(explanations, expected_explanations)
    
    def test_predict_and_explain_not_supported(self, fasttext_config, sample_text_data):
        """Test predict_and_explain when not supported by wrapper."""
        
        # Create a mock wrapper class that doesn't have predict_and_explain
        class MockWrapperWithoutExplain:
            pass
        
        wrapper = FastTextWrapper(fasttext_config)
        classifier = torchTextClassifiers(wrapper)
        classifier.classifier = MockWrapperWithoutExplain()
        
        with pytest.raises(NotImplementedError, match="Explanation not supported"):
            classifier.predict_and_explain(sample_text_data)
    
    def test_to_json_method(self, fasttext_config):
        """Test to_json serialization method."""
        wrapper = FastTextWrapper(fasttext_config)
        classifier = torchTextClassifiers(wrapper)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            classifier.to_json(temp_path)
            
            # Verify file was created and has correct content
            assert os.path.exists(temp_path)
            
            with open(temp_path, 'r') as f:
                data = json.load(f)
            
            assert 'wrapper_class_info' in data
            assert 'config' in data
            assert data['config']['embedding_dim'] == fasttext_config.embedding_dim
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_from_json_method(self, fasttext_config):
        """Test from_json deserialization method."""
        # First create a JSON file
        wrapper = FastTextWrapper(fasttext_config)
        original_classifier = torchTextClassifiers(wrapper)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            original_classifier.to_json(temp_path)
            
            # Load from JSON
            loaded_classifier = torchTextClassifiers.from_json(temp_path)
            
            assert isinstance(loaded_classifier.classifier, FastTextWrapper)
            assert loaded_classifier.config.embedding_dim == fasttext_config.embedding_dim
            assert loaded_classifier.config.sparse == fasttext_config.sparse
            assert loaded_classifier.config.num_tokens == fasttext_config.num_tokens
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_from_json_missing_wrapper_info(self):
        """Test from_json with missing wrapper class info."""
        # Create a JSON without wrapper_class_info
        fake_data = {
            "config": {"embedding_dim": 50, "sparse": False, "num_tokens": 1000,
                      "min_count": 1, "min_n": 3, "max_n": 6, "len_word_ngrams": 2}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(fake_data, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="No wrapper_class_info found"):
                torchTextClassifiers.from_json(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_from_json_with_explicit_wrapper_class(self, fasttext_config):
        """Test from_json with explicitly provided wrapper class."""
        # First create a JSON file
        wrapper = FastTextWrapper(fasttext_config)
        original_classifier = torchTextClassifiers(wrapper)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            original_classifier.to_json(temp_path)
            
            # Load from JSON with explicit wrapper class
            loaded_classifier = torchTextClassifiers.from_json(temp_path, FastTextWrapper)
            
            assert isinstance(loaded_classifier.classifier, FastTextWrapper)
            assert loaded_classifier.config.embedding_dim == fasttext_config.embedding_dim
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)