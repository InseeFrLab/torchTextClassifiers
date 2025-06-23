import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock

from torchTextClassifiers.classifiers.fasttext.wrapper import FastTextWrapper
from torchTextClassifiers.classifiers.fasttext.core import FastTextConfig
from torchTextClassifiers.classifiers.fasttext.tokenizer import NGramTokenizer
from torchTextClassifiers.classifiers.fasttext.model import FastTextModelDataset, FastTextModel, FastTextModule


class TestFastTextConfig:
    """Test FastTextConfig class."""
    
    def test_config_creation(self):
        """Test basic config creation."""
        config = FastTextConfig(
            embedding_dim=100,
            sparse=True,
            num_tokens=5000,
            min_count=2,
            min_n=3,
            max_n=6,
            len_word_ngrams=2,
            num_classes=3
        )
        
        assert config.embedding_dim == 100
        assert config.sparse == True
        assert config.num_tokens == 5000
        assert config.min_count == 2
        assert config.min_n == 3
        assert config.max_n == 6
        assert config.len_word_ngrams == 2
        assert config.num_classes == 3
    
    def test_config_to_dict(self):
        """Test config serialization to dictionary."""
        config = FastTextConfig(
            embedding_dim=100,
            sparse=False,
            num_tokens=1000,
            min_count=1,
            min_n=3,
            max_n=6,
            len_word_ngrams=2
        )
        
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict['embedding_dim'] == 100
        assert config_dict['sparse'] == False
        assert config_dict['num_tokens'] == 1000
    
    def test_config_from_dict(self):
        """Test config deserialization from dictionary."""
        config_dict = {
            'embedding_dim': 50,
            'sparse': True,
            'num_tokens': 2000,
            'min_count': 3,
            'min_n': 2,
            'max_n': 5,
            'len_word_ngrams': 3,
            'num_classes': 4
        }
        
        config = FastTextConfig.from_dict(config_dict)
        
        assert config.embedding_dim == 50
        assert config.sparse == True
        assert config.num_tokens == 2000
        assert config.num_classes == 4


class TestFastTextWrapper:
    """Test FastTextWrapper class."""
    
    def test_wrapper_initialization(self, fasttext_config):
        """Test wrapper initialization."""
        wrapper = FastTextWrapper(fasttext_config)
        
        assert wrapper.config == fasttext_config
        assert wrapper.tokenizer is None
        assert wrapper.pytorch_model is None
        assert wrapper.lightning_module is None
        assert wrapper.trained == False
        assert wrapper.device is None
    
    @patch('torchTextClassifiers.classifiers.fasttext.wrapper.NGramTokenizer')
    def test_build_tokenizer(self, mock_tokenizer_class, fasttext_config, sample_text_data):
        """Test tokenizer building."""
        mock_tokenizer = Mock()
        mock_tokenizer_class.return_value = mock_tokenizer
        
        wrapper = FastTextWrapper(fasttext_config)
        wrapper.build_tokenizer(sample_text_data)
        
        mock_tokenizer_class.assert_called_once_with(
            fasttext_config.min_count,
            fasttext_config.min_n,
            fasttext_config.max_n,
            fasttext_config.num_tokens,
            fasttext_config.len_word_ngrams,
            sample_text_data
        )
        assert wrapper.tokenizer == mock_tokenizer
    
    @patch('torchTextClassifiers.classifiers.fasttext.wrapper.FastTextModel')
    def test_build_pytorch_model_with_tokenizer(self, mock_model_class, fasttext_config, mock_tokenizer):
        """Test PyTorch model building with existing tokenizer."""
        mock_model = Mock()
        mock_model_class.return_value = mock_model
        
        wrapper = FastTextWrapper(fasttext_config)
        wrapper.tokenizer = mock_tokenizer
        
        wrapper._build_pytorch_model()
        
        # Verify model was created with correct parameters
        mock_model_class.assert_called_once()
        call_kwargs = mock_model_class.call_args[1]
        assert call_kwargs['tokenizer'] == mock_tokenizer
        assert call_kwargs['embedding_dim'] == fasttext_config.embedding_dim
        assert call_kwargs['num_classes'] == fasttext_config.num_classes
        
        assert wrapper.pytorch_model == mock_model
    
    def test_build_pytorch_model_no_tokenizer_no_num_rows(self, fasttext_config):
        """Test PyTorch model building fails without tokenizer or num_rows."""
        wrapper = FastTextWrapper(fasttext_config)
        # No tokenizer and no num_rows in config
        wrapper.tokenizer = None
        wrapper.config.num_rows = None
        
        with pytest.raises(ValueError, match="Please provide a tokenizer or num_rows"):
            wrapper._build_pytorch_model()
    
    @patch('torch.optim.lr_scheduler.ReduceLROnPlateau')
    @patch('torchTextClassifiers.classifiers.fasttext.wrapper.FastTextModule')
    def test_check_and_init_lightning_basic(self, mock_module_class, mock_scheduler, fasttext_config, mock_pytorch_model):
        """Test Lightning module initialization."""
        mock_module = Mock()
        mock_module_class.return_value = mock_module
        
        wrapper = FastTextWrapper(fasttext_config)
        wrapper.pytorch_model = mock_pytorch_model
        
        wrapper._check_and_init_lightning(lr=0.01)
        
        # Verify Lightning module was created
        mock_module_class.assert_called_once()
        assert wrapper.lightning_module == mock_module
        assert wrapper.optimizer_params == {"lr": 0.01}
    
    @patch('torchTextClassifiers.classifiers.fasttext.wrapper.FastTextModule')
    def test_check_and_init_lightning_uses_config_lr(self, mock_module_class, fasttext_config, mock_pytorch_model):
        """Test Lightning module initialization uses config learning rate as default."""
        wrapper = FastTextWrapper(fasttext_config)
        wrapper.pytorch_model = mock_pytorch_model
        mock_module = Mock()
        mock_module_class.return_value = mock_module
        
        # Should not raise an error since learning_rate is in config
        wrapper._check_and_init_lightning()
        
        # Check that the learning rate from config was used
        assert wrapper.optimizer_params['lr'] == fasttext_config.learning_rate
        assert wrapper.lightning_module == mock_module
    
    @patch('torchTextClassifiers.classifiers.fasttext.wrapper.check_X')
    def test_predict_not_trained(self, mock_check_X, fasttext_config, sample_text_data):
        """Test prediction fails when model not trained."""
        wrapper = FastTextWrapper(fasttext_config)
        wrapper.trained = False
        
        with pytest.raises(Exception, match="Model must be trained first"):
            wrapper.predict(sample_text_data)
    
    @patch('torchTextClassifiers.classifiers.fasttext.wrapper.check_X')
    def test_predict_success(self, mock_check_X, fasttext_config, sample_text_data, mock_pytorch_model):
        """Test successful prediction."""
        mock_check_X.return_value = (sample_text_data, None, True)
        expected_predictions = np.array([[1], [0], [1]])  # With top_k dimension
        expected_confidence = np.array([[0.9], [0.8], [0.95]])
        mock_pytorch_model.predict.return_value = (expected_predictions, expected_confidence)
        mock_pytorch_model.no_cat_var = True
        
        wrapper = FastTextWrapper(fasttext_config)
        wrapper.trained = True
        wrapper.pytorch_model = mock_pytorch_model
        wrapper.config.num_categorical_features = None
        
        result = wrapper.predict(sample_text_data)
        
        mock_pytorch_model.predict.assert_called_once()
        # The wrapper should squeeze the top_k dimension for top_k=1
        expected_result = np.array([1, 0, 1])
        np.testing.assert_array_equal(result, expected_result)
    
    @patch('torchTextClassifiers.classifiers.fasttext.wrapper.FastTextWrapper.predict')
    @patch('torchTextClassifiers.classifiers.fasttext.wrapper.check_Y')
    def test_validate_success(self, mock_check_Y, mock_predict, fasttext_config, 
                             sample_text_data, sample_labels):
        """Test successful validation."""
        mock_predictions = np.array([1, 0, 1])
        mock_predict.return_value = mock_predictions
        mock_check_Y.return_value = np.array([1, 0, 1])  # Perfect predictions
        
        wrapper = FastTextWrapper(fasttext_config)
        wrapper.trained = True
        
        result = wrapper.validate(sample_text_data, sample_labels)
        
        mock_predict.assert_called_once_with(sample_text_data)
        mock_check_Y.assert_called_once_with(sample_labels)
        assert result == 1.0  # Perfect accuracy
    
    def test_create_dataset(self, fasttext_config, sample_text_data, sample_labels, mock_tokenizer):
        """Test dataset creation."""
        wrapper = FastTextWrapper(fasttext_config)
        wrapper.tokenizer = mock_tokenizer
        
        with patch('torchTextClassifiers.classifiers.fasttext.wrapper.FastTextModelDataset') as mock_dataset_class:
            mock_dataset = Mock()
            mock_dataset_class.return_value = mock_dataset
            
            result = wrapper.create_dataset(sample_text_data, sample_labels)
            
            mock_dataset_class.assert_called_once_with(
                categorical_variables=None,
                texts=sample_text_data,
                outputs=sample_labels,
                tokenizer=mock_tokenizer
            )
            assert result == mock_dataset
    
    def test_create_dataset_with_categorical(self, fasttext_config, sample_text_data, sample_labels, 
                                           sample_categorical_data, mock_tokenizer):
        """Test dataset creation with categorical variables."""
        wrapper = FastTextWrapper(fasttext_config)
        wrapper.tokenizer = mock_tokenizer
        
        with patch('torchTextClassifiers.classifiers.fasttext.wrapper.FastTextModelDataset') as mock_dataset_class:
            mock_dataset = Mock()
            mock_dataset_class.return_value = mock_dataset
            
            result = wrapper.create_dataset(sample_text_data, sample_labels, sample_categorical_data)
            
            mock_dataset_class.assert_called_once_with(
                categorical_variables=sample_categorical_data,
                texts=sample_text_data,
                outputs=sample_labels,
                tokenizer=mock_tokenizer
            )
            assert result == mock_dataset
    
    def test_create_dataloader(self, fasttext_config, mock_dataset):
        """Test dataloader creation."""
        mock_dataloader = Mock()
        mock_dataset.create_dataloader.return_value = mock_dataloader
        
        wrapper = FastTextWrapper(fasttext_config)
        
        result = wrapper.create_dataloader(mock_dataset, batch_size=32, num_workers=4, shuffle=True)
        
        mock_dataset.create_dataloader.assert_called_once_with(
            batch_size=32, num_workers=4, shuffle=True
        )
        assert result == mock_dataloader
    
    @patch('torchTextClassifiers.classifiers.fasttext.wrapper.FastTextModule')
    def test_load_best_model(self, mock_module_class, fasttext_config, mock_pytorch_model):
        """Test loading best model from checkpoint."""
        mock_loaded_module = Mock()
        mock_loaded_module.model = mock_pytorch_model
        mock_module_class.load_from_checkpoint.return_value = mock_loaded_module
        
        wrapper = FastTextWrapper(fasttext_config)
        wrapper.pytorch_model = mock_pytorch_model
        wrapper.loss = Mock()
        wrapper.optimizer = Mock()
        wrapper.optimizer_params = {}
        wrapper.scheduler = Mock()
        wrapper.scheduler_params = {}
        
        mock_pytorch_model.to.return_value = mock_pytorch_model
        mock_pytorch_model.eval = Mock()
        
        checkpoint_path = "/fake/checkpoint/path"
        wrapper.load_best_model(checkpoint_path)
        
        # Verify checkpoint loading
        mock_module_class.load_from_checkpoint.assert_called_once_with(
            checkpoint_path,
            model=mock_pytorch_model,
            loss=wrapper.loss,
            optimizer=wrapper.optimizer,
            optimizer_params=wrapper.optimizer_params,
            scheduler=wrapper.scheduler,
            scheduler_params=wrapper.scheduler_params,
            scheduler_interval="epoch"
        )
        
        # Verify model state updates
        assert wrapper.lightning_module == mock_loaded_module
        assert wrapper.pytorch_model == mock_pytorch_model
        assert wrapper.trained == True
        mock_pytorch_model.eval.assert_called_once()
    
    @patch('torchTextClassifiers.classifiers.fasttext.wrapper.check_X')
    def test_predict_and_explain_success(self, mock_check_X, fasttext_config, sample_text_data, mock_pytorch_model):
        """Test successful predict_and_explain."""
        mock_check_X.return_value = (sample_text_data, None, True)
        expected_result = (np.array([1, 0, 1]), np.array([0.8, 0.2, 0.9]))
        mock_pytorch_model.predict_and_explain.return_value = expected_result
        mock_pytorch_model.no_cat_var = True
        
        wrapper = FastTextWrapper(fasttext_config)
        wrapper.trained = True
        wrapper.pytorch_model = mock_pytorch_model
        wrapper.config.num_categorical_features = None
        
        result = wrapper.predict_and_explain(sample_text_data)
        
        mock_pytorch_model.predict_and_explain.assert_called_once()
        assert result == expected_result
    
    @patch('torchTextClassifiers.classifiers.fasttext.wrapper.check_X')
    def test_predict_and_explain_not_trained(self, mock_check_X, fasttext_config, sample_text_data):
        """Test predict_and_explain fails when model not trained."""
        wrapper = FastTextWrapper(fasttext_config)
        wrapper.trained = False
        
        with pytest.raises(Exception, match="Model must be trained first"):
            wrapper.predict_and_explain(sample_text_data)


class TestFastTextModelDataset:
    """Test FastTextModelDataset class."""
    
    def test_dataset_initialization_text_only(self, sample_text_data, sample_labels, mock_tokenizer):
        """Test dataset initialization with text only."""
        dataset = FastTextModelDataset(
            categorical_variables=None,
            texts=sample_text_data,
            outputs=sample_labels,
            tokenizer=mock_tokenizer
        )
        
        assert len(dataset) == len(sample_text_data)
        assert dataset.texts is sample_text_data
        assert dataset.outputs is sample_labels
        assert dataset.tokenizer is mock_tokenizer
        assert dataset.categorical_variables is None
    
    def test_dataset_initialization_with_categorical(self, sample_text_data, sample_labels, 
                                                   sample_categorical_data, mock_tokenizer):
        """Test dataset initialization with categorical variables."""
        dataset = FastTextModelDataset(
            categorical_variables=sample_categorical_data,
            texts=sample_text_data,
            outputs=sample_labels,
            tokenizer=mock_tokenizer
        )
        
        assert len(dataset) == len(sample_text_data)
        assert dataset.categorical_variables is sample_categorical_data
    
    def test_dataset_length_mismatch_categorical(self, sample_text_data, sample_labels, mock_tokenizer):
        """Test dataset initialization fails with mismatched lengths."""
        wrong_categorical = np.array([[1, 2]])  # Wrong length
        
        with pytest.raises(ValueError, match="Categorical variables and texts must have the same length"):
            FastTextModelDataset(
                categorical_variables=wrong_categorical,
                texts=sample_text_data,
                outputs=sample_labels,
                tokenizer=mock_tokenizer
            )
    
    def test_dataset_length_mismatch_outputs(self, sample_text_data, mock_tokenizer):
        """Test dataset initialization fails with mismatched output lengths."""
        wrong_outputs = np.array([1, 0])  # Wrong length
        
        with pytest.raises(ValueError, match="Outputs and texts must have the same length"):
            FastTextModelDataset(
                categorical_variables=None,
                texts=sample_text_data,
                outputs=wrong_outputs,
                tokenizer=mock_tokenizer
            )
    
    def test_dataset_getitem_with_outputs(self, sample_text_data, sample_labels, mock_tokenizer):
        """Test dataset __getitem__ with outputs."""
        dataset = FastTextModelDataset(
            categorical_variables=None,
            texts=sample_text_data,
            outputs=sample_labels,
            tokenizer=mock_tokenizer
        )
        
        text, categorical_vars, output = dataset[0]
        
        assert text == sample_text_data[0]
        assert categorical_vars is None
        assert output == sample_labels[0]
    
    def test_dataset_getitem_without_outputs(self, sample_text_data, mock_tokenizer):
        """Test dataset __getitem__ without outputs."""
        dataset = FastTextModelDataset(
            categorical_variables=None,
            texts=sample_text_data,
            outputs=None,
            tokenizer=mock_tokenizer
        )
        
        text, categorical_vars = dataset[0]
        
        assert text == sample_text_data[0]
        assert categorical_vars is None
    
    @patch('torch.utils.data.DataLoader')
    def test_create_dataloader(self, mock_dataloader_class, sample_text_data, sample_labels, mock_tokenizer):
        """Test dataloader creation."""
        mock_dataloader = Mock()
        mock_dataloader_class.return_value = mock_dataloader
        
        dataset = FastTextModelDataset(
            categorical_variables=None,
            texts=sample_text_data,
            outputs=sample_labels,
            tokenizer=mock_tokenizer
        )
        
        result = dataset.create_dataloader(batch_size=16, shuffle=True, num_workers=2)
        
        mock_dataloader_class.assert_called_once()
        call_kwargs = mock_dataloader_class.call_args[1]
        assert call_kwargs['batch_size'] == 16
        assert call_kwargs['shuffle'] == True
        assert call_kwargs['num_workers'] == 2
        
        assert result == mock_dataloader