import logging
import time
import json
from typing import Optional, Union, Type, List, Dict, Any

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from .utilities.checkers import check_X, check_Y, NumpyJSONEncoder
from .classifiers.base import BaseClassifierConfig, BaseClassifierWrapper


logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
)




class torchTextClassifiers:
    """Generic text classifier framework supporting multiple architectures.
    
    This is the main class that provides a unified interface for different types
    of text classifiers. It acts as a high-level wrapper that delegates operations
    to specific classifier implementations while providing a consistent API.
    
    The class supports the full machine learning workflow including:
    - Building tokenizers from training data
    - Model training with validation
    - Prediction and evaluation
    - Model serialization and loading
    
    Attributes:
        config: Configuration object specific to the classifier type
        classifier: The underlying classifier implementation
        
    Example:
        >>> from torchTextClassifiers import torchTextClassifiers
        >>> from torchTextClassifiers.classifiers.fasttext.config import FastTextConfig
        >>> from torchTextClassifiers.classifiers.fasttext.wrapper import FastTextWrapper
        >>> 
        >>> # Create configuration
        >>> config = FastTextConfig(
        ...     embedding_dim=100,
        ...     num_tokens=10000,
        ...     min_count=1,
        ...     min_n=3,
        ...     max_n=6,
        ...     len_word_ngrams=2,
        ...     num_classes=2
        ... )
        >>> 
        >>> # Initialize classifier with wrapper
        >>> wrapper = FastTextWrapper(config)
        >>> classifier = torchTextClassifiers(wrapper)
        >>> 
        >>> # Build and train
        >>> classifier.build(X_train, y_train)
        >>> classifier.train(X_train, y_train, X_val, y_val, num_epochs=10, batch_size=32)
        >>> 
        >>> # Predict
        >>> predictions = classifier.predict(X_test)
    """
    
    def __init__(self, classifier: BaseClassifierWrapper):
        """Initialize the torchTextClassifiers instance.
        
        Args:
            classifier: An instance of a classifier wrapper that implements BaseClassifierWrapper
            
        Example:
            >>> from torchTextClassifiers.classifiers.fasttext.wrapper import FastTextWrapper
            >>> from torchTextClassifiers.classifiers.fasttext.config import FastTextConfig
            >>> config = FastTextConfig(embedding_dim=50, num_tokens=5000)
            >>> wrapper = FastTextWrapper(config)
            >>> classifier = torchTextClassifiers(wrapper)
        """
        self.classifier = classifier
        self.config = classifier.config
    
    
    def build_tokenizer(self, training_text: np.ndarray) -> None:
        """Build tokenizer from training text data.
        
        This method is kept for backward compatibility. It delegates to
        prepare_text_features which handles the actual text preprocessing.
        
        Args:
            training_text: Array of text strings to build the tokenizer from
            
        Example:
            >>> import numpy as np
            >>> texts = np.array(["Hello world", "This is a test", "Another example"])
            >>> classifier.build_tokenizer(texts)
        """
        self.classifier.prepare_text_features(training_text)
    
    def prepare_text_features(self, training_text: np.ndarray) -> None:
        """Prepare text features for the classifier.
        
        This method handles text preprocessing which could involve tokenization,
        vectorization, or other approaches depending on the classifier type.
        
        Args:
            training_text: Array of text strings to prepare features from
            
        Example:
            >>> import numpy as np
            >>> texts = np.array(["Hello world", "This is a test", "Another example"])
            >>> classifier.prepare_text_features(texts)
        """
        self.classifier.prepare_text_features(training_text)
    
    def build(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray = None,
        lightning=True,
        **kwargs
    ) -> None:
        """Build the complete classifier from training data.
        
        This method handles the full model building process including:
        - Input validation and preprocessing
        - Tokenizer creation from training text
        - Model architecture initialization
        - Lightning module setup (if enabled)
        
        Args:
            X_train: Training input data (text and optional categorical features)
            y_train: Training labels (optional, can be inferred if num_classes is set)
            lightning: Whether to initialize PyTorch Lightning components
            **kwargs: Additional arguments passed to Lightning initialization
            
        Raises:
            ValueError: If y_train is None and num_classes is not set in config
            ValueError: If label values are outside expected range
            
        Example:
            >>> X_train = np.array(["text sample 1", "text sample 2"])
            >>> y_train = np.array([0, 1])
            >>> classifier.build(X_train, y_train)
        """
        training_text, categorical_variables, no_cat_var = check_X(X_train)
        
        if y_train is not None:
            if self.config.num_classes is not None:
                if self.config.num_classes != len(np.unique(y_train)):
                    logger.warning(
                        f"Updating num_classes from {self.config.num_classes} to {len(np.unique(y_train))}"
                    )
            
            y_train = check_Y(y_train)
            self.config.num_classes = len(np.unique(y_train))
            
            if np.max(y_train) >= self.config.num_classes:
                raise ValueError(
                    "y_train must contain values between 0 and num_classes-1"
                )
        else:
            if self.config.num_classes is None:
                raise ValueError(
                    "Either num_classes must be provided at init or y_train must be provided here."
                )
        
        # Handle categorical variables
        if not no_cat_var:
            if hasattr(self.config, 'num_categorical_features') and self.config.num_categorical_features is not None:
                if self.config.num_categorical_features != categorical_variables.shape[1]:
                    logger.warning(
                        f"Updating num_categorical_features from {self.config.num_categorical_features} to {categorical_variables.shape[1]}"
                    )
            
            if hasattr(self.config, 'num_categorical_features'):
                self.config.num_categorical_features = categorical_variables.shape[1]
            
            categorical_vocabulary_sizes = np.max(categorical_variables, axis=0) + 1
            
            if hasattr(self.config, 'categorical_vocabulary_sizes') and self.config.categorical_vocabulary_sizes is not None:
                if self.config.categorical_vocabulary_sizes != list(categorical_vocabulary_sizes):
                    logger.warning(
                        "Overwriting categorical_vocabulary_sizes with values from training data."
                    )
            if hasattr(self.config, 'categorical_vocabulary_sizes'):
                self.config.categorical_vocabulary_sizes = list(categorical_vocabulary_sizes)
        
        self.classifier.prepare_text_features(training_text)
        self.classifier._build_pytorch_model()
        
        if lightning:
            self.classifier._check_and_init_lightning(**kwargs)
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        num_epochs: int,
        batch_size: int,
        cpu_run: bool = False,
        num_workers: int = 12,
        patience_train: int = 3,
        verbose: bool = False,
        trainer_params: Optional[dict] = None,
        **kwargs
    ) -> None:
        """Train the classifier using PyTorch Lightning.
        
        This method handles the complete training process including:
        - Data validation and preprocessing
        - Dataset and DataLoader creation
        - PyTorch Lightning trainer setup with callbacks
        - Model training with early stopping
        - Best model loading after training
        
        Args:
            X_train: Training input data
            y_train: Training labels
            X_val: Validation input data
            y_val: Validation labels
            num_epochs: Maximum number of training epochs
            batch_size: Batch size for training and validation
            cpu_run: If True, force training on CPU instead of GPU
            num_workers: Number of worker processes for data loading
            patience_train: Number of epochs to wait for improvement before early stopping
            verbose: If True, print detailed training progress
            trainer_params: Additional parameters to pass to PyTorch Lightning Trainer
            **kwargs: Additional arguments passed to the build method
            
        Example:
            >>> classifier.train(
            ...     X_train, y_train, X_val, y_val,
            ...     num_epochs=50,
            ...     batch_size=32,
            ...     patience_train=5,
            ...     verbose=True
            ... )
        """
        # Input validation
        training_text, train_categorical_variables, train_no_cat_var = check_X(X_train)
        val_text, val_categorical_variables, val_no_cat_var = check_X(X_val)
        y_train = check_Y(y_train)
        y_val = check_Y(y_val)
        
        # Consistency checks
        assert train_no_cat_var == val_no_cat_var, (
            "X_train and X_val must have the same number of categorical variables."
        )
        assert X_train.shape[0] == y_train.shape[0], (
            "X_train and y_train must have the same number of observations."
        )
        assert X_train.ndim > 1 and X_train.shape[1] == X_val.shape[1] or X_val.ndim == 1, (
            "X_train and X_val must have the same number of columns."
        )
        
        if verbose:
            logger.info("Starting training process...")
        
        # Device setup
        if cpu_run:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.classifier.device = device
        
        if verbose:
            logger.info(f"Running on: {device}")
        
        # Build model if not already built
        if self.classifier.pytorch_model is None:
            if verbose:
                start = time.time()
                logger.info("Building the model...")
            self.build(X_train, y_train, **kwargs)
            if verbose:
                end = time.time()
                logger.info(f"Model built in {end - start:.2f} seconds.")
        
        self.classifier.pytorch_model = self.classifier.pytorch_model.to(device)
        
        # Create datasets and dataloaders using wrapper methods
        train_dataset = self.classifier.create_dataset(
            texts=training_text,
            labels=y_train,
            categorical_variables=train_categorical_variables,
        )
        val_dataset = self.classifier.create_dataset(
            texts=val_text,
            labels=y_val,
            categorical_variables=val_categorical_variables,
        )
        
        train_dataloader = self.classifier.create_dataloader(
            dataset=train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True
        )
        val_dataloader = self.classifier.create_dataloader(
            dataset=val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False
        )
        
        # Setup trainer
        callbacks = [
            ModelCheckpoint(
                monitor="val_loss",
                save_top_k=1,
                save_last=False,
                mode="min",
            ),
            EarlyStopping(
                monitor="val_loss",
                patience=patience_train,
                mode="min",
            ),
            LearningRateMonitor(logging_interval="step"),
        ]
        
        train_params = {
            "callbacks": callbacks,
            "max_epochs": num_epochs,
            "num_sanity_val_steps": 2,
            "strategy": "auto",
            "log_every_n_steps": 1,
            "enable_progress_bar": True,
        }
        
        if trainer_params is not None:
            train_params.update(trainer_params)
        
        trainer = pl.Trainer(**train_params)
        
        torch.cuda.empty_cache()
        torch.set_float32_matmul_precision("medium")
        
        if verbose:
            logger.info("Launching training...")
            start = time.time()
        
        trainer.fit(self.classifier.lightning_module, train_dataloader, val_dataloader)
        
        if verbose:
            end = time.time()
            logger.info(f"Training completed in {end - start:.2f} seconds.")
        
        # Load best model using wrapper method
        best_model_path = trainer.checkpoint_callback.best_model_path
        self.classifier.load_best_model(best_model_path)
    
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Make predictions on input data.
        
        Args:
            X: Input data for prediction (text and optional categorical features)
            **kwargs: Additional arguments passed to the underlying predictor
            
        Returns:
            np.ndarray: Predicted class labels
            
        Example:
            >>> X_test = np.array(["new text sample", "another sample"])
            >>> predictions = classifier.predict(X_test)
            >>> print(predictions)  # [0, 1]
        """
        return self.classifier.predict(X, **kwargs)
    
    def validate(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> float:
        """Validate the model on test data.
        
        Args:
            X: Input data for validation
            Y: True labels for validation
            **kwargs: Additional arguments passed to the validator
            
        Returns:
            float: Validation accuracy score
            
        Example:
            >>> accuracy = classifier.validate(X_test, y_test)
            >>> print(f"Accuracy: {accuracy:.3f}")
        """
        return self.classifier.validate(X, Y, **kwargs)
    
    def predict_and_explain(self, X: np.ndarray, **kwargs):
        """Make predictions with explanations (if supported).
        
        This method provides both predictions and explanations for the model's
        decisions. Availability depends on the specific classifier implementation.
        
        Args:
            X: Input data for prediction
            **kwargs: Additional arguments passed to the explainer
            
        Returns:
            tuple: (predictions, explanations) where explanations format depends
                  on the classifier type
                  
        Raises:
            NotImplementedError: If the classifier doesn't support explanations
            
        Example:
            >>> predictions, explanations = classifier.predict_and_explain(X_test)
            >>> print(f"Predictions: {predictions}")
            >>> print(f"Explanations: {explanations}")
        """
        if hasattr(self.classifier, 'predict_and_explain'):
            return self.classifier.predict_and_explain(X, **kwargs)
        else:
            raise NotImplementedError(f"Explanation not supported for {type(self.classifier).__name__}")
    
    def to_json(self, filepath: str) -> None:
        """Save classifier configuration to JSON file.
        
        This method serializes the classifier configuration to a JSON
        file. Note: This only saves configuration, not trained model weights.
        Custom classifier wrappers should implement a class method `get_wrapper_class_info()`
        that returns a dict with 'module' and 'class_name' keys for proper reconstruction.
        
        Args:
            filepath: Path where to save the JSON configuration file
            
        Example:
            >>> classifier.to_json('my_classifier_config.json')
        """
        with open(filepath, "w") as f:
            data = {
                "config": self.config.to_dict(),
            }
            
            # Try to get wrapper class info for reconstruction
            if hasattr(self.classifier.__class__, 'get_wrapper_class_info'):
                data["wrapper_class_info"] = self.classifier.__class__.get_wrapper_class_info()
            else:
                # Fallback: store module and class name
                data["wrapper_class_info"] = {
                    "module": self.classifier.__class__.__module__,
                    "class_name": self.classifier.__class__.__name__
                }
            
            json.dump(data, f, cls=NumpyJSONEncoder, indent=4)
    
    @classmethod
    def from_json(cls, filepath: str, wrapper_class: Optional[Type[BaseClassifierWrapper]] = None) -> "torchTextClassifiers":
        """Load classifier configuration from JSON file.
        
        This method creates a new classifier instance from a previously saved
        configuration file. The classifier will need to be built and trained again.
        
        Args:
            filepath: Path to the JSON configuration file
            wrapper_class: Optional wrapper class to use. If not provided, will try to
                          reconstruct from saved wrapper_class_info
            
        Returns:
            torchTextClassifiers: New classifier instance with loaded configuration
            
        Raises:
            ImportError: If the wrapper class cannot be imported
            FileNotFoundError: If the configuration file doesn't exist
            
        Example:
            >>> # Using saved wrapper class info
            >>> classifier = torchTextClassifiers.from_json('my_classifier_config.json')
            >>> 
            >>> # Or providing wrapper class explicitly
            >>> from torchTextClassifiers.classifiers.fasttext.wrapper import FastTextWrapper
            >>> classifier = torchTextClassifiers.from_json('config.json', FastTextWrapper)
        """
        with open(filepath, "r") as f:
            data = json.load(f)
        
        if wrapper_class is None:
            # Try to reconstruct wrapper class from saved info
            if "wrapper_class_info" not in data:
                raise ValueError("No wrapper_class_info found in config file and no wrapper_class provided")
            
            wrapper_info = data["wrapper_class_info"]
            module_name = wrapper_info["module"]
            class_name = wrapper_info["class_name"]
            
            # Dynamically import the wrapper class
            import importlib
            module = importlib.import_module(module_name)
            wrapper_class = getattr(module, class_name)
        
        # Reconstruct config using wrapper class's config class
        config_class = wrapper_class.get_config_class()
        config = config_class.from_dict(data["config"])
        
        # Create wrapper instance
        wrapper = wrapper_class(config)
        
        return cls(wrapper)