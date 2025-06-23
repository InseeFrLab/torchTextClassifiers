import logging
import time
import json
from typing import Optional, Union, Type, List, Dict, Any
from enum import Enum

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
from .factories import create_config_from_dict


logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
)


class ClassifierType(Enum):
    """Enumeration of supported classifier types.
    
    This enum defines the available classifier types that can be used
    with the torchTextClassifiers framework. Each type corresponds to
    a specific implementation with its own configuration and wrapper.
    
    Attributes:
        FASTTEXT: FastText-based text classifier for efficient text classification
        
    Example:
        >>> from torchTextClassifiers import ClassifierType
        >>> classifier_type = ClassifierType.FASTTEXT
        >>> print(classifier_type.value)
        'fasttext'
    """
    FASTTEXT = "fasttext"
    # Add more classifier types here as needed
    # BERT = "bert"
    # LSTM = "lstm"
    # CNN = "cnn"




class ClassifierFactory:
    """Factory class for creating and managing classifier wrappers.
    
    This factory implements a registry pattern that allows for dynamic loading
    and creation of different classifier types. It supports both automatic
    registration of built-in classifiers and manual registration of custom
    classifier implementations.
    
    The factory uses lazy loading, attempting to import and register classifier
    modules only when they are first requested.
    
    Attributes:
        _registry: Dictionary mapping ClassifierType to wrapper class implementations
        
    Example:
        >>> from torchTextClassifiers import ClassifierFactory, ClassifierType
        >>> from torchTextClassifiers.classifiers.fasttext.config import FastTextConfig
        >>> config = FastTextConfig(embedding_dim=100, num_tokens=1000, ...)
        >>> wrapper = ClassifierFactory.create_classifier(ClassifierType.FASTTEXT, config)
    """
    
    _registry: Dict[ClassifierType, Type[BaseClassifierWrapper]] = {}
    
    @classmethod
    def create_classifier(cls, classifier_type: ClassifierType, config: BaseClassifierConfig) -> BaseClassifierWrapper:
        """Create a classifier wrapper instance.
        
        This method creates a classifier wrapper of the specified type using the
        provided configuration. If the classifier type is not yet registered,
        it attempts to load the corresponding module dynamically.
        
        Args:
            classifier_type: The type of classifier to create
            config: Configuration object for the classifier
            
        Returns:
            BaseClassifierWrapper: Initialized classifier wrapper instance
            
        Raises:
            ValueError: If the classifier type is not supported or cannot be loaded
            
        Example:
            >>> config = FastTextConfig(embedding_dim=50, num_tokens=5000)
            >>> classifier = ClassifierFactory.create_classifier(
            ...     ClassifierType.FASTTEXT, config
            ... )
        """
        if classifier_type not in cls._registry:
            # Try to load the classifier module dynamically
            cls._try_load_classifier(classifier_type)
            
        if classifier_type not in cls._registry:
            raise ValueError(f"Unsupported classifier type: {classifier_type}")
        
        wrapper_class = cls._registry[classifier_type]
        return wrapper_class(config)
    
    @classmethod
    def register_classifier(cls, classifier_type: ClassifierType, wrapper_class: Type[BaseClassifierWrapper]):
        """Register a new classifier type with its wrapper class.
        
        This method allows registration of custom classifier implementations
        that can then be used through the factory pattern.
        
        Args:
            classifier_type: The classifier type enum value
            wrapper_class: The wrapper class that implements BaseClassifierWrapper
            
        Example:
            >>> class MyCustomWrapper(BaseClassifierWrapper):
            ...     # Implementation here
            ...     pass
            >>> ClassifierFactory.register_classifier(
            ...     ClassifierType.CUSTOM, MyCustomWrapper
            ... )
        """
        cls._registry[classifier_type] = wrapper_class
    
    @classmethod
    def _try_load_classifier(cls, classifier_type: ClassifierType):
        """Attempt to dynamically load and register a classifier module.
        
        This method implements lazy loading for built-in classifier types.
        It tries to import the corresponding wrapper module and register
        it automatically.
        
        Args:
            classifier_type: The classifier type to attempt loading
        """
        if classifier_type == ClassifierType.FASTTEXT:
            try:
                from .classifiers.fasttext.wrapper import FastTextWrapper
                cls.register_classifier(ClassifierType.FASTTEXT, FastTextWrapper)
            except ImportError:
                pass  # Module not available


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
        classifier_type: The type of classifier being used
        config: Configuration object specific to the classifier type
        classifier_wrapper: The underlying classifier implementation
        
    Example:
        >>> from torchTextClassifiers import torchTextClassifiers, ClassifierType
        >>> from torchTextClassifiers.classifiers.fasttext.config import FastTextConfig
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
        >>> # Initialize classifier
        >>> classifier = torchTextClassifiers(ClassifierType.FASTTEXT, config)
        >>> 
        >>> # Build and train
        >>> classifier.build(X_train, y_train)
        >>> classifier.train(X_train, y_train, X_val, y_val, num_epochs=10, batch_size=32)
        >>> 
        >>> # Predict
        >>> predictions = classifier.predict(X_test)
    """
    
    def __init__(self, classifier_type: ClassifierType, config: BaseClassifierConfig):
        """Initialize the torchTextClassifiers instance.
        
        Args:
            classifier_type: The type of classifier to create (e.g., ClassifierType.FASTTEXT)
            config: Configuration object containing classifier-specific parameters
            
        Raises:
            ValueError: If the classifier type is not supported
            
        Example:
            >>> config = FastTextConfig(embedding_dim=50, num_tokens=5000)
            >>> classifier = torchTextClassifiers(ClassifierType.FASTTEXT, config)
        """
        self.classifier_type = classifier_type
        self.config = config
        self.classifier_wrapper: Optional[BaseClassifierWrapper] = None
        self.__post_init__()
    
    def __post_init__(self):
        """Initialize the classifier wrapper after instance creation.
        
        This method creates the appropriate classifier wrapper based on the
        specified classifier type and configuration.
        """
        self.classifier_wrapper = ClassifierFactory.create_classifier(
            self.classifier_type, self.config
        )
    
    def build_tokenizer(self, training_text: np.ndarray) -> None:
        """Build tokenizer from training text data.
        
        This method creates and trains a tokenizer using the provided text data.
        The tokenizer learns vocabulary and encoding schemes specific to the
        classifier type (e.g., n-gram tokenization for FastText).
        
        Args:
            training_text: Array of text strings to build the tokenizer from
            
        Example:
            >>> import numpy as np
            >>> texts = np.array(["Hello world", "This is a test", "Another example"])
            >>> classifier.build_tokenizer(texts)
        """
        self.classifier_wrapper.build_tokenizer(training_text)
    
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
        
        self.classifier_wrapper.build_tokenizer(training_text)
        self.classifier_wrapper._build_pytorch_model()
        
        if lightning:
            self.classifier_wrapper._check_and_init_lightning(**kwargs)
    
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
        
        self.classifier_wrapper.device = device
        
        if verbose:
            logger.info(f"Running on: {device}")
        
        # Build model if not already built
        if self.classifier_wrapper.tokenizer is None or self.classifier_wrapper.pytorch_model is None:
            if verbose:
                start = time.time()
                logger.info("Building the model...")
            self.build(X_train, y_train, **kwargs)
            if verbose:
                end = time.time()
                logger.info(f"Model built in {end - start:.2f} seconds.")
        
        self.classifier_wrapper.pytorch_model = self.classifier_wrapper.pytorch_model.to(device)
        
        # Create datasets and dataloaders using wrapper methods
        train_dataset = self.classifier_wrapper.create_dataset(
            texts=training_text,
            labels=y_train,
            categorical_variables=train_categorical_variables,
        )
        val_dataset = self.classifier_wrapper.create_dataset(
            texts=val_text,
            labels=y_val,
            categorical_variables=val_categorical_variables,
        )
        
        train_dataloader = self.classifier_wrapper.create_dataloader(
            dataset=train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True
        )
        val_dataloader = self.classifier_wrapper.create_dataloader(
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
        
        trainer.fit(self.classifier_wrapper.lightning_module, train_dataloader, val_dataloader)
        
        if verbose:
            end = time.time()
            logger.info(f"Training completed in {end - start:.2f} seconds.")
        
        # Load best model using wrapper method
        best_model_path = trainer.checkpoint_callback.best_model_path
        self.classifier_wrapper.load_best_model(best_model_path)
    
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
        return self.classifier_wrapper.predict(X, **kwargs)
    
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
        return self.classifier_wrapper.validate(X, Y, **kwargs)
    
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
        if hasattr(self.classifier_wrapper, 'predict_and_explain'):
            return self.classifier_wrapper.predict_and_explain(X, **kwargs)
        else:
            raise NotImplementedError(f"Explanation not supported for {self.classifier_type}")
    
    def to_json(self, filepath: str) -> None:
        """Save classifier configuration to JSON file.
        
        This method serializes the classifier type and configuration to a JSON
        file, allowing the classifier to be recreated later with from_json().
        Note: This only saves configuration, not trained model weights.
        
        Args:
            filepath: Path where to save the JSON configuration file
            
        Example:
            >>> classifier.to_json('my_classifier_config.json')
        """
        with open(filepath, "w") as f:
            data = {
                "classifier_type": self.classifier_type.value,
                "config": self.config.to_dict(),
            }
            json.dump(data, f, cls=NumpyJSONEncoder, indent=4)
    
    @classmethod
    def from_json(cls, filepath: str) -> "torchTextClassifiers":
        """Load classifier configuration from JSON file.
        
        This method creates a new classifier instance from a previously saved
        configuration file. The classifier will need to be built and trained again.
        
        Args:
            filepath: Path to the JSON configuration file
            
        Returns:
            torchTextClassifiers: New classifier instance with loaded configuration
            
        Raises:
            ValueError: If the classifier type in the file is not supported
            FileNotFoundError: If the configuration file doesn't exist
            
        Example:
            >>> classifier = torchTextClassifiers.from_json('my_classifier_config.json')
            >>> # Now you need to build and train the classifier
            >>> classifier.build(X_train, y_train)
        """
        with open(filepath, "r") as f:
            data = json.load(f)
        
        try:
            classifier_type = ClassifierType(data["classifier_type"])
        except ValueError:
            raise ValueError(f"Unsupported classifier type: {data['classifier_type']}")
        
        # Use the generic config factory
        config = create_config_from_dict(data["classifier_type"], data["config"])
        
        return cls(classifier_type, config)