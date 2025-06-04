import logging
import time
import json
from typing import Optional, Union, Type, List, Dict, Any
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from torch.optim import SGD, Adam

from .utilities.checkers import check_X, check_Y, NumpyJSONEncoder
from .classifiers.fasttext.wrapper import FastTextWrapper
from .classifiers.fasttext.config import FastTextConfig
from .classifiers.fasttext.tokenizer import NGramTokenizer
from .classifiers.base import BaseClassifierConfig, BaseClassifierWrapper


logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
)


class ClassifierType(Enum):
    """Enum for different classifier types."""
    FASTTEXT = "fasttext"
    # Add more classifier types here as needed
    # BERT = "bert"
    # LSTM = "lstm"
    # CNN = "cnn"




class ClassifierFactory:
    """Factory class to create classifier wrappers."""
    
    _registry = {
        ClassifierType.FASTTEXT: FastTextWrapper,
    }
    
    @classmethod
    def create_classifier(cls, classifier_type: ClassifierType, config: BaseClassifierConfig) -> BaseClassifierWrapper:
        """Create a classifier wrapper based on type and configuration."""
        if classifier_type not in cls._registry:
            raise ValueError(f"Unsupported classifier type: {classifier_type}")
        
        wrapper_class = cls._registry[classifier_type]
        return wrapper_class(config)
    
    @classmethod
    def register_classifier(cls, classifier_type: ClassifierType, wrapper_class: Type[BaseClassifierWrapper]):
        """Register a new classifier type."""
        cls._registry[classifier_type] = wrapper_class


@dataclass
class torchTextClassifiers:
    """
    Generic wrapper class for different types of text classifiers.
    
    Args:
        classifier_type (ClassifierType): Type of classifier to use
        config (BaseClassifierConfig): Configuration for the specific classifier
    """
    
    classifier_type: ClassifierType
    config: BaseClassifierConfig
    
    # Internal fields
    classifier_wrapper: Optional[BaseClassifierWrapper] = field(init=False, default=None)
    
    def __post_init__(self):
        """Initialize the classifier wrapper after dataclass initialization."""
        self.classifier_wrapper = ClassifierFactory.create_classifier(
            self.classifier_type, self.config
        )
    
    @classmethod
    def create_fasttext(
        cls,
        embedding_dim: int,
        sparse: bool,
        num_tokens: int,
        min_count: int,
        min_n: int,
        max_n: int,
        len_word_ngrams: int,
        **kwargs
    ) -> "torchTextClassifiers":
        """Convenience method to create FastText classifier."""
        config = FastTextConfig(
            embedding_dim=embedding_dim,
            sparse=sparse,
            num_tokens=num_tokens,
            min_count=min_count,
            min_n=min_n,
            max_n=max_n,
            len_word_ngrams=len_word_ngrams,
            **kwargs
        )
        return cls(ClassifierType.FASTTEXT, config)
    
    @classmethod
    def build_from_tokenizer(
        cls,
        classifier_type: ClassifierType,
        tokenizer: NGramTokenizer,
        embedding_dim: int,
        num_classes: Optional[int],
        categorical_vocabulary_sizes: Optional[List[int]] = None,
        sparse: bool = False,
        **kwargs
    ) -> "torchTextClassifiers":
        """Build classifier from existing tokenizer."""
        if classifier_type == ClassifierType.FASTTEXT:
            # Ensure the tokenizer has required attributes
            if not all(
                hasattr(tokenizer, attr)
                for attr in ["min_count", "min_n", "max_n", "num_tokens", "word_ngrams"]
            ):
                raise ValueError(f"Missing attributes in tokenizer: {tokenizer}")
            
            config = FastTextConfig(
                num_tokens=tokenizer.num_tokens,
                embedding_dim=embedding_dim,
                min_count=tokenizer.min_count,
                min_n=tokenizer.min_n,
                max_n=tokenizer.max_n,
                len_word_ngrams=tokenizer.word_ngrams,
                sparse=sparse,
                num_classes=num_classes,
                categorical_vocabulary_sizes=categorical_vocabulary_sizes,
                **kwargs
            )
            
            wrapper = cls(classifier_type, config)
            wrapper.classifier_wrapper.tokenizer = tokenizer
            wrapper.classifier_wrapper._build_pytorch_model()
            
            return wrapper
        else:
            raise ValueError(f"Unsupported classifier type: {classifier_type}")
    
    def build_tokenizer(self, training_text: np.ndarray) -> None:
        """Build tokenizer from training text."""
        self.classifier_wrapper.build_tokenizer(training_text)
    
    def build(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray = None,
        lightning=True,
        **kwargs
    ) -> None:
        """Build the classifier from training data."""
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
        """Train the classifier."""
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
        """Make predictions."""
        return self.classifier_wrapper.predict(X, **kwargs)
    
    def validate(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> float:
        """Validate the model."""
        return self.classifier_wrapper.validate(X, Y, **kwargs)
    
    def predict_and_explain(self, X: np.ndarray, **kwargs):
        """Predict and explain (if supported by the classifier)."""
        if hasattr(self.classifier_wrapper, 'predict_and_explain'):
            return self.classifier_wrapper.predict_and_explain(X, **kwargs)
        else:
            raise NotImplementedError(f"Explanation not supported for {self.classifier_type}")
    
    def to_json(self, filepath: str) -> None:
        """Save configuration to JSON."""
        with open(filepath, "w") as f:
            data = {
                "classifier_type": self.classifier_type.value,
                "config": self.config.to_dict(),
            }
            json.dump(data, f, cls=NumpyJSONEncoder, indent=4)
    
    @classmethod
    def from_json(cls, filepath: str) -> "torchTextClassifiers":
        """Load configuration from JSON."""
        with open(filepath, "r") as f:
            data = json.load(f)
        
        classifier_type = ClassifierType(data["classifier_type"])
        
        if classifier_type == ClassifierType.FASTTEXT:
            config = FastTextConfig.from_dict(data["config"])
        else:
            raise ValueError(f"Unsupported classifier type: {classifier_type}")
        
        return cls(classifier_type, config)