import logging
import pickle
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

try:
    from captum.attr import LayerIntegratedGradients

    HAS_CAPTUM = True
except ImportError:
    HAS_CAPTUM = False


import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from torchTextClassifiers.dataset import TextClassificationDataset
from torchTextClassifiers.model import TextClassificationModel, TextClassificationModule
from torchTextClassifiers.model.components import (
    AttentionConfig,
    CategoricalForwardType,
    CategoricalVariableNet,
    ClassificationHead,
    TextEmbedder,
    TextEmbedderConfig,
)
from torchTextClassifiers.tokenizers import BaseTokenizer, TokenizerOutput

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
)


@dataclass
class ModelConfig:
    """Base configuration class for text classifiers."""

    embedding_dim: int
    categorical_vocabulary_sizes: Optional[List[int]] = None
    categorical_embedding_dims: Optional[Union[List[int], int]] = None
    num_classes: Optional[int] = None
    attention_config: Optional[AttentionConfig] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        return cls(**data)


@dataclass
class TrainingConfig:
    num_epochs: int
    batch_size: int
    lr: float
    loss: torch.nn.Module = field(default_factory=lambda: torch.nn.CrossEntropyLoss())
    optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam
    scheduler: Optional[Type[torch.optim.lr_scheduler._LRScheduler]] = None
    accelerator: str = "auto"
    num_workers: int = 12
    patience_early_stopping: int = 3
    dataloader_params: Optional[dict] = None
    trainer_params: Optional[dict] = None
    optimizer_params: Optional[dict] = None
    scheduler_params: Optional[dict] = None
    save_path: Optional[str] = "my_ttc"

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # Serialize loss and scheduler as their class names
        data["loss"] = self.loss.__class__.__name__
        if self.scheduler is not None:
            data["scheduler"] = self.scheduler.__name__
        return data


class torchTextClassifiers:
    """Generic text classifier framework supporting multiple architectures.

    Given a tokenizer and model configuration, this class initializes:
    - Text embedding layer (if needed)
    - Categorical variable embedding network (if categorical variables are provided)
    - Classification head
    The resulting model can be trained using PyTorch Lightning and used for predictions.

    """

    def __init__(
        self,
        tokenizer: BaseTokenizer,
        model_config: ModelConfig,
        ragged_multilabel: bool = False,
    ):
        """Initialize the torchTextClassifiers instance.

        Args:
            tokenizer: A tokenizer instance for text preprocessing
            model_config: Configuration parameters for the text classification model

        Example:
            >>> from torchTextClassifiers import ModelConfig, TrainingConfig, torchTextClassifiers
            >>>  # Assume tokenizer is a trained BaseTokenizer instance
            >>> model_config = ModelConfig(
            ...     embedding_dim=10,
            ...     categorical_vocabulary_sizes=[30, 25],
            ...     categorical_embedding_dims=[10, 5],
            ...     num_classes=10,
            ... )
            >>> ttc = torchTextClassifiers(
            ...     tokenizer=tokenizer,
            ...     model_config=model_config,
            ... )
        """

        self.model_config = model_config
        self.tokenizer = tokenizer
        self.ragged_multilabel = ragged_multilabel

        if hasattr(self.tokenizer, "trained"):
            if not self.tokenizer.trained:
                raise RuntimeError(
                    f"Tokenizer {type(self.tokenizer)} must be trained before initializing the classifier."
                )

        self.vocab_size = tokenizer.vocab_size
        self.embedding_dim = model_config.embedding_dim
        self.categorical_vocabulary_sizes = model_config.categorical_vocabulary_sizes
        self.num_classes = model_config.num_classes

        if self.tokenizer.output_vectorized:
            self.text_embedder = None
            logger.info(
                "Tokenizer outputs vectorized tokens; skipping TextEmbedder initialization."
            )
            self.embedding_dim = self.tokenizer.output_dim
        else:
            text_embedder_config = TextEmbedderConfig(
                vocab_size=self.vocab_size,
                embedding_dim=self.embedding_dim,
                padding_idx=tokenizer.padding_idx,
                attention_config=model_config.attention_config,
            )
            self.text_embedder = TextEmbedder(
                text_embedder_config=text_embedder_config,
            )

        classif_head_input_dim = self.embedding_dim
        if self.categorical_vocabulary_sizes:
            self.categorical_var_net = CategoricalVariableNet(
                categorical_vocabulary_sizes=self.categorical_vocabulary_sizes,
                categorical_embedding_dims=model_config.categorical_embedding_dims,
                text_embedding_dim=self.embedding_dim,
            )

            if self.categorical_var_net.forward_type != CategoricalForwardType.SUM_TO_TEXT:
                classif_head_input_dim += self.categorical_var_net.output_dim

        else:
            self.categorical_var_net = None

        self.classification_head = ClassificationHead(
            input_dim=classif_head_input_dim,
            num_classes=model_config.num_classes,
        )

        self.pytorch_model = TextClassificationModel(
            text_embedder=self.text_embedder,
            categorical_variable_net=self.categorical_var_net,
            classification_head=self.classification_head,
        )

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        training_config: TrainingConfig,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = False,
    ) -> None:
        """Train the classifier using PyTorch Lightning.

        This method handles the complete training process including:
        - Data validation and preprocessing
        - Dataset and DataLoader creation
        - PyTorch Lightning trainer setup with callbacks
        - Model training with early stopping
        - Best model loading after training

        Note on Checkpoints:
            After training, the best model checkpoint is automatically loaded.
            This checkpoint contains the full training state (model weights,
            optimizer, and scheduler state). Loading uses weights_only=False
            as the checkpoint is self-generated and trusted.

        Args:
            X_train: Training input data
            y_train: Training labels
            X_val: Validation input data
            y_val: Validation labels
            training_config: Configuration parameters for training
            verbose: Whether to print training progress information


        Example:

                >>> training_config = TrainingConfig(
                ...     lr=1e-3,
                ...     batch_size=4,
                ...     num_epochs=1,
                ... )
                >>> ttc.train(
                ...     X_train=X,
                ...     y_train=Y,
                ...     X_val=X,
                ...     y_val=Y,
                ...     training_config=training_config,
                ... )
        """
        # Input validation
        X_train, y_train = self._check_XY(X_train, y_train)

        if X_val is not None:
            assert y_val is not None, "y_val must be provided if X_val is provided."
        if y_val is not None:
            assert X_val is not None, "X_val must be provided if y_val is provided."

        if X_val is not None and y_val is not None:
            X_val, y_val = self._check_XY(X_val, y_val)

        if (
            X_train["categorical_variables"] is not None
            and X_val["categorical_variables"] is not None
        ):
            assert (
                X_train["categorical_variables"].ndim > 1
                and X_train["categorical_variables"].shape[1]
                == X_val["categorical_variables"].shape[1]
                or X_val["categorical_variables"].ndim == 1
            ), "X_train and X_val must have the same number of columns."

        if verbose:
            logger.info("Starting training process...")

        if training_config.accelerator == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(training_config.accelerator)

        self.device = device

        optimizer_params = {"lr": training_config.lr}
        if training_config.optimizer_params is not None:
            optimizer_params.update(training_config.optimizer_params)

        if training_config.loss is torch.nn.CrossEntropyLoss and self.ragged_multilabel:
            logger.warning(
                "⚠️ You have set ragged_multilabel to True but are using CrossEntropyLoss. We would recommend to use torch.nn.BCEWithLogitsLoss for multilabel classification tasks."
            )

        self.lightning_module = TextClassificationModule(
            model=self.pytorch_model,
            loss=training_config.loss,
            optimizer=training_config.optimizer,
            optimizer_params=optimizer_params,
            scheduler=training_config.scheduler,
            scheduler_params=training_config.scheduler_params
            if training_config.scheduler_params
            else {},
            scheduler_interval="epoch",
        )

        self.pytorch_model.to(self.device)

        if verbose:
            logger.info(f"Running on: {device}")

        train_dataset = TextClassificationDataset(
            texts=X_train["text"],
            categorical_variables=X_train["categorical_variables"],  # None if no cat vars
            tokenizer=self.tokenizer,
            labels=y_train.tolist(),
            ragged_multilabel=self.ragged_multilabel,
        )
        train_dataloader = train_dataset.create_dataloader(
            batch_size=training_config.batch_size,
            num_workers=training_config.num_workers,
            shuffle=True,
            **training_config.dataloader_params if training_config.dataloader_params else {},
        )

        if X_val is not None and y_val is not None:
            val_dataset = TextClassificationDataset(
                texts=X_val["text"],
                categorical_variables=X_val["categorical_variables"],  # None if no cat vars
                tokenizer=self.tokenizer,
                labels=y_val,
                ragged_multilabel=self.ragged_multilabel,
            )
            val_dataloader = val_dataset.create_dataloader(
                batch_size=training_config.batch_size,
                num_workers=training_config.num_workers,
                shuffle=False,
                **training_config.dataloader_params if training_config.dataloader_params else {},
            )
        else:
            val_dataloader = None

        # Setup trainer
        callbacks = [
            ModelCheckpoint(
                monitor="val_loss" if val_dataloader is not None else "train_loss",
                save_top_k=1,
                save_last=False,
                mode="min",
            ),
            EarlyStopping(
                monitor="val_loss" if val_dataloader is not None else "train_loss",
                patience=training_config.patience_early_stopping,
                mode="min",
            ),
            LearningRateMonitor(logging_interval="step"),
        ]

        trainer_params = {
            "accelerator": training_config.accelerator,
            "callbacks": callbacks,
            "max_epochs": training_config.num_epochs,
            "num_sanity_val_steps": 2,
            "strategy": "auto",
            "log_every_n_steps": 1,
            "enable_progress_bar": True,
        }

        if training_config.trainer_params is not None:
            trainer_params.update(training_config.trainer_params)

        trainer = pl.Trainer(**trainer_params)

        torch.cuda.empty_cache()
        torch.set_float32_matmul_precision("medium")

        if verbose:
            logger.info("Launching training...")
            start = time.time()

        trainer.fit(self.lightning_module, train_dataloader, val_dataloader)

        if verbose:
            end = time.time()
            logger.info(f"Training completed in {end - start:.2f} seconds.")

        best_model_path = trainer.checkpoint_callback.best_model_path
        self.checkpoint_path = best_model_path

        self.lightning_module = TextClassificationModule.load_from_checkpoint(
            best_model_path,
            model=self.pytorch_model,
            loss=training_config.loss,
            weights_only=False,  # Required: checkpoint contains optimizer/scheduler state
        )

        self.pytorch_model = self.lightning_module.model.to(self.device)

        self.save_path = training_config.save_path
        self.save(self.save_path)

        self.lightning_module.eval()

    def _check_XY(self, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = self._check_X(X)
        Y = self._check_Y(Y)

        if X["text"].shape[0] != len(Y):
            raise ValueError("X_train and y_train must have the same number of observations.")

        return X, Y

    @staticmethod
    def _check_text_col(X):
        assert isinstance(
            X, np.ndarray
        ), "X must be a numpy array of shape (N,d), with the first column being the text and the rest being the categorical variables."

        try:
            if X.ndim > 1:
                text = X[:, 0].astype(str)
            else:
                text = X[:].astype(str)
        except ValueError:
            logger.error("The first column of X must be castable in string format.")

        return text

    def _check_categorical_variables(self, X: np.ndarray) -> None:
        """Check if categorical variables in X match training configuration.

        Args:
            X: Input data to check

        Raises:
            ValueError: If the number of categorical variables does not match
                        the training configuration
        """

        assert self.categorical_var_net is not None

        if X.ndim > 1:
            num_cat_vars = X.shape[1] - 1
        else:
            num_cat_vars = 0

        if num_cat_vars != self.categorical_var_net.num_categorical_features:
            raise ValueError(
                f"X must have the same number of categorical variables as the number of embedding layers in the categorical net: ({self.categorical_var_net.num_categorical_features})."
            )

        try:
            categorical_variables = X[:, 1:].astype(int)
        except ValueError:
            logger.error(
                f"Columns {1} to {X.shape[1] - 1} of X_train must be castable in integer format."
            )

        for j in range(X.shape[1] - 1):
            max_cat_value = categorical_variables[:, j].max()
            if max_cat_value >= self.categorical_var_net.categorical_vocabulary_sizes[j]:
                raise ValueError(
                    f"Categorical variable at index {j} has value {max_cat_value} which exceeds the vocabulary size of {self.categorical_var_net.categorical_vocabulary_sizes[j]}."
                )

        return categorical_variables

    def _check_X(self, X: np.ndarray) -> np.ndarray:
        text = self._check_text_col(X)

        categorical_variables = None
        if self.categorical_var_net is not None:
            categorical_variables = self._check_categorical_variables(X)

        return {"text": text, "categorical_variables": categorical_variables}

    def _check_Y(self, Y):
        if self.ragged_multilabel:
            assert isinstance(
                Y, list
            ), "Y must be a list of lists for ragged multilabel classification."
            for row in Y:
                assert isinstance(row, list), "Each element of Y must be a list of labels."

            return Y

        else:
            assert isinstance(Y, np.ndarray), "Y must be a numpy array of shape (N,) or (N,1)."
            assert (
                len(Y.shape) == 1 or len(Y.shape) == 2
            ), "Y must be a numpy array of shape (N,) or (N, num_labels)."

            try:
                Y = Y.astype(int)
            except ValueError:
                logger.error("Y must be castable in integer format.")

            if Y.max() >= self.num_classes or Y.min() < 0:
                raise ValueError(
                    f"Y contains class labels outside the range [0, {self.num_classes - 1}]."
                )

            return Y

    def predict(
        self,
        X_test: np.ndarray,
        top_k=1,
        explain=False,
    ):
        """
        Args:
            X_test (np.ndarray): input data to predict on, shape (N,d) where the first column is text and the rest are categorical variables
            top_k (int): for each sentence, return the top_k most likely predictions (default: 1)
            explain (bool): launch gradient integration to have an explanation of the prediction (default: False)

        Returns: A dictionary containing the following fields:
                - predictions (torch.Tensor, shape (len(text), top_k)): A tensor containing the top_k most likely codes to the query.
                - confidence (torch.Tensor, shape (len(text), top_k)): A tensor array containing the corresponding confidence scores.
                - if explain is True:
                    - attributions (torch.Tensor, shape (len(text), top_k, seq_len)): A tensor containing the attributions for each token in the text.
        """

        if explain:
            return_offsets_mapping = True  # to be passed to the tokenizer
            return_word_ids = True
            if self.pytorch_model.text_embedder is None:
                raise RuntimeError(
                    "Explainability is not supported when the tokenizer outputs vectorized text directly. Please use a tokenizer that outputs token IDs."
                )
            else:
                if not HAS_CAPTUM:
                    raise ImportError(
                        "Captum is not installed and is required for explainability. Run 'pip install/uv add torchFastText[explainability]'."
                    )
                lig = LayerIntegratedGradients(
                    self.pytorch_model, self.pytorch_model.text_embedder.embedding_layer
                )  # initialize a Captum layer gradient integrator
        else:
            return_offsets_mapping = False
            return_word_ids = False

        X_test = self._check_X(X_test)
        text = X_test["text"]
        categorical_variables = X_test["categorical_variables"]

        self.pytorch_model.eval().cpu()

        tokenize_output = self.tokenizer.tokenize(
            text.tolist(),
            return_offsets_mapping=return_offsets_mapping,
            return_word_ids=return_word_ids,
        )

        if not isinstance(tokenize_output, TokenizerOutput):
            raise TypeError(
                f"Expected TokenizerOutput, got {type(tokenize_output)} from tokenizer.tokenize method."
            )

        encoded_text = tokenize_output.input_ids  # (batch_size, seq_len)
        attention_mask = tokenize_output.attention_mask  # (batch_size, seq_len)

        if categorical_variables is not None:
            categorical_vars = torch.tensor(
                categorical_variables, dtype=torch.float32
            )  # (batch_size, num_categorical_features)
        else:
            categorical_vars = torch.empty((encoded_text.shape[0], 0), dtype=torch.float32)

        pred = self.pytorch_model(
            encoded_text, attention_mask, categorical_vars
        )  # forward pass, contains the prediction scores (len(text), num_classes)

        label_scores = pred.detach().cpu().softmax(dim=1)  # convert to probabilities

        label_scores_topk = torch.topk(label_scores, k=top_k, dim=1)

        predictions = label_scores_topk.indices  # get the top_k most likely predictions
        confidence = torch.round(label_scores_topk.values, decimals=2)  # and their scores

        if explain:
            all_attributions = []
            for k in range(top_k):
                attributions = lig.attribute(
                    (encoded_text, attention_mask, categorical_vars),
                    target=torch.Tensor(predictions[:, k]).long(),
                )  # (batch_size, seq_len)
                attributions = attributions.sum(dim=-1)
                all_attributions.append(attributions.detach().cpu())

            all_attributions = torch.stack(all_attributions, dim=1)  # (batch_size, top_k, seq_len)

            return {
                "prediction": predictions,
                "confidence": confidence,
                "attributions": all_attributions,
                "offset_mapping": tokenize_output.offset_mapping,
                "word_ids": tokenize_output.word_ids,
            }
        else:
            return {
                "prediction": predictions,
                "confidence": confidence,
            }

    def save(self, path: Union[str, Path]) -> None:
        """Save the complete torchTextClassifiers instance to disk.

        This saves:
        - Model configuration
        - Tokenizer state
        - PyTorch Lightning checkpoint (if trained)
        - All other instance attributes

        Args:
            path: Directory path where the model will be saved

        Example:
            >>> ttc = torchTextClassifiers(tokenizer, model_config)
            >>> ttc.train(X_train, y_train, training_config)
            >>> ttc.save("my_model")
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save the checkpoint if model has been trained
        checkpoint_path = None
        if hasattr(self, "lightning_module"):
            checkpoint_path = path / "model_checkpoint.ckpt"
            # Save the current state as a checkpoint
            trainer = pl.Trainer()
            trainer.strategy.connect(self.lightning_module)
            trainer.save_checkpoint(checkpoint_path)

        # Prepare metadata to save
        metadata = {
            "model_config": self.model_config.to_dict(),
            "ragged_multilabel": self.ragged_multilabel,
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "categorical_vocabulary_sizes": self.categorical_vocabulary_sizes,
            "num_classes": self.num_classes,
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
            "device": str(self.device) if hasattr(self, "device") else None,
        }

        # Save metadata
        with open(path / "metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)

        # Save tokenizer
        tokenizer_path = path / "tokenizer.pkl"
        with open(tokenizer_path, "wb") as f:
            pickle.dump(self.tokenizer, f)

        logger.info(f"Model saved successfully to {path}")

    @classmethod
    def load(cls, path: Union[str, Path], device: str = "auto") -> "torchTextClassifiers":
        """Load a torchTextClassifiers instance from disk.

        Args:
            path: Directory path where the model was saved
            device: Device to load the model on ('auto', 'cpu', 'cuda', etc.)

        Returns:
            Loaded torchTextClassifiers instance

        Example:
            >>> loaded_ttc = torchTextClassifiers.load("my_model")
            >>> predictions = loaded_ttc.predict(X_test)
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Model directory not found: {path}")

        # Load metadata
        with open(path / "metadata.pkl", "rb") as f:
            metadata = pickle.load(f)

        # Load tokenizer
        with open(path / "tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)

        # Reconstruct model_config
        model_config = ModelConfig.from_dict(metadata["model_config"])

        # Create instance
        instance = cls(
            tokenizer=tokenizer,
            model_config=model_config,
            ragged_multilabel=metadata["ragged_multilabel"],
        )

        # Set device
        if device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)
        instance.device = device

        # Load checkpoint if it exists
        if metadata["checkpoint_path"]:
            checkpoint_path = path / "model_checkpoint.ckpt"
            if checkpoint_path.exists():
                # Load the checkpoint with weights_only=False since it's our own trusted checkpoint
                instance.lightning_module = TextClassificationModule.load_from_checkpoint(
                    str(checkpoint_path),
                    model=instance.pytorch_model,
                    weights_only=False,
                )
                instance.pytorch_model = instance.lightning_module.model.to(device)
                instance.checkpoint_path = str(checkpoint_path)
                logger.info(f"Model checkpoint loaded from {checkpoint_path}")
            else:
                logger.warning(f"Checkpoint file not found at {checkpoint_path}")

        logger.info(f"Model loaded successfully from {path}")
        return instance

    def __repr__(self):
        model_type = (
            self.lightning_module.__repr__()
            if hasattr(self, "lightning_module")
            else self.pytorch_model.__repr__()
        )

        tokenizer_info = self.tokenizer.__repr__()

        cat_forward_type = (
            self.categorical_var_net.forward_type.name
            if self.categorical_var_net is not None
            else "None"
        )

        lines = [
            "torchTextClassifiers(",
            f"  tokenizer = {tokenizer_info},",
            f"  model = {model_type},",
            f"  categorical_forward_type = {cat_forward_type},",
            f"  num_classes = {self.model_config.num_classes},",
            f"  embedding_dim = {self.embedding_dim},",
            ")",
        ]
        return "\n".join(lines)
