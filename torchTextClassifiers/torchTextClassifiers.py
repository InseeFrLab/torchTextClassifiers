import logging
import pickle
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union, cast

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
from torch import nn

from torchTextClassifiers.dataset import TextClassificationDataset
from torchTextClassifiers.model import TextClassificationModel, TextClassificationModule
from torchTextClassifiers.model.components import (
    AttentionConfig,
    CategoricalForwardType,
    CategoricalVariableNet,
    ClassificationHead,
    LabelAttentionConfig,
    SentenceEmbedder,
    SentenceEmbedderConfig,
    TokenEmbedder,
    TokenEmbedderConfig,
)
from torchTextClassifiers.tokenizers import BaseTokenizer, TokenizerOutput
from torchTextClassifiers.value_encoder import ValueEncoder

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
    num_classes: Optional[int | list[int]] = None
    categorical_vocabulary_sizes: Optional[List[int]] = None
    categorical_embedding_dims: Optional[Union[List[int], int]] = None
    attention_config: Optional[AttentionConfig] = None
    n_heads_label_attention: Optional[int] = None
    aggregation_method: Optional[str] = "mean"

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
    raw_categorical_inputs: Optional[bool] = True
    raw_labels: Optional[bool] = True
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
        value_encoder: Optional[ValueEncoder] = None,
    ):
        """Initialize the torchTextClassifiers instance.

        Args:
            tokenizer: A tokenizer instance for text preprocessing
            model_config: Configuration parameters for the text classification model
            ragged_multilabel: Whether to use ragged multilabel classification
            value_encoder: Optional ValueEncoder for encoding
                raw string (or mixed) categorical values to integers. Build it
                beforehand from DictEncoder or sklearn LabelEncoder instances and
                pass it here. If None, categorical columns in X must already be
                integer-encoded.

        Example:
            >>> from torchTextClassifiers import ModelConfig, TrainingConfig, torchTextClassifiers
            >>> from torchTextClassifiers.value_encoder import ValueEncoder, DictEncoder
            >>> # Build one DictEncoder per categorical feature
            >>> encoders = {str(i): DictEncoder({v: j for j, v in enumerate(sorted(set(X_categorical[:, i])))})
            ...             for i in range(X_categorical.shape[1])}
            >>> encoder = ValueEncoder(encoders)
            >>> model_config = ModelConfig(
            ...     embedding_dim=10,
            ...     categorical_vocabulary_sizes=encoder.vocabulary_sizes,
            ...     categorical_embedding_dims=[10, 5],
            ...     num_classes=10,
            ... )
            >>> ttc = torchTextClassifiers(
            ...     tokenizer=tokenizer,
            ...     model_config=model_config,
            ...     value_encoder=encoder,
            ... )
        """

        self.model_config = model_config
        self.tokenizer = tokenizer
        self.ragged_multilabel = ragged_multilabel
        self.value_encoder: ValueEncoder | None = value_encoder

        if hasattr(self.tokenizer, "trained"):
            if not self.tokenizer.trained:
                raise RuntimeError(
                    f"Tokenizer {type(self.tokenizer)} must be trained before initializing the classifier."
                )

        self.vocab_size = tokenizer.vocab_size
        self.embedding_dim = model_config.embedding_dim

        if self.value_encoder is not None:
            if (model_config.num_classes != self.value_encoder.num_classes) or (
                model_config.categorical_vocabulary_sizes != self.value_encoder.vocabulary_sizes
            ):
                logger.info(
                    "Overriding model_config num_classes and/or categorical_vocabulary_sizes with values from value_encoder."
                )
            self.categorical_vocabulary_sizes = self.value_encoder.vocabulary_sizes
            self.num_classes = self.value_encoder.num_classes
        else:
            self.categorical_vocabulary_sizes = model_config.categorical_vocabulary_sizes
            if model_config.num_classes is None:
                raise ValueError(
                    "num_classes must be specified in the model configuration if no value_encoder is provided."
                )
            self.num_classes = model_config.num_classes

        self.enable_label_attention = model_config.n_heads_label_attention is not None

        if self.tokenizer.output_vectorized:
            self.token_embedder = None
            logger.info(
                "Tokenizer outputs vectorized tokens; skipping TextEmbedder initialization."
            )
            self.embedding_dim = self.tokenizer.output_dim
        else:
            token_embedder_config = TokenEmbedderConfig(
                vocab_size=self.vocab_size,
                embedding_dim=self.embedding_dim,
                padding_idx=tokenizer.padding_idx,
                attention_config=model_config.attention_config,
            )
            sentence_embedder_config = SentenceEmbedderConfig(
                label_attention_config=LabelAttentionConfig(
                    n_head=model_config.n_heads_label_attention,
                    num_classes=model_config.num_classes,
                    embedding_dim=self.embedding_dim,
                )
                if self.enable_label_attention
                else None,
                aggregation_method=model_config.aggregation_method,
            )
            self.token_embedder = TokenEmbedder(
                token_embedder_config=token_embedder_config,
            )
            self.sentence_embedder = SentenceEmbedder(
                sentence_embedder_config=sentence_embedder_config
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
            num_classes=1
            if self.enable_label_attention
            else self.num_classes,  # output dim is 1 when using label attention, because embeddings are (num_classes, embedding_dim)
        )

        self.pytorch_model = TextClassificationModel(
            token_embedder=self.token_embedder,
            sentence_embedder=self.sentence_embedder,
            categorical_variable_net=self.categorical_var_net,
            classification_head=self.classification_head,
        )

    @classmethod
    def from_model(
        cls,
        tokenizer: BaseTokenizer,
        pytorch_model: nn.Module,
        value_encoder: Optional[ValueEncoder] = None,
        ragged_multilabel: Optional[bool] = False,
    ):
        """Initialize torchTextClassifiers from a pre-built PyTorch model.

        This method allows users to create a torchTextClassifiers instance using a pre-built PyTorch model that may not follow the standard architecture expected by the main constructor. The provided model should be compatible with the input format used in the predict method (i.e., it should accept tokenized text and categorical variables as input).

        Args:
            tokenizer: A tokenizer instance for text preprocessing
            pytorch_model: A pre-built PyTorch model to be used for predictions
            value_encoder: Optional ValueEncoder for encoding raw string (or mixed) categorical values to integers. Build it beforehand from DictEncoder or sklearn LabelEncoder instances and pass it here. If None, categorical columns in X must already be integer-encoded.

        Returns:
            An instance of torchTextClassifiers initialized with the provided model and tokenizer.
        """
        instance = cls.__new__(cls)
        instance.tokenizer = tokenizer
        instance.pytorch_model = pytorch_model
        instance.num_classes = pytorch_model.num_classes
        instance.categorical_var_net = cast(
            Optional[CategoricalVariableNet], pytorch_model.categorical_variable_net
        )
        instance.value_encoder = value_encoder
        instance.ragged_multilabel = ragged_multilabel
        instance._custom_model = True
        instance.enable_label_attention = False
        instance.categorical_vocabulary_sizes = (
            instance.categorical_var_net.categorical_vocabulary_sizes
            if instance.categorical_var_net is not None
            else None
        )
        return instance

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
        X_train, y_train = self._check_XY(
            X_train, y_train, training_config.raw_categorical_inputs, training_config.raw_labels
        )

        if X_val is not None:
            assert y_val is not None, "y_val must be provided if X_val is provided."
        if y_val is not None:
            assert X_val is not None, "X_val must be provided if y_val is provided."

        X_val: Optional[Dict[str, Any]] = None
        if X_val is not None and y_val is not None:
            X_val, y_val = self._check_XY(X_val, y_val)

        if (
            (X_train["categorical_variables"] is not None)
            and (X_val is not None)
            and (X_val["categorical_variables"] is not None)
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

    def _check_XY(
        self, X: np.ndarray, Y: np.ndarray, raw_categorical_inputs, raw_labels
    ) -> Tuple[Dict[str, Any], np.ndarray]:
        X_checked = self._check_X(X, raw_categorical_inputs)
        Y_checked = self._check_Y(Y, raw_labels)

        if X_checked["text"].shape[0] != len(Y_checked):
            raise ValueError("X_train and y_train must have the same number of observations.")

        return X_checked, Y_checked

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

    def _check_categorical_variables(
        self, X: np.ndarray, raw_categorical_inputs: bool
    ) -> np.ndarray:
        """Validate and encode categorical variables from X.

        If a ``value_encoder`` was provided at initialization, raw string
        or mixed values are encoded to integers via that encoder.  Otherwise the
        categorical columns must already be integer-encodable.

        Args:
            X: Full input array whose first column is text and whose remaining
               columns are categorical variables.

        Returns:
            Integer-encoded categorical array of shape (N, n_cat_features).

        Raises:
            ValueError: If the number of categorical features does not match the
                model configuration, if values exceed vocabulary bounds, or if
                values cannot be cast to integers and no encoder was provided.
        """
        assert self.categorical_var_net is not None

        num_cat_vars = X.shape[1] - 1 if X.ndim > 1 else 0

        if num_cat_vars != self.categorical_var_net.num_categorical_features:
            raise ValueError(
                f"X must have the same number of categorical variables as the number of "
                f"embedding layers in the categorical net: ({self.categorical_var_net.num_categorical_features})."
            )

        if raw_categorical_inputs:
            if self.value_encoder is None:
                raise ValueError(
                    "Raw categorical input encoding is enabled, but no value_encoder was provided. Please provide a ValueEncoder to encode raw categorical values to integers."
                )
            categorical_variables = self.value_encoder.transform(X[:, 1:]).astype(int)
        else:
            categorical_variables = X[:, 1:].astype(int)

        for j in range(num_cat_vars):
            max_cat_value = categorical_variables[:, j].max()
            if max_cat_value >= self.categorical_var_net.categorical_vocabulary_sizes[j]:
                raise ValueError(
                    f"Categorical variable at index {j} has value {max_cat_value} which exceeds "
                    f"the vocabulary size of {self.categorical_var_net.categorical_vocabulary_sizes[j]}."
                )

        return categorical_variables

    def _check_X(self, X: np.ndarray, raw_categorical_inputs: bool) -> Dict[str, Any]:
        text = self._check_text_col(X)

        categorical_variables = None
        if self.categorical_var_net is not None:
            categorical_variables = self._check_categorical_variables(X, raw_categorical_inputs)

        return {"text": text, "categorical_variables": categorical_variables}

    def _check_Y(self, Y, raw_labels: bool) -> np.ndarray:
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

            if raw_labels:
                if self.value_encoder is None:
                    raise ValueError(
                        "Raw label encoding is enabled, but no value_encoder was provided. Please provide a ValueEncoder to encode raw labels to integers."
                    )
                Y = self.value_encoder.transform_labels(Y)
            Y = Y.astype(int)

            if isinstance(self.num_classes, list):
                num_classes_arr = np.array(self.num_classes)

                print(Y, num_classes_arr)
                if (Y.max(axis=0) >= num_classes_arr).any() or Y.min() < 0:
                    raise ValueError(
                        f"Y contains class labels outside the expected per-level ranges "
                        f"[0, {[nc - 1 for nc in self.num_classes]}]."
                    )
            elif Y.max() >= self.num_classes or Y.min() < 0:
                raise ValueError(
                    f"Y contains class labels outside the range [0, {self.num_classes - 1}]."
                )

            return Y

    def predict(
        self,
        X_test: np.ndarray,
        raw_categorical_inputs: bool = True,
        top_k=1,
        explain_with_label_attention: bool = False,
        explain_with_captum=False,
    ):
        """
        Args:
            X_test (np.ndarray): input data to predict on, shape (N,d) where the first column is text and the rest are categorical variables
            top_k (int): for each sentence, return the top_k most likely predictions (default: 1)
            explain_with_label_attention (bool): if enabled, use attention matrix labels x tokens to have an explanation of the prediction (default: False)
            explain_with_captum (bool): launch gradient integration with Captum for explanation (default: False)

        Returns: A dictionary containing the following fields:
                - predictions (torch.Tensor, shape (len(text), top_k)): A tensor containing the top_k most likely codes to the query.
                - confidence (torch.Tensor, shape (len(text), top_k)): A tensor array containing the corresponding confidence scores.
                - if explain is True:
                    - attributions (torch.Tensor, shape (len(text), top_k, seq_len)): A tensor containing the attributions for each token in the text.
        """

        explain = explain_with_label_attention or explain_with_captum
        if explain:
            return_offsets_mapping = True  # to be passed to the tokenizer
            return_word_ids = True
            if self.pytorch_model.token_embedder is None:
                raise RuntimeError(
                    "Explainability is not supported when the tokenizer outputs vectorized text directly. Please use a tokenizer that outputs token IDs."
                )
            else:
                if explain_with_captum:
                    if not HAS_CAPTUM:
                        raise ImportError(
                            "Captum is not installed and is required for explainability. Run 'pip install/uv add torchFastText[explainability]'."
                        )
                    lig = LayerIntegratedGradients(
                        self.pytorch_model, self.pytorch_model.token_embedder.embedding_layer
                    )  # initialize a Captum layer gradient integrator
                if explain_with_label_attention:
                    if not self.enable_label_attention:
                        raise RuntimeError(
                            "Label attention explainability is enabled, but the model was not configured with label attention. Please enable label attention in the model configuration during initialization and retrain."
                        )
        else:
            return_offsets_mapping = False
            return_word_ids = False

        X_test = self._check_X(X_test, raw_categorical_inputs)
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

        model_output = self.pytorch_model(
            encoded_text,
            attention_mask,
            categorical_vars,
            return_label_attention_matrix=explain_with_label_attention,
        )  # forward pass, contains the prediction scores (len(text), num_classes)

        # Multilevel custom model: returns a list of per-level logit tensors.
        # Each level may have a different number of classes, so we process them
        # separately then stack into (N, top_k, n_levels) before decoding.
        if isinstance(model_output, list):
            int_preds_per_level: List[np.ndarray] = []
            conf_per_level: List[torch.Tensor] = []
            for level_logits in cast(List[torch.Tensor], model_output):
                scores = level_logits.detach().cpu().softmax(dim=-1)
                level_topk = torch.topk(scores, k=top_k, dim=-1)
                int_preds_per_level.append(level_topk.indices.numpy())  # (N, top_k)
                conf_per_level.append(level_topk.values)  # (N, top_k)

            # (N, top_k, n_levels)
            int_preds_stacked = np.stack(int_preds_per_level, axis=-1)
            conf_stacked = torch.stack(conf_per_level, dim=-1)

            if self.value_encoder is not None:
                predictions = self.value_encoder.inverse_transform_labels(int_preds_stacked)
            else:
                predictions = int_preds_stacked

            return {
                "prediction": predictions,
                "confidence": torch.round(conf_stacked, decimals=2),
            }

        pred = (
            model_output["logits"] if explain_with_label_attention else model_output
        )  # (batch_size, num_classes)

        label_attention_matrix = (
            model_output["label_attention_matrix"] if explain_with_label_attention else None
        )

        label_scores = pred.detach().cpu().softmax(dim=1)  # convert to probabilities

        label_scores_topk = torch.topk(label_scores, k=top_k, dim=1)

        integer_predictions = label_scores_topk.indices  # integer class indices (needed for captum)
        if self.value_encoder is not None:
            predictions = self.value_encoder.inverse_transform_labels(integer_predictions.numpy())
        else:
            predictions = integer_predictions

        confidence = torch.round(label_scores_topk.values, decimals=2)  # and their scores

        if explain:
            if explain_with_captum:
                # Captum explanations
                captum_attributions = []
                for k in range(top_k):
                    attributions = lig.attribute(
                        (encoded_text, attention_mask, categorical_vars),
                        target=integer_predictions[:, k],
                    )  # (batch_size, seq_len)
                    attributions = attributions.sum(dim=-1)
                    captum_attributions.append(attributions.detach().cpu())

                captum_attributions = torch.stack(
                    captum_attributions, dim=1
                )  # (batch_size, top_k, seq_len)
            else:
                captum_attributions = None

            return {
                "prediction": predictions,
                "confidence": confidence,
                "captum_attributions": captum_attributions,
                "label_attention_attributions": label_attention_matrix,
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

        is_custom_model = getattr(self, "_custom_model", False)

        # Custom models: save architecture as pickle + weights as state_dict.
        # Standard models: save a full PyTorch Lightning checkpoint.
        checkpoint_path = None
        if is_custom_model:
            with open(path / "pytorch_model.pkl", "wb") as f:
                pickle.dump(self.pytorch_model, f)
            torch.save(self.pytorch_model.state_dict(), path / "model_weights.pt")
        elif hasattr(self, "lightning_module"):
            checkpoint_path = path / "model_checkpoint.ckpt"
            trainer = pl.Trainer()
            trainer.strategy.connect(self.lightning_module)
            trainer.save_checkpoint(checkpoint_path)

        metadata: Dict[str, Any] = {
            "is_custom_model": is_custom_model,
            "loss": self.lightning_module.loss if hasattr(self, "lightning_module") else None,
            "ragged_multilabel": self.ragged_multilabel,
            "num_classes": self.num_classes,
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
            "device": str(self.device) if hasattr(self, "device") else None,
            "has_value_encoder": self.value_encoder is not None,
        }

        if not is_custom_model:
            metadata.update(
                {
                    "model_config": self.model_config.to_dict(),
                    "vocab_size": self.vocab_size,
                    "embedding_dim": self.embedding_dim,
                    "categorical_vocabulary_sizes": self.categorical_vocabulary_sizes,
                }
            )

        with open(path / "metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)

        tokenizer_path = path / "tokenizer.pkl"
        with open(tokenizer_path, "wb") as f:
            pickle.dump(self.tokenizer, f)

        if self.value_encoder is not None:
            with open(path / "value_encoder.pkl", "wb") as f:
                pickle.dump(self.value_encoder, f)

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

        with open(path / "metadata.pkl", "rb") as f:
            metadata = pickle.load(f)

        with open(path / "tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)

        resolved_device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else torch.device(device)
        )

        value_encoder = None
        if metadata.get("has_value_encoder"):
            encoder_path = path / "value_encoder.pkl"
            if encoder_path.exists():
                with open(encoder_path, "rb") as f:
                    value_encoder = pickle.load(f)

        if metadata.get("is_custom_model", False):
            with open(path / "pytorch_model.pkl", "rb") as f:
                pytorch_model = pickle.load(f)
            weights_path = path / "model_weights.pt"
            if weights_path.exists():
                pytorch_model.load_state_dict(torch.load(weights_path, weights_only=True))
                logger.info(f"Model weights loaded from {weights_path}")
            instance = cls.from_model(
                tokenizer=tokenizer,
                pytorch_model=pytorch_model,
                value_encoder=value_encoder,
                ragged_multilabel=metadata["ragged_multilabel"],
            )
            instance.device = resolved_device
            pytorch_model.to(resolved_device)
            logger.info(f"Model loaded successfully from {path}")
            return instance

        model_config = ModelConfig.from_dict(metadata["model_config"])

        instance = cls(
            tokenizer=tokenizer,
            model_config=model_config,
            ragged_multilabel=metadata["ragged_multilabel"],
            value_encoder=value_encoder,
        )

        instance.device = resolved_device

        if metadata.get("checkpoint_path"):
            checkpoint_path = path / "model_checkpoint.ckpt"
            if checkpoint_path.exists():
                loss = metadata.get("loss") or torch.nn.CrossEntropyLoss()
                instance.lightning_module = TextClassificationModule.load_from_checkpoint(
                    str(checkpoint_path),
                    model=instance.pytorch_model,
                    loss=loss,
                    weights_only=False,
                )
                instance.pytorch_model = instance.lightning_module.model.to(resolved_device)
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
