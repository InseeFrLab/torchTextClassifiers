"""FastText model components.

This module contains the PyTorch model, Lightning module, and dataset classes
for FastText classification. Consolidates what was previously in pytorch_model.py,
lightning_module.py, and dataset.py.
"""

import os
import logging
from typing import List, Union
import torch
import pytorch_lightning as pl
from torch import nn
from torchmetrics import Accuracy

try:
    from captum.attr import LayerIntegratedGradients
    HAS_CAPTUM = True
except ImportError:
    HAS_CAPTUM = False

from ...utilities.utils import (
    compute_preprocessed_word_score,
    compute_word_score,
    explain_continuous,
)
from ...utilities.checkers import validate_categorical_inputs

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
)


# ============================================================================
# PyTorch Model
# ============================================================================

class FastTextModel(nn.Module):
    """FastText Pytorch Model."""

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        tokenizer=None,
        num_rows: int = None,
        categorical_vocabulary_sizes: List[int] = None,
        categorical_embedding_dims: Union[List[int], int] = None,
        padding_idx: int = 0,
        sparse: bool = True,
        direct_bagging: bool = False,
    ):
        """
        Constructor for the FastTextModel class.

        Args:
            embedding_dim (int): Dimension of the text embedding space.
            buckets (int): Number of rows in the embedding matrix.
            num_classes (int): Number of classes.
            categorical_vocabulary_sizes (List[int]): List of the number of
                modalities for additional categorical features.
            padding_idx (int, optional): Padding index for the text
                descriptions. Defaults to 0.
            sparse (bool): Indicates if Embedding layer is sparse.
            direct_bagging (bool): Use EmbeddingBag instead of Embedding for the text embedding.
        """
        super(FastTextModel, self).__init__()

        if isinstance(categorical_embedding_dims, int):
            self.average_cat_embed = True  # if provided categorical embedding dims is an int, average the categorical embeddings before concatenating to sentence embedding
        else:
            self.average_cat_embed = False

        categorical_vocabulary_sizes, categorical_embedding_dims, num_categorical_features = (
            validate_categorical_inputs(
                categorical_vocabulary_sizes,
                categorical_embedding_dims,
                num_categorical_features=None,
            )
        )

        assert isinstance(categorical_embedding_dims, list) or categorical_embedding_dims is None, (
            "categorical_embedding_dims must be a list of int at this stage"
        )

        if categorical_embedding_dims is None:
            self.average_cat_embed = False

        if tokenizer is None:
            if num_rows is None:
                raise ValueError(
                    "Either tokenizer or num_rows must be provided (number of rows in the embedding matrix)."
                )
        else:
            if num_rows is not None:
                if num_rows != tokenizer.num_tokens:
                    logger.warning(
                        "num_rows is different from the number of tokens in the tokenizer. Using provided num_rows."
                    )

        self.num_rows = num_rows

        self.num_classes = num_classes
        self.padding_idx = padding_idx
        self.tokenizer = tokenizer
        self.embedding_dim = embedding_dim
        self.direct_bagging = direct_bagging
        self.sparse = sparse

        self.categorical_embedding_dims = categorical_embedding_dims

        self.embeddings = (
            nn.Embedding(
                embedding_dim=embedding_dim,
                num_embeddings=num_rows,
                padding_idx=padding_idx,
                sparse=sparse,
            )
            if not direct_bagging
            else nn.EmbeddingBag(
                embedding_dim=embedding_dim,
                num_embeddings=num_rows,
                padding_idx=padding_idx,
                sparse=sparse,
                mode="mean",
            )
        )

        self.categorical_embedding_layers = {}

        # Entry dim for the last layer:
        #   1. embedding_dim if no categorical variables or summing the categrical embeddings to sentence embedding
        #   2. embedding_dim + cat_embedding_dim if averaging the categorical embeddings before concatenating to sentence embedding (categorical_embedding_dims is a int)
        #   3. embedding_dim + sum(categorical_embedding_dims) if concatenating individually the categorical embeddings to sentence embedding (no averaging, categorical_embedding_dims is a list)
        dim_in_last_layer = embedding_dim
        if self.average_cat_embed:
            dim_in_last_layer += categorical_embedding_dims[0]

        if categorical_vocabulary_sizes is not None:
            self.no_cat_var = False
            for var_idx, num_rows in enumerate(categorical_vocabulary_sizes):
                if categorical_embedding_dims is not None:
                    emb = nn.Embedding(
                        embedding_dim=categorical_embedding_dims[var_idx], num_embeddings=num_rows
                    )  # concatenate to sentence embedding
                    if not self.average_cat_embed:
                        dim_in_last_layer += categorical_embedding_dims[var_idx]
                else:
                    emb = nn.Embedding(
                        embedding_dim=embedding_dim, num_embeddings=num_rows
                    )  # sum to sentence embedding
                self.categorical_embedding_layers[var_idx] = emb
                setattr(self, "emb_{}".format(var_idx), emb)
        else:
            self.no_cat_var = True

        self.fc = nn.Linear(dim_in_last_layer, num_classes)

    def forward(self, encoded_text: torch.Tensor, additional_inputs: torch.Tensor) -> torch.Tensor:
        """
        Memory-efficient forward pass implementation.

        Args:
            encoded_text (torch.Tensor[Long]), shape (batch_size, seq_len): Tokenized + padded text
            additional_inputs (torch.Tensor[Long]): Additional categorical features, (batch_size, num_categorical_features)

        Returns:
            torch.Tensor: Model output scores for each class
        """
        batch_size = encoded_text.size(0)

        # Ensure correct dtype and device once
        if encoded_text.dtype != torch.long:
            encoded_text = encoded_text.to(torch.long)

        # Compute text embeddings
        if self.direct_bagging:
            x_text = self.embeddings(encoded_text)  # (batch_size, embedding_dim)
        else:
            # Compute embeddings and averaging in a memory-efficient way
            x_text = self.embeddings(encoded_text)  # (batch_size, seq_len, embedding_dim)
            # Calculate non-zero tokens mask once
            non_zero_mask = (x_text.sum(-1) != 0).float()  # (batch_size, seq_len)
            token_counts = non_zero_mask.sum(-1, keepdim=True)  # (batch_size, 1)

            # Sum and average in place
            x_text = (x_text * non_zero_mask.unsqueeze(-1)).sum(
                dim=1
            )  # (batch_size, embedding_dim)
            x_text = torch.div(x_text, token_counts.clamp(min=1.0))
            x_text = torch.nan_to_num(x_text, 0.0)

        # Handle categorical variables efficiently
        if not self.no_cat_var and additional_inputs.numel() > 0:
            cat_embeds = []
            # Process categorical embeddings in batch
            for i, (_, embed_layer) in enumerate(self.categorical_embedding_layers.items()):
                cat_input = additional_inputs[:, i].long()
                
                # Check if categorical values are within valid range and clamp if needed
                vocab_size = embed_layer.num_embeddings
                max_val = cat_input.max().item()
                min_val = cat_input.min().item()
                
                if max_val >= vocab_size or min_val < 0:
                    logger.warning(f"Categorical feature {i}: values range [{min_val}, {max_val}] exceed vocabulary size {vocab_size}. Clamping to valid range [0, {vocab_size - 1}]")
                    # Clamp values to valid range
                    cat_input = torch.clamp(cat_input, 0, vocab_size - 1)
                
                cat_embed = embed_layer(cat_input)
                if cat_embed.dim() > 2:
                    cat_embed = cat_embed.squeeze(1)
                cat_embeds.append(cat_embed)

            if cat_embeds:  # If we have categorical embeddings
                if self.categorical_embedding_dims is not None:
                    if self.average_cat_embed:
                        # Stack and average in one operation
                        x_cat = torch.stack(cat_embeds, dim=0).mean(dim=0)
                        x_combined = torch.cat([x_text, x_cat], dim=1)
                    else:
                        # Optimize concatenation
                        x_combined = torch.cat([x_text] + cat_embeds, dim=1)
                else:
                    # Sum embeddings efficiently
                    x_combined = x_text + torch.stack(cat_embeds, dim=0).sum(dim=0)
            else:
                x_combined = x_text
        else:
            x_combined = x_text

        # Final linear layer
        return self.fc(x_combined)

    def predict(
        self,
        text: List[str],
        categorical_variables: List[List[int]],
        top_k=1,
        explain=False,
        preprocess=True,
    ):
        """
        Args:
            text (List[str]): A list of text observations.
            params (Optional[Dict[str, Any]]): Additional parameters to
                pass to the model for inference.
            top_k (int): for each sentence, return the top_k most likely predictions (default: 1)
            explain (bool): launch gradient integration to have an explanation of the prediction (default: False)
            preprocess (bool): If True, preprocess text. Needs unidecode library.

        Returns:
        if explain is False:
            predictions (torch.Tensor, shape (len(text), top_k)): A tensor containing the top_k most likely codes to the query.
            confidence (torch.Tensor, shape (len(text), top_k)): A tensor array containing the corresponding confidence scores.
        if explain is True:
            predictions (torch.Tensor, shape (len(text), top_k)): Containing the top_k most likely codes to the query.
            confidence (torch.Tensor, shape (len(text), top_k)): Corresponding confidence scores.
            all_attributions (torch.Tensor, shape (len(text), top_k, seq_len)): A tensor containing the attributions for each token in the text.
            x (torch.Tensor): A tensor containing the token indices of the text.
            id_to_token_dicts (List[Dict[int, str]]): A list of dictionaries mapping token indices to tokens (one for each sentence).
            token_to_id_dicts (List[Dict[str, int]]): A list of dictionaries mapping tokens to token indices: the reverse of those in id_to_token_dicts.
            text (list[str]): A plist containing the preprocessed text (one line for each sentence).
        """

        flag_change_embed = False
        if explain:
            if not HAS_CAPTUM:
                raise ImportError(
                    "Captum is not installed and is required for explainability. Run 'pip install torchFastText[explainability]'."
                )
            if self.direct_bagging:
                # Get back the classical embedding layer for explainability
                new_embed_layer = nn.Embedding(
                    embedding_dim=self.embedding_dim,
                    num_embeddings=self.num_rows,
                    padding_idx=self.padding_idx,
                    sparse=self.sparse,
                )
                new_embed_layer.load_state_dict(
                    self.embeddings.state_dict()
                )  # No issues, as exactly the same parameters
                self.embeddings = new_embed_layer
                self.direct_bagging = (
                    False  # To inform the forward pass that we are not using EmbeddingBag anymore
                )
                flag_change_embed = True

            lig = LayerIntegratedGradients(
                self, self.embeddings
            )  # initialize a Captum layer gradient integrator

        self.eval()
        batch_size = len(text)

        indices_batch, id_to_token_dicts, token_to_id_dicts = self.tokenizer.tokenize(
            text, text_tokens=False, preprocess=preprocess
        )

        padding_index = (
            self.tokenizer.get_buckets() + self.tokenizer.get_nwords()
        )  # padding index, the integer value of the padding token

        padded_batch = torch.nn.utils.rnn.pad_sequence(
            indices_batch,
            batch_first=True,
            padding_value=padding_index,
        )  # (batch_size, seq_len) - Tokenized (int) + padded text

        x = padded_batch

        if not self.no_cat_var:
            other_features = []
            # Transpose categorical_variables to iterate over features instead of samples
            categorical_variables_transposed = categorical_variables.T
            for i, categorical_variable in enumerate(categorical_variables_transposed):
                other_features.append(
                    torch.tensor(categorical_variable).reshape(batch_size, -1).to(torch.int64)
                )

            other_features = torch.stack(other_features).reshape(batch_size, -1).long()
        else:
            other_features = torch.empty(batch_size)

        pred = self(
            x, other_features
        )  # forward pass, contains the prediction scores (len(text), num_classes)
        label_scores = pred.detach().cpu()
        label_scores_topk = torch.topk(label_scores, k=top_k, dim=1)

        predictions = label_scores_topk.indices  # get the top_k most likely predictions
        confidence = torch.round(label_scores_topk.values, decimals=2)  # and their scores

        if explain:
            assert not self.direct_bagging, "Direct bagging should be False for explainability"
            all_attributions = []
            for k in range(top_k):
                attributions = lig.attribute(
                    (x, other_features), target=torch.Tensor(predictions[:, k]).long()
                )  # (batch_size, seq_len)
                attributions = attributions.sum(dim=-1)
                all_attributions.append(attributions.detach().cpu())

            all_attributions = torch.stack(all_attributions, dim=1)  # (batch_size, top_k, seq_len)

            # Get back to initial embedding layer:
            # EmbeddingBag -> Embedding -> EmbeddingBag
            # or keep Embedding with no change
            if flag_change_embed:
                new_embed_layer = nn.EmbeddingBag(
                    embedding_dim=self.embedding_dim,
                    num_embeddings=self.num_rows,
                    padding_idx=self.padding_idx,
                    sparse=self.sparse,
                )
                new_embed_layer.load_state_dict(
                    self.embeddings.state_dict()
                )  # No issues, as exactly the same parameters
                self.embeddings = new_embed_layer
                self.direct_bagging = True
            return (
                predictions,
                confidence,
                all_attributions,
                x,
                id_to_token_dicts,
                token_to_id_dicts,
                text,
            )
        else:
            return predictions, confidence

    def predict_and_explain(self, text, categorical_variables, top_k=1, n=5, cutoff=0.65):
        """
        Args:
            text (List[str]): A list of sentences.
            params (Optional[Dict[str, Any]]): Additional parameters to
                pass to the model for inference.
            top_k (int): for each sentence, return the top_k most likely predictions (default: 1)
            n (int): mapping processed to original words: max number of candidate processed words to consider per original word (default: 5)
            cutoff (float): mapping processed to original words: minimum similarity score to consider a candidate processed word (default: 0.75)

        Returns:
            predictions (torch.Tensor, shape (len(text), top_k)): Containing the top_k most likely codes to the query.
            confidence (torch.Tensor, shape (len(text), top_k)): Corresponding confidence scores.
            all_scores (List[List[List[float]]]): For each sentence, list of the top_k lists of attributions for each word in the sentence (one for each pred).
        """

        # Step 1: Get the predictions, confidence scores and attributions at token level
        (
            pred,
            confidence,
            all_attr,
            tokenized_text,
            id_to_token_dicts,
            token_to_id_dicts,
            processed_text,
        ) = self.predict(
            text=text, categorical_variables=categorical_variables, top_k=top_k, explain=True
        )

        tokenized_text_tokens = self.tokenizer._tokenized_text_in_tokens(
            tokenized_text, id_to_token_dicts
        )

        # Step 2: Map the attributions at token level to the processed words
        processed_word_to_score_dicts, processed_word_to_token_idx_dicts = (
            compute_preprocessed_word_score(
                processed_text,
                tokenized_text_tokens,
                all_attr,
                id_to_token_dicts,
                token_to_id_dicts,
                min_n=self.tokenizer.min_n,
                padding_index=self.padding_idx,
                end_of_string_index=0,
            )
        )

        # Step 3: Map the processed words to the original words
        all_scores, orig_to_processed_mappings = compute_word_score(
            processed_word_to_score_dicts, text, n=n, cutoff=cutoff
        )

        # Step 2bis: Get the attributions at letter level
        all_scores_letters = explain_continuous(
            text,
            processed_text,
            tokenized_text_tokens,
            orig_to_processed_mappings,
            processed_word_to_token_idx_dicts,
            all_attr,
            top_k,
        )

        return pred, confidence, all_scores, all_scores_letters


# ============================================================================
# PyTorch Lightning Module
# ============================================================================

class FastTextModule(pl.LightningModule):
    """Pytorch Lightning Module for FastTextModel."""

    def __init__(
        self,
        model: FastTextModel,
        loss,
        optimizer,
        optimizer_params,
        scheduler,
        scheduler_params,
        scheduler_interval="epoch",
        **kwargs,
    ):
        """
        Initialize FastTextModule.

        Args:
            model: Model.
            loss: Loss
            optimizer: Optimizer
            optimizer_params: Optimizer parameters.
            scheduler: Scheduler.
            scheduler_params: Scheduler parameters.
            scheduler_interval: Scheduler interval.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model", "loss"])

        self.model = model
        self.loss = loss
        self.accuracy_fn = Accuracy(task="multiclass", num_classes=self.model.num_classes)
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.scheduler = scheduler
        self.scheduler_params = scheduler_params
        self.scheduler_interval = scheduler_interval

    def forward(self, inputs) -> torch.Tensor:
        """
        Perform forward-pass.

        Args:
            batch (List[torch.LongTensor]): Batch to perform forward-pass on.

        Returns (torch.Tensor): Prediction.
        """
        return self.model(inputs[0], inputs[1])

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """
        Training step.

        Args:
            batch (List[torch.LongTensor]): Training batch.
            batch_idx (int): Batch index.

        Returns (torch.Tensor): Loss tensor.
        """

        inputs, targets = batch[:-1], batch[-1]
        outputs = self.forward(inputs)
        loss = self.loss(outputs, targets)
        self.log("train_loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        accuracy = self.accuracy_fn(outputs, targets)
        self.log("train_accuracy", accuracy, on_epoch=True, on_step=False, prog_bar=True)

        torch.cuda.empty_cache()

        return loss

    def validation_step(self, batch, batch_idx: int):
        """
        Validation step.

        Args:
            batch (List[torch.LongTensor]): Validation batch.
            batch_idx (int): Batch index.

        Returns (torch.Tensor): Loss tensor.
        """
        inputs, targets = batch[:-1], batch[-1]
        outputs = self.forward(inputs)
        loss = self.loss(outputs, targets)
        self.log("val_loss", loss, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)

        accuracy = self.accuracy_fn(outputs, targets)
        self.log("val_accuracy", accuracy, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx: int):
        """
        Test step.

        Args:
            batch (List[torch.LongTensor]): Test batch.
            batch_idx (int): Batch index.

        Returns (torch.Tensor): Loss tensor.
        """
        inputs, targets = batch[:-1], batch[-1]
        outputs = self.forward(inputs)
        loss = self.loss(outputs, targets)

        accuracy = self.accuracy_fn(outputs, targets)

        return loss, accuracy

    def configure_optimizers(self):
        """
        Configure optimizer for Pytorch lighting.

        Returns: Optimizer and scheduler for pytorch lighting.
        """
        optimizer = self.optimizer(self.parameters(), **self.optimizer_params)
        
        # Only use scheduler if it's not ReduceLROnPlateau or if we can ensure val_loss is available
        # For complex training setups, sometimes val_loss is not available every epoch
        if hasattr(self.scheduler, '__name__') and 'ReduceLROnPlateau' in self.scheduler.__name__:
            # For ReduceLROnPlateau, use train_loss as it's always available
            scheduler = self.scheduler(optimizer, **self.scheduler_params)
            scheduler_config = {
                "scheduler": scheduler,
                "monitor": "train_loss",
                "interval": self.scheduler_interval,
            }
            return [optimizer], [scheduler_config]
        else:
            # For other schedulers (StepLR, etc.), no monitoring needed
            scheduler = self.scheduler(optimizer, **self.scheduler_params)
            return [optimizer], [scheduler]


# ============================================================================
# Dataset
# ============================================================================

class FastTextModelDataset(torch.utils.data.Dataset):
    """FastTextModelDataset class."""

    def __init__(
        self,
        categorical_variables: List[List[int]],
        texts: List[str],
        tokenizer,  # NGramTokenizer
        outputs: List[int] = None,
        **kwargs,
    ):
        """
        Constructor for the TorchDataset class.

        Args:
            categorical_variables (List[List[int]]): The elements of this list
                are the values of each categorical variable across the dataset.
            text (List[str]): List of text descriptions.
            y (List[int]): List of outcomes.
            tokenizer (Tokenizer): Tokenizer.
        """

        if categorical_variables is not None and len(categorical_variables) != len(texts):
            raise ValueError("Categorical variables and texts must have the same length.")
        
        if outputs is not None and len(outputs) != len(texts):
            raise ValueError("Outputs and texts must have the same length.")
            
        self.categorical_variables = categorical_variables
        self.texts = texts
        self.outputs = outputs
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        """
        Returns length of the data.

        Returns:
            int: Number of observations.
        """
        return len(self.texts)

    def __str__(self) -> str:
        """
        Returns description of the Dataset.

        Returns:
            str: Description.
        """
        return f"<FastTextModelDataset(N={len(self)})>"

    def __getitem__(self, index: int) -> List:
        """
        Returns observation for a given index.

        Args:
            index (int): Index.

        Returns:
            List[int, str]: Observation with given index.
        """
        categorical_variables = (
            self.categorical_variables[index] if self.categorical_variables is not None else None
        )
        text = self.texts[index]

        if self.outputs is not None:
            y = self.outputs[index]
            return text, categorical_variables, y
        else:
            return text, categorical_variables

    def collate_fn(self, batch):
        """
        Efficient batch processing without explicit loops.

        Args:
            batch: Data batch.

        Returns:
            Tuple[torch.LongTensor]: Observation with given index.
        """

        # Unzip the batch in one go using zip(*batch)
        if self.outputs is not None:
            text, *categorical_vars, y = zip(*batch)
        else:
            text, *categorical_vars = zip(*batch)

        # Convert text to indices in parallel using map
        indices_batch = list(map(lambda x: self.tokenizer.indices_matrix(x)[0], text))

        # Get padding index once
        padding_index = self.tokenizer.get_buckets() + self.tokenizer.get_nwords()

        # Pad sequences efficiently
        padded_batch = torch.nn.utils.rnn.pad_sequence(
            indices_batch,
            batch_first=True,
            padding_value=padding_index,
        )

        # Handle categorical variables efficiently
        if self.categorical_variables is not None:
            categorical_tensors = torch.stack(
                [
                    torch.tensor(cat_var, dtype=torch.float32)
                    for cat_var in categorical_vars[
                        0
                    ]  # Access first element since zip returns tuple
                ]
            )
        else:
            categorical_tensors = torch.empty(
                padded_batch.shape[0], 1, dtype=torch.float32, device=padded_batch.device
            )

        if self.outputs is not None:
            # Convert labels to tensor in one go
            y = torch.tensor(y, dtype=torch.long)
            return (padded_batch, categorical_tensors, y)
        else:
            return (padded_batch, categorical_tensors)

    def create_dataloader(
        self,
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
        num_workers: int = os.cpu_count() - 1,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        **kwargs,
    ) -> torch.utils.data.DataLoader:
        """
        Creates a Dataloader from the FastTextModelDataset.
        Use collate_fn() to tokenize and pad the sequences.

        Args:
            batch_size (int): Batch size.
            shuffle (bool, optional): Shuffle option. Defaults to False.
            drop_last (bool, optional): Drop last option. Defaults to False.
            num_workers (int, optional): Number of workers. Defaults to os.cpu_count() - 1.
            pin_memory (bool, optional): Set True if working on GPU, False if CPU. Defaults to True.
            persistent_workers (bool, optional): Set True for training, False for inference. Defaults to True.
            **kwargs: Additional arguments for PyTorch DataLoader.

        Returns:
            torch.utils.data.DataLoader: Dataloader.
        """

        logger.info(f"Creating DataLoader with {num_workers} workers.")

        # persistent_workers requires num_workers > 0
        if num_workers == 0:
            persistent_workers = False

        return torch.utils.data.DataLoader(
            dataset=self,
            batch_size=batch_size,
            collate_fn=self.collate_fn,
            shuffle=shuffle,
            drop_last=drop_last,
            pin_memory=pin_memory,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            **kwargs,
        )