import logging
import os
from typing import List, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from torchTextClassifiers.tokenizers import BaseTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = logging.getLogger(__name__)


class TextClassificationDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        categorical_variables: Union[List[List[int]], np.array, None],
        tokenizer: BaseTokenizer,
        labels: Union[List[int], List[List[int]], np.array, None] = None,
        ragged_multilabel: bool = False,
        sample_weights: Union[List[float], np.ndarray, None] = None,
    ):
        self.categorical_variables = categorical_variables

        self.sample_weights = (
            torch.tensor(sample_weights, dtype=torch.float32)
            if sample_weights is not None
            else None
        )

        self.texts = texts

        if hasattr(tokenizer, "trained") and not tokenizer.trained:
            raise RuntimeError(
                f"Tokenizer {type(tokenizer)} must be trained before creating dataset."
            )

        self.tokenizer = tokenizer

        self.texts = texts
        self.tokenizer = tokenizer
        self.labels = labels
        self.ragged_multilabel = ragged_multilabel

        if self.ragged_multilabel and self.labels is not None:
            max_value = int(max(max(row) for row in labels if row))
            self.num_classes = max_value + 1

            if max_value == 1:
                try:
                    labels = np.array(labels)
                    logger.critical(
                        """ragged_multilabel set to True but max label value is 1 and all samples have the same number of labels.
                        If your labels are already one-hot encoded, set ragged_multilabel to False. Otherwise computations are likely to be wrong."""
                    )
                except ValueError:
                    logger.warning(
                        "ragged_multilabel set to True but max label value is 1. If your labels are already one-hot encoded, set ragged_multilabel to False. Otherwise computations are likely to be wrong."
                    )
        elif not self.ragged_multilabel and self.labels is not None:
            self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        weight = self.sample_weights[idx] if self.sample_weights is not None else 1.0
        if self.labels is not None:
            return (
                str(self.texts[idx]),
                (
                    self.categorical_variables[idx]
                    if self.categorical_variables is not None
                    else None
                ),
                self.labels[idx],
                weight,
            )
        else:
            return (
                str(self.texts[idx]),
                (
                    self.categorical_variables[idx]
                    if self.categorical_variables is not None
                    else None
                ),
                None,
                weight,
            )

    def collate_fn(self, batch):
        text, *categorical_vars, labels, weights = zip(*batch)

        if self.labels is not None:
            if self.ragged_multilabel:
                # Pad labels to the max length in the batch
                labels_padded = torch.nn.utils.rnn.pad_sequence(
                    [torch.tensor(label) for label in labels],
                    batch_first=True,
                    padding_value=-1,  # use impossible class
                ).int()

                labels_tensor = torch.zeros(labels_padded.size(0), self.num_classes).float()
                mask = labels_padded != -1

                batch_size = labels_padded.size(0)
                rows = torch.arange(batch_size).unsqueeze(1).expand_as(labels_padded)[mask]
                cols = labels_padded[mask]

                labels_tensor[rows, cols] = 1

            else:
                labels_tensor = torch.stack(list(labels))
        else:
            labels_tensor = None

        tokenize_output = self.tokenizer.tokenize(list(text))

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
            categorical_tensors = None

        weights_tensor = torch.tensor([float(w) for w in weights], dtype=torch.float32)

        return {
            "input_ids": tokenize_output.input_ids,
            "attention_mask": tokenize_output.attention_mask,
            "categorical_vars": categorical_tensors,
            "labels": labels_tensor,
            "sample_weights": weights_tensor,
        }

    def create_dataloader(
        self,
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
        num_workers: int = os.cpu_count() - 1,
        pin_memory: bool = False,
        persistent_workers: bool = True,
        **kwargs,
    ):
        # persistent_workers requires num_workers > 0
        if num_workers == 0:
            persistent_workers = False

        return DataLoader(
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
