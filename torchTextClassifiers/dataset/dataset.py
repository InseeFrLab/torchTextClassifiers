import os
from typing import List, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from torchTextClassifiers.tokenizers import BaseTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class TextClassificationDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        categorical_variables: Union[List[List[int]], np.array, None],
        tokenizer: BaseTokenizer,
        labels: Union[List[int], None] = None,
    ):
        self.categorical_variables = categorical_variables

        self.texts = texts

        if hasattr(tokenizer, "trained") and not tokenizer.trained:
            raise RuntimeError(
                f"Tokenizer {type(tokenizer)} must be trained before creating dataset."
            )

        self.tokenizer = tokenizer

        self.texts = texts
        self.tokenizer = tokenizer
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        if self.labels is not None:
            return (
                self.texts[idx],
                (
                    self.categorical_variables[idx]
                    if self.categorical_variables is not None
                    else None
                ),
                self.labels[idx],
            )
        else:
            return (
                self.texts[idx],
                (
                    self.categorical_variables[idx]
                    if self.categorical_variables is not None
                    else None
                ),
                None,
            )

    def collate_fn(self, batch):
        text, *categorical_vars, y = zip(*batch)

        if self.labels is not None:
            labels_tensor = torch.tensor(y, dtype=torch.long)
        else:
            labels_tensor = None

        tokenize_output = self.tokenizer.tokenize(text)

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

        return {
            "input_ids": tokenize_output.input_ids,
            "attention_mask": tokenize_output.attention_mask,
            "categorical_vars": categorical_tensors,
            "labels": labels_tensor,
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
