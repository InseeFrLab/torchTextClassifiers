from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

try:
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    HAS_HF = True
except ImportError:
    HAS_HF = False


@dataclass
class TokenizerOutput:
    input_ids: torch.Tensor  # shape: (batch_size, seq_len)
    attention_mask: torch.Tensor  # shape: (batch_size, seq_len)
    offset_mapping: Optional[torch.Tensor] = None  # shape: (batch_size, seq_len, 2)
    word_ids: Optional[np.ndarray] = None  # shape: (batch_size, seq_len)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TokenizerOutput":
        return cls(**data)

    def __post_init__(self):
        # --- Basic type checks ---
        if not isinstance(self.input_ids, torch.Tensor):
            raise TypeError(f"token_ids must be a torch.Tensor, got {type(self.input_ids)}")
        if not isinstance(self.attention_mask, torch.Tensor):
            raise TypeError(
                f"attention_mask must be a torch.Tensor, got {type(self.attention_mask)}"
            )
        if self.offset_mapping is not None and not isinstance(self.offset_mapping, torch.Tensor):
            raise TypeError(
                f"offset_mapping must be a torch.Tensor or None, got {type(self.offset_mapping)}"
            )
        if self.word_ids is not None and not isinstance(self.word_ids, np.ndarray):
            raise TypeError(f"word_ids must be a numpy.ndarray or None, got {type(self.word_ids)}")

        # --- Shape consistency checks ---
        if self.input_ids.shape != self.attention_mask.shape:
            raise ValueError(
                f"Shape mismatch: token_ids {self.token_ids.shape} and attention_mask {self.attention_mask.shape}"
            )

        if self.offset_mapping is not None:
            expected_shape = (*self.input_ids.shape, 2)
            if self.offset_mapping.shape != expected_shape:
                raise ValueError(
                    f"offset_mapping should have shape {expected_shape}, got {self.offset_mapping.shape}"
                )

        if self.word_ids is not None:
            if self.word_ids.shape != self.input_ids.shape:
                raise ValueError(
                    f"word_ids should have shape {self.input_ids.shape}, got {self.word_ids.shape}"
                )


class BaseTokenizer(ABC):
    def __init__(
        self,
        vocab_size: int,
        padding_idx: int,
        output_vectorized: bool = False,
        output_dim: Optional[int] = None,
    ):
        """
        Base class for tokenizers.
        Args:
            vocab_size (int): Size of the vocabulary.
            output_vectorized (bool): Whether the tokenizer outputs vectorized tokens.
                True for instance for a TF-IDF tokenizer.
        """

        self.vocab_size = vocab_size
        self.output_vectorized = output_vectorized
        self.output_dim = output_dim
        self.padding_idx = padding_idx
        if self.output_vectorized:
            if output_dim is None:
                raise ValueError(
                    "Tokenizer's output_dim must be provided if output_vectorized is True."
                )

    @abstractmethod
    def tokenize(self, text: Union[str, List[str]]) -> TokenizerOutput:
        """Tokenizes the raw input text into a list of tokens."""
        pass

    def __len__(self):
        return self.vocab_size

    def __repr__(self):
        return f"{self.__class__.__name__}(vocab_size={self.vocab_size}, output_vectorized={self.output_vectorized}, output_dim={self.output_dim})"

    def __call__(self, text: Union[str, List[str]], **kwargs) -> list:
        return self.tokenize(text, **kwargs)


class HuggingFaceTokenizer(BaseTokenizer):
    def __init__(
        self,
        vocab_size: int,
        output_dim: Optional[int] = None,
        padding_idx: Optional[int] = None,
        trained: bool = False,
    ):
        super().__init__(
            vocab_size, output_vectorized=False, output_dim=output_dim, padding_idx=padding_idx
        )  # it outputs token ids and not vectors

        self.trained = trained
        self.tokenizer = None
        self.padding_idx = padding_idx
        self.output_dim = output_dim  # constant context size for all batch

    def tokenize(
        self,
        text: Union[str, List[str]],
        return_offsets_mapping: Optional[bool] = False,
        return_word_ids: Optional[bool] = False,
    ) -> list:
        if not self.trained:
            raise RuntimeError("Tokenizer must be trained before tokenization.")

        # Pad to longest sequence if no output_dim is specified
        padding = True if self.output_dim is None else "max_length"
        truncation = True if self.output_dim is not None else False

        tokenize_output = self.tokenizer(
            text,
            padding=padding,
            return_tensors="pt",
            truncation=truncation,
            max_length=self.output_dim,
            return_offsets_mapping=return_offsets_mapping,
        )  # method from PreTrainedTokenizerFast

        encoded_text = tokenize_output["input_ids"]

        if return_word_ids:
            word_ids = np.array([tokenize_output.word_ids(i) for i in range(len(encoded_text))])
        else:
            word_ids = None

        return TokenizerOutput(
            input_ids=encoded_text,
            attention_mask=tokenize_output["attention_mask"],
            offset_mapping=tokenize_output.get("offset_mapping", None),
            word_ids=word_ids,
        )

    @classmethod
    def load_from_pretrained(cls, tokenizer_name: str, output_dim: Optional[int] = None):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        padding_idx = tokenizer.pad_token_id
        instance = cls(
            vocab_size=len(tokenizer), trained=True, padding_idx=padding_idx, output_dim=output_dim
        )
        instance.tokenizer = tokenizer
        return instance

    @classmethod
    def load(cls, load_path: str):
        loaded_tokenizer = PreTrainedTokenizerFast(tokenizer_file=load_path)
        instance = cls(vocab_size=len(loaded_tokenizer), trained=True)
        instance.tokenizer = loaded_tokenizer
        # instance._post_training()
        return instance

    @classmethod
    def load_from_s3(cls, s3_path: str, filesystem):
        if filesystem.exists(s3_path) is False:
            raise FileNotFoundError(
                f"Tokenizer not found at {s3_path}. Please train it first (see src/train_tokenizers)."
            )

        with filesystem.open(s3_path, "rb") as f:
            json_str = f.read().decode("utf-8")

        tokenizer_obj = Tokenizer.from_str(json_str)
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_obj)
        instance = cls(vocab_size=len(tokenizer), trained=True)
        instance.tokenizer = tokenizer
        instance._post_training()
        return instance

    def train(self, *args, **kwargs):
        raise NotImplementedError(
            "This tokenizer cannot be trained directly. "
            "Load it from pretrained or implement train() in a subclass."
        )

    def _post_training(self):
        raise NotImplementedError("_post_training() not implemented for HuggingFaceTokenizer.")

    def __repr__(self):
        return f"{self.__class__.__name__} \n HuggingFace tokenizer: {self.tokenizer.__repr__()}"
