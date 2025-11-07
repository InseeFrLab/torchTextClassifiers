from abc import ABC, abstractmethod
from typing import List, Optional, Union

try:
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast

    HAS_HF = True
except ImportError:
    HAS_HF = False


class BaseTokenizer(ABC):
    def __init__(
        self, vocab_size: int, output_vectorized: bool = False, output_dim: Optional[int] = None
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
        if self.output_vectorized:
            if output_dim is None:
                raise ValueError(
                    "Tokenizer's output_dim must be provided if output_vectorized is True."
                )

    @abstractmethod
    def tokenize(self, text: Union[str, List[str]]) -> list:
        """Tokenizes the raw input text into a list of tokens."""
        pass

    def __len__(self):
        return self.vocab_size


class HuggingFaceTokenizer(BaseTokenizer, ABC):
    def __init__(self, vocab_size: int, output_dim: Optional[int] = None):
        super().__init__(
            vocab_size, output_vectorized=False, output_dim=output_dim
        )  # it outputs token ids and not vectors

        self.trained = False
        self.tokenizer = None

    def tokenize(self, text: Union[str, List[str]]) -> list:
        if not self.trained:
            raise RuntimeError("Tokenizer must be trained before tokenization.")

        # Pad to longest sequence if no output_dim is specified
        padding = True if self.output_dim is None else "max_length"

        return self.tokenizer(
            text,
            padding=padding,
            return_tensors="pt",
            max_length=self.output_dim,
        )  # method from PreTrainedTokenizerFast

    @classmethod
    def load(cls, load_path: str):
        loaded_tokenizer = PreTrainedTokenizerFast(tokenizer_file=load_path)
        instance = cls(vocab_size=len(loaded_tokenizer), trained=True)
        instance.tokenizer = loaded_tokenizer
        instance._post_training()
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

    @abstractmethod
    def train(
        self,
        training_corpus: list,
        save_path: str = None,
        filesystem=None,
        s3_save_path=None,
        **kwargs,
    ):
        """Trains the tokenizer on the provided training corpus."""
        pass

    @abstractmethod
    def _post_training(self):

    def __repr__(self):
        return self.tokenizer.__repr__()
