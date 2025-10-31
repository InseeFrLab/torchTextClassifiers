from abc import ABC, abstractmethod
from typing import List, Union


class BaseTokenizer(ABC):
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    @abstractmethod
    def tokenize(self, text: Union[str, List[str]]) -> list:
        """Tokenizes the raw input text into a list of tokens."""
        pass

    def __len__(self):
        return self.vocab_size


class HuggingFaceTokenizer(BaseTokenizer, ABC):
    def __init__(self, vocab_size: int):
        super().__init__(vocab_size)

        self.trained = False
        self.tokenizer = None

    def tokenize(self, text: Union[str, List[str]]) -> list:
        if not self.trained:
            raise RuntimeError("Tokenizer must be trained before tokenization.")

        return self.tokenizer(
            text, padding=True, return_tensors="pt"
        )  # method from PreTrainedTokenizerFast

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
        """Applies post-training configurations to the tokenizer."""
        pass
