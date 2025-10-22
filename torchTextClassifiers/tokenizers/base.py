from abc import ABC, abstractmethod


class BaseTokenizer(ABC):
    @abstractmethod
    def tokenize(self, text: str) -> list:
        """Tokenizes the raw input text into a list of tokens."""
        pass
