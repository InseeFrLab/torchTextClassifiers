from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
from ..base import BaseClassifierConfig
from typing import Optional, Union, Type, List, Dict, Any

@dataclass
class FastTextConfig(BaseClassifierConfig):
    """Configuration for FastText classifier."""
    # Embedding matrix
    embedding_dim: int
    sparse: bool

    # Tokenizer-related
    num_tokens: int
    min_count: int
    min_n: int
    max_n: int
    len_word_ngrams: int

    # Optional parameters
    num_classes: Optional[int] = None
    num_rows: Optional[int] = None

    # Categorical variables
    categorical_vocabulary_sizes: Optional[List[int]] = None
    categorical_embedding_dims: Optional[Union[List[int], int]] = None
    num_categorical_features: Optional[int] = None

    # Model-specific parameters
    direct_bagging: Optional[bool] = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FastTextConfig":
        return cls(**data)