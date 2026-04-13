import numpy as np
from sklearn.preprocessing import LabelEncoder


class DictEncoder:
    def __init__(self, mapping: dict):
        self.mapping: dict[str, int] = mapping
        self.inverse_mapping: dict[int, str] = {v: k for k, v in mapping.items()}

    def __call__(self, value):
        return self.mapping.get(value, None)

    def transform(self, col):
        return self._dict_map(self.mapping, col)

    def inverse_transform(self, col):
        return self._dict_map(self.inverse_mapping, col)

    @staticmethod
    def _dict_map(dic, col):
        if isinstance(col, np.ndarray):
            return np.vectorize(dic.get)(col)
        elif isinstance(col, list):
            return [dic.get(v, None) for v in col]
        else:
            raise TypeError("Unsupported type for encoding: {}".format(type(col)))


class CategoricalValueEncoder:
    """
    An object to encode raw categorical values into numerical indices.

    Initialized with pre-built DictEncoder or sklearn LabelEncoder instances,
    one per categorical feature.

    Build encoders externally before passing them in:
    - DictEncoder: provide a ``{value: index}`` mapping directly.
    - sklearn LabelEncoder: call ``LabelEncoder().fit(column)`` per feature.

    Initialization:
    - encoders: A dictionary mapping feature names to DictEncoder or LabelEncoder instances.

    Properties:
    - vocabulary_sizes: List of vocabulary sizes (number of unique values) for each feature.

    Usage:
    - transform(array): Encode a 2D array of shape (N, n_features) to integers.
    - __call__(array): Alias for transform.
    """

    def __init__(self, encoders: dict[str, DictEncoder | LabelEncoder]):
        self.encoders = encoders

    @property
    def vocabulary_sizes(self) -> list[int]:
        """Number of unique categories per feature, in order."""
        sizes = []
        for enc in self.encoders.values():
            if isinstance(enc, DictEncoder):
                sizes.append(len(enc.mapping))
            elif hasattr(enc, "classes_"):
                sizes.append(len(enc.classes_))
            else:
                raise TypeError(f"Unsupported encoder type: {type(enc)}")
        return sizes

    def transform(self, X_categorical: np.ndarray) -> np.ndarray:
        """Encode all categorical columns to integer indices.

        Values are converted to strings before lookup. Unknown values raise a ValueError.

        Args:
            X_categorical: Array of shape (N, n_features) with categorical values.

        Returns:
            Integer-encoded array of shape (N, n_features), dtype int64.

        Raises:
            ValueError: If any value was not seen during fitting.
        """
        if X_categorical.ndim == 1:
            X_categorical = X_categorical.reshape(-1, 1)

        result = np.empty(X_categorical.shape, dtype=np.int64)
        for idx, (name, encoder) in enumerate(self.encoders.items()):
            col = X_categorical[:, idx].astype(str)
            encoded = encoder.transform(col)
            try:
                result[:, idx] = encoded.astype(np.int64)
            except (TypeError, ValueError):
                unknown = [v for v, e in zip(col.tolist(), encoded.tolist()) if e is None]
                raise ValueError(
                    f"Unknown values in categorical feature '{name}': {unknown}. "
                    "These values were not seen during fitting."
                )

        return result

    def __call__(self, array: np.ndarray) -> np.ndarray:
        return self.transform(array)
