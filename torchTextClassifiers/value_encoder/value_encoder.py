from typing import Optional

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

    @property
    def vocabulary_size(self):
        return len(self.mapping)


class ValueEncoder:
    """
    An object to encode raw categorical values into numerical indices.

    Build encoders externally before passing them in:
    - DictEncoder: provide a ``{value: index}`` mapping directly.
    - sklearn LabelEncoder: call ``LabelEncoder().fit(column)`` per feature.

    Initialization:
    - label_encoder: A DictEncoder or LabelEncoder instance for encoding labels.
    - encoders (optional): A dictionary mapping feature names to DictEncoder or
      LabelEncoder instances.

    Properties:
    - vocabulary_sizes: List of vocabulary sizes (number of unique values) for each feature.
    - num_classes: Number of unique classes in the label encoder.

    Usage:
    - transform(array): Encode a 2D array of shape (N, n_features) to integers.
    - __call__(array): Alias for transform.
    """

    def __init__(
        self,
        label_encoder: DictEncoder | LabelEncoder,
        categorical_encoders: Optional[dict[str, DictEncoder | LabelEncoder]] = None,
    ):
        self.categorical_encoders = categorical_encoders

        if not isinstance(label_encoder, (DictEncoder, LabelEncoder)):
            raise TypeError(
                "label_encoder must be a DictEncoder or LabelEncoder instance, "
                f"got {type(label_encoder)}"
            )
        self.label_encoder = label_encoder

    @property
    def vocabulary_sizes(self) -> list[int]:
        """Number of unique categories per feature, in order."""

        if self.categorical_encoders is None:
            return None
        else:
            sizes = []
            for enc in self.categorical_encoders.values():
                if isinstance(enc, DictEncoder):
                    sizes.append(len(enc.mapping))
                elif hasattr(enc, "classes_"):
                    sizes.append(len(enc.classes_))
                else:
                    raise TypeError(f"Unsupported encoder type: {type(enc)}")
            return sizes

    @property
    def num_classes(self) -> int:
        """Number of unique classes in the label encoder, if provided."""
        if isinstance(self.label_encoder, DictEncoder):
            return len(self.label_encoder.mapping)
        elif hasattr(self.label_encoder, "classes_"):
            return len(self.label_encoder.classes_)
        else:
            raise TypeError(f"Unsupported label encoder type: {type(self.label_encoder)}")

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

        if self.categorical_encoders is None:
            raise ValueError("No categorical encoders provided. Cannot transform data.")

        if X_categorical.ndim == 1:
            X_categorical = X_categorical.reshape(-1, 1)

        result = np.empty(X_categorical.shape, dtype=np.int64)
        for idx, (name, encoder) in enumerate(self.categorical_encoders.items()):
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

    def transform_labels(self, y_labels: np.ndarray) -> np.ndarray:
        """Encode label array to integer indices.

        Values are converted to strings before lookup. Unknown values raise a ValueError.

        Args:
            y_labels: Array of shape (N,) with label values.
        Returns:
            Integer-encoded array of shape (N,), dtype int64.
        Raises:
            ValueError: If any label value was not seen during fitting.
        """

        col = y_labels.astype(str)
        encoded = self.label_encoder.transform(col)
        try:
            return encoded.astype(np.int64)
        except (TypeError, ValueError):
            unknown = [v for v, e in zip(col.tolist(), encoded.tolist()) if e is None]
            raise ValueError(
                f"Unknown values in label encoder: {unknown}. "
                "These values were not seen during fitting."
            )

    def inverse_transform_labels(self, y_encoded: np.ndarray) -> np.ndarray:
        """Decode integer-encoded labels back to original values.

        Args:
            y_encoded: Array of shape (N,) with integer-encoded labels.
        Returns:
            Array of shape (N,) with original label values.
        Raises:
            ValueError: If any encoded label value was not seen during fitting.
        """

        if isinstance(self.label_encoder, DictEncoder):
            inverse_mapping = self.label_encoder.inverse_mapping
            return np.vectorize(inverse_mapping.get)(y_encoded)
        elif hasattr(self.label_encoder, "inverse_transform"):
            shape = y_encoded.shape
            result = self.label_encoder.inverse_transform(y_encoded.ravel())
            return result.reshape(shape) if len(shape) > 1 else result
        else:
            raise TypeError(f"Unsupported label encoder type: {type(self.label_encoder)}")

    def __call__(self, array: np.ndarray) -> np.ndarray:
        return self.transform(array)
