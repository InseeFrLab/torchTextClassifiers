from typing import Optional

import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder


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
    - sklearn OrdinalEncoder: call ``OrdinalEncoder().fit(column.reshape(-1, 1))`` per feature.
      (Only for categorical features, not labels.)

    Initialization:
    - label_encoder: A DictEncoder or LabelEncoder instance for encoding labels.
    - categorical_encoders (optional): A dictionary mapping feature names to DictEncoder,
      LabelEncoder or OrdinalEncoder instances.
    - handle_unknown (default False): If True, categorical values not seen during fitting
      (or missing values) are mapped to an extra "unknown" index, equal to the number of known
      categories of the feature, instead of raising a ValueError. ``vocabulary_sizes`` then
      includes this extra slot so the embedding layer has a row for it.

    Properties:
    - vocabulary_sizes: List of vocabulary sizes (number of unique values) for each feature.
    - num_classes: Number of unique classes in the label encoder.

    Usage:
    - transform(array): Encode a 2D array of shape (N, n_features) to integers.
    - __call__(array): Alias for transform.
    """

    def __init__(
        self,
        label_encoder: DictEncoder | LabelEncoder | list[DictEncoder | LabelEncoder],
        categorical_encoders: Optional[
            dict[str, DictEncoder | LabelEncoder | OrdinalEncoder]
        ] = None,
        handle_unknown: bool = False,
    ):
        self.categorical_encoders = categorical_encoders
        self.handle_unknown = handle_unknown
        if categorical_encoders is not None:
            for name, enc in categorical_encoders.items():
                if not isinstance(enc, (DictEncoder, LabelEncoder, OrdinalEncoder)):
                    raise TypeError(
                        f"Encoder for '{name}' must be a DictEncoder, LabelEncoder or "
                        f"OrdinalEncoder, got {type(enc)}"
                    )

        if isinstance(label_encoder, list):
            for enc in label_encoder:
                if not isinstance(enc, (DictEncoder, LabelEncoder)):
                    raise TypeError(
                        "Each element of label_encoder list must be a DictEncoder or LabelEncoder, "
                        f"got {type(enc)}"
                    )
        elif not isinstance(label_encoder, (DictEncoder, LabelEncoder)):
            raise TypeError(
                "label_encoder must be a DictEncoder, LabelEncoder, or list thereof, "
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
                sizes.append(len(self._known_categories(enc)) + int(self.handle_unknown))
            return sizes

    @staticmethod
    def _known_categories(enc) -> np.ndarray:
        if isinstance(enc, DictEncoder):
            return np.array(list(enc.mapping.keys()))
        elif isinstance(enc, OrdinalEncoder):
            return np.asarray(enc.categories_[0])
        elif hasattr(enc, "classes_"):
            return np.asarray(enc.classes_)
        raise TypeError(f"Unsupported encoder type: {type(enc)}")

    @staticmethod
    def _encode_known(enc, col: np.ndarray) -> np.ndarray:
        """Encode a column that only contains known values."""
        if isinstance(enc, OrdinalEncoder):
            return enc.transform(col.reshape(-1, 1))[:, 0]
        return np.asarray(enc.transform(col))

    @property
    def num_classes(self) -> int:
        """Number of unique classes in the label encoder, if provided."""

        def _get_num_classes(enc):
            if isinstance(enc, DictEncoder):
                return len(enc.mapping)
            elif hasattr(enc, "classes_"):
                return len(enc.classes_)
            else:
                raise TypeError(f"Unsupported encoder type: {type(enc)}")

        if isinstance(self.label_encoder, DictEncoder) or isinstance(
            self.label_encoder, LabelEncoder
        ):
            return _get_num_classes(self.label_encoder)
        elif isinstance(self.label_encoder, list):
            return [_get_num_classes(enc) for enc in self.label_encoder]
        else:
            raise TypeError(f"Unsupported label encoder type: {type(self.label_encoder)}")

    def transform(self, X_categorical: np.ndarray) -> np.ndarray:
        """Encode all categorical columns to integer indices.

        Values are converted to strings before lookup. Unknown values raise a ValueError,
        unless ``handle_unknown=True``, in which case they map to index ``vocabulary_size``.

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
            if self.handle_unknown:
                known = self._known_categories(encoder)
                is_known = np.isin(col, known.astype(str))
                unknown_index = len(known)
                column = np.full(col.shape, unknown_index, dtype=np.int64)
                if is_known.any():
                    column[is_known] = self._encode_known(encoder, col[is_known]).astype(np.int64)
                result[:, idx] = column
                continue
            if isinstance(encoder, OrdinalEncoder):
                known = self._known_categories(encoder).astype(str)
                unknown = sorted(set(col[~np.isin(col, known)].tolist()))
                if unknown:
                    raise ValueError(
                        f"Unknown values in categorical feature '{name}': {unknown}. "
                        "These values were not seen during fitting."
                    )
            encoded = self._encode_known(encoder, col)
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
            y_labels: Array of shape (N,) for a single encoder, or (N, n_levels) for a list
                of encoders.
        Returns:
            Integer-encoded array of shape (N,) or (N, n_levels), dtype int64.
        Raises:
            ValueError: If any label value was not seen during fitting.
        """

        def _encode_col(enc, col, level_name="label encoder"):
            encoded = enc.transform(col)
            try:
                return encoded.astype(np.int64)
            except (TypeError, ValueError):
                unknown = [v for v, e in zip(col.tolist(), encoded.tolist()) if e is None]
                raise ValueError(
                    f"Unknown values in {level_name}: {unknown}. "
                    "These values were not seen during fitting."
                )

        if isinstance(self.label_encoder, list):
            if y_labels.ndim == 1:
                y_labels = y_labels.reshape(-1, 1)
            result = np.empty(y_labels.shape, dtype=np.int64)
            for idx, enc in enumerate(self.label_encoder):
                result[:, idx] = _encode_col(
                    enc, y_labels[:, idx].astype(str), f"label encoder at level {idx}"
                )
            return result

        return _encode_col(self.label_encoder, y_labels.astype(str))

    def inverse_transform_labels(self, y_encoded: np.ndarray) -> np.ndarray:
        """Decode integer-encoded labels back to original values.

        Args:
            y_encoded: Supported shapes:
                - Single encoder: (N,) or (N, top_k)
                - List of encoders: (N, n_levels), or (N, top_k, n_levels)
        Returns:
            Decoded array with the same shape as ``y_encoded``.
        Raises:
            ValueError: If any encoded label value was not seen during fitting.
        """

        def _decode_col(enc, col: np.ndarray) -> np.ndarray:
            """Decode a flat 1-D array using a single encoder."""
            if isinstance(enc, DictEncoder):
                return np.vectorize(enc.inverse_mapping.get)(col)
            elif hasattr(enc, "inverse_transform"):
                return enc.inverse_transform(col)
            else:
                raise TypeError(f"Unsupported label encoder type: {type(enc)}")

        if isinstance(self.label_encoder, list):
            if y_encoded.ndim == 1:
                y_encoded = y_encoded.reshape(-1, 1)

            if y_encoded.ndim == 2:
                # (N, n_levels)
                result = np.empty(y_encoded.shape, dtype=object)
                for idx, enc in enumerate(self.label_encoder):
                    result[:, idx] = _decode_col(enc, y_encoded[:, idx])
                return result

            if y_encoded.ndim == 3:
                # (N, top_k, n_levels)
                n, top_k = y_encoded.shape[0], y_encoded.shape[1]
                result = np.empty(y_encoded.shape, dtype=object)
                for idx, enc in enumerate(self.label_encoder):
                    flat = y_encoded[:, :, idx].ravel()
                    result[:, :, idx] = _decode_col(enc, flat).reshape(n, top_k)
                return result

            raise ValueError(
                f"Expected 1-D, 2-D, or 3-D array for list encoder, got {y_encoded.ndim}-D"
            )

        if isinstance(self.label_encoder, DictEncoder):
            return np.vectorize(self.label_encoder.inverse_mapping.get)(y_encoded)
        elif hasattr(self.label_encoder, "inverse_transform"):
            shape = y_encoded.shape
            result = self.label_encoder.inverse_transform(y_encoded.ravel())
            return result.reshape(shape) if len(shape) > 1 else result
        else:
            raise TypeError(f"Unsupported label encoder type: {type(self.label_encoder)}")

    def __call__(self, array: np.ndarray) -> np.ndarray:
        return self.transform(array)
