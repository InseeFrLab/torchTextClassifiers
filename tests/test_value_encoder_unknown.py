import numpy as np
import pytest
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from torchTextClassifiers.value_encoder.value_encoder import ValueEncoder

X = np.array([["a", "x"], ["b", "y"], ["a", "y"]])


def _make(handle_unknown):
    cats = {"c1": OrdinalEncoder().fit(X[:, [0]]), "c2": LabelEncoder().fit(X[:, 1])}
    return ValueEncoder(LabelEncoder().fit(["p", "q"]), cats, handle_unknown=handle_unknown)


def test_handle_unknown_maps_to_extra_index():
    enc = _make(True)
    assert enc.vocabulary_sizes == [3, 3]
    out = enc.transform(np.array([["a", "z"], ["new", "y"], [np.nan, "x"]], dtype=object))
    assert out.tolist() == [[0, 2], [2, 1], [2, 0]]


def test_unknown_raises_by_default():
    enc = _make(False)
    assert enc.vocabulary_sizes == [2, 2]
    with pytest.raises(ValueError):
        enc.transform(np.array([["new", "x"]], dtype=object))
