import json

import numpy as np
import pytest

pytest.importorskip("huggingface_hub")
pytest.importorskip("safetensors")

from torchTextClassifiers import ModelConfig, torchTextClassifiers
from torchTextClassifiers.model.components import AttentionConfig
from torchTextClassifiers.tokenizers import NGramTokenizer
from torchTextClassifiers.value_encoder import DictEncoder, ValueEncoder

TEXTS = [
    "this is a positive example",
    "this is a negative example",
    "another positive case",
    "another negative case",
    "good example here",
    "bad example here",
] * 3


def _tokenizer():
    return NGramTokenizer(
        min_count=1,
        min_n=2,
        max_n=4,
        num_tokens=50,
        len_word_ngrams=1,
        training_text=TEXTS,
    )


def _encoder(with_categorical):
    labels = DictEncoder({"neg": 0, "pos": 1})
    cats = {"c0": DictEncoder({"a": 0, "b": 1, "c": 2})} if with_categorical else None
    return ValueEncoder(labels, cats)


@pytest.mark.parametrize("with_categorical", [False, True])
def test_round_trip(tmp_path, with_categorical):
    import torch

    torch.manual_seed(0)
    enc = _encoder(with_categorical)
    cfg = ModelConfig(
        embedding_dim=8,
        categorical_vocabulary_sizes=enc.vocabulary_sizes,
        categorical_embedding_dims=4 if with_categorical else None,
        num_classes=2,
    )
    clf = torchTextClassifiers(_tokenizer(), cfg, value_encoder=enc)

    X = np.array(TEXTS[:6]).reshape(-1, 1)
    if with_categorical:
        X = np.column_stack([X, np.array(["a", "b", "c", "a", "b", "c"])])
    before = clf.predict(X, device="cpu")

    clf.save_pretrained(tmp_path)
    for name in ["model.safetensors", "config.json", "labels.json", "README.md"]:
        assert (tmp_path / name).exists(), name
    assert (tmp_path / "tokenizer").is_dir()
    assert (tmp_path / "categorical.json").exists() == with_categorical
    assert not list(tmp_path.rglob("*.pkl"))
    assert json.loads((tmp_path / "config.json").read_text())["task"] == "multiclass"

    loaded = torchTextClassifiers.from_pretrained(str(tmp_path), device="cpu")
    after = loaded.predict(X, device="cpu")
    np.testing.assert_array_equal(before["prediction"], after["prediction"])
    np.testing.assert_allclose(before["confidence"], after["confidence"], rtol=1e-5)


def test_model_card_and_kwargs(tmp_path):
    cfg = ModelConfig(embedding_dim=8, num_classes=2)
    clf = torchTextClassifiers(_tokenizer(), cfg)
    card = clf.generate_model_card(
        license="mit",
        language="fr",
        datasets=["org/data"],
        metrics={"accuracy": 0.9},
        training_code_commit="abc123",
    )
    text = str(card)
    assert "library_name: torchtextclassifiers" in text
    assert "pipeline_tag: text-classification" in text
    assert "license: mit" in text
    assert "abc123" in text
    assert "model-index" in text


def test_custom_model_rejected(tmp_path):
    cfg = ModelConfig(embedding_dim=8, num_classes=2)
    clf = torchTextClassifiers(_tokenizer(), cfg)
    clf._custom_model = True
    with pytest.raises(NotImplementedError):
        clf.save_pretrained(tmp_path)


def test_major_version_mismatch(tmp_path):
    clf = torchTextClassifiers(_tokenizer(), ModelConfig(embedding_dim=8, num_classes=2))
    clf.save_pretrained(tmp_path)
    cfg_path = tmp_path / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["library_version"] = "99.0.0"
    cfg_path.write_text(json.dumps(cfg))
    with pytest.raises(RuntimeError):
        torchTextClassifiers.from_pretrained(str(tmp_path))
    with pytest.warns(UserWarning):
        torchTextClassifiers.from_pretrained(str(tmp_path), strict=False, device="cpu")


def test_model_config_attention_round_trip():
    cfg = ModelConfig(
        embedding_dim=8,
        num_classes=2,
        attention_config=AttentionConfig(n_layers=1, n_head=2, n_kv_head=2),
    )
    assert ModelConfig.from_dict(cfg.to_dict()) == cfg
