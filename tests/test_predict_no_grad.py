import numpy as np
import pytest
import torch

from torchTextClassifiers import ModelConfig, torchTextClassifiers
from torchTextClassifiers.tokenizers import NGramTokenizer


def _build_ttc(texts):
    tokenizer = NGramTokenizer(
        min_count=1, min_n=2, max_n=4, num_tokens=50, len_word_ngrams=2, output_dim=20
    )
    tokenizer.train(list(texts))
    model_config = ModelConfig(embedding_dim=8, num_classes=2)
    return torchTextClassifiers(tokenizer=tokenizer, model_config=model_config)


def test_predict_does_not_track_gradients(sample_text_data):
    """predict() must run its forward pass without building an autograd graph (issue #111)."""
    ttc = _build_ttc(sample_text_data)

    recorded = {}

    def hook(module, inputs, output):
        recorded["grad_enabled"] = torch.is_grad_enabled()
        recorded["requires_grad"] = output.requires_grad

    handle = ttc.pytorch_model.register_forward_hook(hook)
    try:
        assert torch.is_grad_enabled()
        out = ttc.predict(np.asarray(sample_text_data), raw_categorical_inputs=False)
    finally:
        handle.remove()

    assert recorded["grad_enabled"] is False
    assert recorded["requires_grad"] is False
    assert out["prediction"].shape[0] == len(sample_text_data)
    # Global grad mode is restored after predict()
    assert torch.is_grad_enabled()


def test_predict_captum_works_under_user_no_grad(sample_text_data):
    """Captum attributions still work when predict() is called inside torch.no_grad()."""
    pytest.importorskip("captum")
    torch.manual_seed(0)
    ttc = _build_ttc(sample_text_data)
    # The classification head is zero-initialized, which would give all-zero
    # attributions on an untrained model: randomize it so the check is meaningful.
    with torch.no_grad():
        ttc.pytorch_model.classification_head.net.weight.normal_()
    X = np.asarray(sample_text_data)

    ref = ttc.predict(X, raw_categorical_inputs=False, explain_with_captum=True)
    with torch.no_grad():
        out = ttc.predict(X, raw_categorical_inputs=False, explain_with_captum=True)
        assert not torch.is_grad_enabled()  # caller's grad mode untouched

    attributions = out["captum_attributions"]
    assert attributions.shape[:2] == (len(sample_text_data), 1)
    assert torch.isfinite(attributions).all()
    assert attributions.abs().sum() > 0
    torch.testing.assert_close(attributions, ref["captum_attributions"])
