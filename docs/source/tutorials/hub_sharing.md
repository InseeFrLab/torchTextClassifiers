# Sharing Models on the Hugging Face Hub

Save trained classifiers to disk, reload them, and publish them on the [Hugging Face Hub](https://huggingface.co/).

## Prerequisites

```bash
pip install "torchTextClassifiers[hub]"
```

The `hub` extra installs `huggingface_hub` and `safetensors`. Without it, the Hub methods raise an `ImportError` explaining how to install it; the rest of the library is unaffected.

:::{note}
Current limitations: only standard models built from a `ModelConfig` are supported (not custom models passed to `from_model`), and only the `NGramTokenizer` can be serialized. WordPiece and HuggingFace tokenizers are not yet supported.
:::

## Save and reload locally

```python
from torchTextClassifiers import torchTextClassifiers

classifier.save_pretrained("my_model/")

reloaded = torchTextClassifiers.from_pretrained("my_model/", device="cpu")
reloaded.predict(["some text"])
```

The directory contains:

| File | Content |
|------|---------|
| `model.safetensors` | Model weights |
| `config.json` | Model config, task type, tokenizer type, library version |
| `tokenizer/` | Serialized tokenizer |
| `labels.json` | Label mapping (when a value encoder is used) |
| `categorical.json` | Categorical variable mappings (if any) |

No pickle is used, so loading a model never executes arbitrary code.

## Push to the Hub

```python
classifier.push_to_hub(
    "my-org/my-model",
    model_card_kwargs={
        "license": "mit",
        "language": "fr",
        "datasets": ["my-org/my-dataset"],
        "metrics": {"accuracy": 0.91},
    },
)

classifier = torchTextClassifiers.from_pretrained("my-org/my-model")
```

Authenticate beforehand with `huggingface-cli login` or pass `token=...`.

A model card with the YAML metadata and a usage snippet is generated automatically. Supported `model_card_kwargs` keys: `license`, `language`, `datasets`, `metrics` (`{name: value}`, reported against `datasets[0]`), `dataset_revision`, `training_code_commit`, `model_name`, `tags`. Any other key is listed as a bullet in the card body.

## Version compatibility

The library version is stored in `config.json`. When loading:

- Any version difference emits a warning.
- A **major** version mismatch raises a `RuntimeError`; pass `strict=False` to `from_pretrained` to try loading anyway.

## Choosing the device

`from_pretrained(..., device="auto")` picks CUDA when available, else CPU. Pass `"cpu"` or `"cuda"` to override.
