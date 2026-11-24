"""Hugging Face Hub integration helpers (optional ``hub`` extra).

``huggingface_hub`` and ``safetensors`` are optional dependencies
(``pip install torchTextClassifiers[hub]``). When ``huggingface_hub`` is missing, the
base class below is a stub whose Hub methods raise a helpful ``ImportError``, so the
core library keeps working without the extra.
"""

import json
import warnings
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from huggingface_hub import ModelCard, ModelCardData, ModelHubMixin

    HAS_HUB = True
except ImportError:  # pragma: no cover - exercised only without the extra
    HAS_HUB = False

LIBRARY_NAME = "torchtextclassifiers"
REPO_URL = "https://github.com/InseeFrLab/torchTextClassifiers"
DOCS_URL = "https://inseefrlab.github.io/torchTextClassifiers/"

CONFIG_FILE = "config.json"
WEIGHTS_FILE = "model.safetensors"
LABELS_FILE = "labels.json"
CATEGORICAL_FILE = "categorical.json"
TOKENIZER_DIR = "tokenizer"
FORMAT_VERSION = 1

_MISSING_MSG = (
    "Hugging Face Hub support requires the 'hub' extra: pip install 'torchTextClassifiers[hub]'"
)

if HAS_HUB:
    HubBase = ModelHubMixin
else:

    class HubBase:  # type: ignore[no-redef]
        """Stub used when ``huggingface_hub`` is not installed."""

        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__()

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise ImportError(_MISSING_MSG)

        def push_to_hub(self, *args, **kwargs):
            raise ImportError(_MISSING_MSG)

        def save_pretrained(self, *args, **kwargs):
            raise ImportError(_MISSING_MSG)


def require_hub() -> None:
    if not HAS_HUB:
        raise ImportError(_MISSING_MSG)
    try:
        import safetensors  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError(_MISSING_MSG) from e


def library_version() -> str:
    try:
        return metadata.version("torchtextclassifiers")
    except metadata.PackageNotFoundError:
        from torchTextClassifiers import __version__

        return __version__


def _major(version: str) -> Optional[int]:
    try:
        return int(version.split(".")[0])
    except (ValueError, IndexError):
        return None


def check_version(saved: Optional[str], strict: bool = True) -> None:
    """Warn on any version difference; raise on a major mismatch when ``strict``."""
    current = library_version()
    if saved is None or saved == current:
        return
    msg = f"Model was saved with torchTextClassifiers {saved}, but version {current} is installed."
    if strict and _major(saved) is not None and _major(saved) != _major(current):
        raise RuntimeError(msg + " Pass strict=False to try loading anyway.")
    warnings.warn(msg, stacklevel=3)


# ----------------------------------------------------------------------
# Tokenizer (pickle-free) serialization
# ----------------------------------------------------------------------


def save_tokenizer(tokenizer, directory: Path) -> str:
    """Save ``tokenizer`` in ``directory`` and return its registered type name."""
    from torchTextClassifiers.tokenizers import NGramTokenizer

    if isinstance(tokenizer, NGramTokenizer):
        tokenizer.save_pretrained(str(directory))
        return "NGramTokenizer"
    raise NotImplementedError(
        f"Hub serialization is not supported yet for tokenizer {type(tokenizer).__name__}. "
        "Currently supported: NGramTokenizer."
    )


def load_tokenizer(tokenizer_type: str, directory: Path):
    from torchTextClassifiers.tokenizers import NGramTokenizer

    if tokenizer_type == "NGramTokenizer":
        return NGramTokenizer.load(str(directory / "tokenizer.json"))
    raise ValueError(f"Unknown tokenizer type in config.json: {tokenizer_type!r}")


# ----------------------------------------------------------------------
# Model card
# ----------------------------------------------------------------------

_CARD_BODY = """
# {model_name}

Text classifier trained with [torchTextClassifiers]({repo_url}).

## Usage

```bash
pip install "torchTextClassifiers[hub]"=={version}
```

```python
from torchTextClassifiers import torchTextClassifiers

clf = torchTextClassifiers.from_pretrained("{repo_id}")
clf.predict(["some text"])
```

## Model details

- Task: {task}
- Number of classes: {num_classes}
- Library version: {version}
{extra}
"""


def build_model_card(
    clf,
    *,
    license: Optional[str] = None,
    language: Optional[Any] = None,
    datasets: Optional[list] = None,
    metrics: Optional[Dict[str, float]] = None,
    model_name: Optional[str] = None,
    repo_id: Optional[str] = None,
    dataset_revision: Optional[str] = None,
    training_code_commit: Optional[str] = None,
    tags: Optional[list] = None,
    **extra_fields,
):
    """Build the model card (YAML header + markdown body) for a classifier."""
    require_hub()
    model_name = model_name or (repo_id.split("/")[-1] if repo_id else "torchtextclassifiers-model")
    all_tags = ["text-classification", LIBRARY_NAME] + list(tags or [])

    eval_results = None
    if metrics and datasets:
        from huggingface_hub import EvalResult

        eval_results = [
            EvalResult(
                task_type="text-classification",
                dataset_type=datasets[0],
                dataset_name=datasets[0],
                dataset_revision=dataset_revision,
                metric_type=name,
                metric_value=value,
            )
            for name, value in metrics.items()
        ]

    card_data = ModelCardData(
        library_name=LIBRARY_NAME,
        pipeline_tag="text-classification",
        tags=all_tags,
        license=license,
        language=language,
        datasets=datasets,
        eval_results=eval_results,
        model_name=model_name,
    )

    extra_lines = []
    if training_code_commit:
        extra_lines.append(f"- Training code commit: `{training_code_commit}`")
    if dataset_revision and datasets:
        extra_lines.append(f"- Dataset revision: `{dataset_revision}`")
    if metrics and not datasets:
        extra_lines += [f"- {k}: {v}" for k, v in metrics.items()]
    extra_lines += [f"- {k}: {v}" for k, v in extra_fields.items()]

    body = _CARD_BODY.format(
        model_name=model_name,
        repo_url=REPO_URL,
        repo_id=repo_id or "<namespace>/<model>",
        version=library_version(),
        task="multilabel" if clf.ragged_multilabel else "multiclass",
        num_classes=clf.num_classes,
        extra="\n".join(extra_lines),
    )
    return ModelCard(f"---\n{card_data.to_yaml()}\n---\n{body}")


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())
