# Custom Architectures with from_model

**Difficulty:** Advanced | **Time:** 30 minutes

## When to use this

The standard `torchTextClassifiers` constructor + `ModelConfig` covers most
single-task classification needs.  Use `from_model` when you need something the
standard architecture cannot express:

- **Multiple classification heads** (multi-task / hierarchical labels)
- **Shared encoders** across several outputs
- **Custom combination logic** between text and categorical embeddings
- **Any other topology** that does not fit a single linear pipeline

---

## Required interface

Your custom model must satisfy three contracts so that the wrapper's `predict`,
`save`, and `load` methods work correctly.

### 1. `forward` signature

```python
def forward(
    self,
    input_ids: torch.Tensor,        # (batch, seq_len) — Long
    attention_mask: torch.Tensor,   # (batch, seq_len) — int
    categorical_vars: torch.Tensor, # (batch, n_cats)  — Long, may be None
    **kwargs,                       # ignored by the wrapper
) -> torch.Tensor | list[torch.Tensor]:
    ...
```

- The argument **names must match exactly** — the wrapper calls the model with
  keyword arguments from the dataloader collate function.
- The return value must be **raw logits** (not softmaxed).
  - Single task → `torch.Tensor` of shape `(batch, num_classes)`
  - Multi-task → `list[torch.Tensor]`, one tensor per task

### 2. `num_classes` attribute

```python
model.num_classes  # int  (single task)
model.num_classes  # list[int]  (multi-task — one entry per task head)
```

### 3. `categorical_variable_net` attribute

```python
model.categorical_variable_net  # CategoricalVariableNet | None
```

Set this to `None` if your model does not use categorical features.  When it is
not `None` the wrapper reads
`categorical_variable_net.categorical_vocabulary_sizes` to configure data
encoding.

---

## Minimal example — single-task custom model

```python
import torch
import torch.nn as nn
from torchTextClassifiers import torchTextClassifiers
from torchTextClassifiers.model.components import TokenEmbedder, TokenEmbedderConfig
from torchTextClassifiers.tokenizers import WordPieceTokenizer

class MyClassifier(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int):
        super().__init__()
        self.token_embedder = TokenEmbedder(TokenEmbedderConfig(
            vocab_size=vocab_size, embedding_dim=64, padding_idx=0,
        ))
        self.pool = lambda x, mask: (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        self.head = nn.Linear(64, num_classes)

        # Required attributes
        self.num_classes = num_classes
        self.categorical_variable_net = None  # no categorical features

    def forward(self, input_ids, attention_mask, categorical_vars=None, **kwargs):
        out = self.token_embedder(input_ids, attention_mask)
        sentence = self.pool(out["token_embeddings"], attention_mask.float())
        return self.head(sentence)   # (batch, num_classes) — raw logits

tokenizer = WordPieceTokenizer(vocab_size=5000)
tokenizer.train(texts)

model = MyClassifier(vocab_size=tokenizer.vocab_size, num_classes=3)

classifier = torchTextClassifiers.from_model(
    tokenizer=tokenizer,
    pytorch_model=model,
)
classifier.train(texts, labels, training_config)
predictions = classifier.predict(new_texts)
```

---

## Multi-task example — contrib architecture

For multi-task classification the `contrib` sub-package provides ready-made
classes that follow the interface above.

```python
from torchTextClassifiers import torchTextClassifiers, TrainingConfig
from torchTextClassifiers.contrib import (
    MultiLevelTextClassificationModel,
    MultiLevelCrossEntropyLoss,
)
from torchTextClassifiers.model.components import (
    CategoricalVariableNet,
    ClassificationHead,
    LabelAttentionConfig,
    SentenceEmbedder, SentenceEmbedderConfig,
    TokenEmbedder, TokenEmbedderConfig,
)
from torchTextClassifiers.value_encoder import ValueEncoder

# Assume tokenizer, value_encoder, and model_config are already built.
# value_encoder.num_classes is a list[int] — one count per task level.

token_embedder = TokenEmbedder(TokenEmbedderConfig(
    vocab_size=tokenizer.vocab_size,
    embedding_dim=64,
    padding_idx=tokenizer.padding_idx,
))
cat_net = CategoricalVariableNet(
    categorical_vocabulary_sizes=value_encoder.vocabulary_sizes,
    categorical_embedding_dims=8,
    text_embedding_dim=64,
)

sentence_embedders = []
classification_heads = []
for n_cls in value_encoder.num_classes:
    sentence_embedders.append(SentenceEmbedder(SentenceEmbedderConfig(
        aggregation_method=None,
        label_attention_config=LabelAttentionConfig(n_head=2, num_classes=n_cls, embedding_dim=64),
    )))
    classification_heads.append(ClassificationHead(input_dim=64 + cat_net.output_dim, num_classes=1))

model = MultiLevelTextClassificationModel(
    token_embedder=token_embedder,
    sentence_embedders=sentence_embedders,
    classification_heads=classification_heads,
    categorical_variable_net=cat_net,
)

classifier = torchTextClassifiers.from_model(
    tokenizer=tokenizer,
    pytorch_model=model,
    value_encoder=value_encoder,
)

training_config = TrainingConfig(
    num_epochs=10,
    batch_size=32,
    lr=1e-3,
    raw_categorical_inputs=True,
    loss=MultiLevelCrossEntropyLoss(num_classes=list(value_encoder.num_classes)),
)
classifier.train(X_train, y_train, training_config)
predictions = classifier.predict(X_test)
```

`predictions` is a dict with one key per task level.

See [examples/multilevel_example.py](https://github.com/InseeFrLab/torchTextClassifiers/blob/main/examples/multilevel_example.py)
for the full runnable script.

---

## contrib reference

| Class | Description |
|---|---|
| `MultiLevelTextClassificationModel` | Shared `TokenEmbedder`, one `SentenceEmbedder` + `ClassificationHead` per task |
| `MultiLevelCrossEntropyLoss` | Per-task cross-entropy, optionally weighted by `num_classes` |

```python
from torchTextClassifiers.contrib import (
    MultiLevelTextClassificationModel,
    MultiLevelCrossEntropyLoss,
)
```

These classes are reference implementations — use them directly or as a
starting point for your own architecture.

---

## Saving and loading

`save` and `load` work the same way regardless of which path was used.  Custom
models are serialised as a pickle of the model structure plus a separate
state-dict file; the `_custom_model` flag in the checkpoint tells `load` which
strategy to use.

```python
classifier.save("my_classifier/")
loaded = torchTextClassifiers.load("my_classifier/")
```

---

## Next steps

- **Architecture overview**: {doc}`../architecture/overview` — component reference and design philosophy
- **API reference**: {doc}`../api/wrapper` — full `torchTextClassifiers` API
- **contrib source**: `torchTextClassifiers/contrib/multilevel.py`
