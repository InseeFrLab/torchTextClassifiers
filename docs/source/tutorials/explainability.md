# Model Explainability

Understand which words drive your model's predictions using two complementary methods:
**Captum** (gradient-based attribution) and **label attention** (class-specific cross-attention).

## Learning Objectives

By the end of this tutorial, you'll be able to:

- Generate token-level attribution scores with Captum
- Use label attention to see which tokens influence each class
- Visualize word-level contributions
- Choose the right explainability method for your use case

## Prerequisites

- Completed {doc}`basic_classification` tutorial
- (Optional) Understanding of gradient-based attribution methods

## What Is Explainability?

**Model explainability** reveals which parts of the input contribute most to a prediction. For text classification:

- **Word-level**: Which words influence the prediction?
- **Character-level**: Which characters matter most?
- **Attribution scores**: How much each token contributes (positive or negative)

### Why Use Explainability?

✅ **Debugging**: Identify if model focuses on correct features
✅ **Trust**: Understand and validate model decisions
✅ **Bias detection**: Discover unwanted correlations
✅ **Feature engineering**: Guide feature selection

---

## Method 1: Captum (Integrated Gradients)

Captum computes gradient-based token attributions, measuring how much each token
contributes to the final prediction score.

### Setup

Install the optional explainability dependencies:

```bash
uv sync --extra explainability
# or
pip install torchTextClassifiers[explainability]
```

### Quick Example

```python
import numpy as np
from torchTextClassifiers import ModelConfig, TrainingConfig, torchTextClassifiers
from torchTextClassifiers.tokenizers import WordPieceTokenizer

# Training data
X_train = np.array([
    "I love this product",
    "Great quality and excellent service",
    "Amazing design and fantastic performance",
    "This is terrible quality",
    "Poor design and cheap materials",
    "Awful experience with this product"
])
y_train = np.array([1, 1, 1, 0, 0, 0])

tokenizer = WordPieceTokenizer(vocab_size=5000)
tokenizer.train(X_train.tolist())

model_config = ModelConfig(embedding_dim=50, num_classes=2)
classifier = torchTextClassifiers(tokenizer=tokenizer, model_config=model_config)

training_config = TrainingConfig(num_epochs=25, batch_size=8, lr=1e-3,
                                 raw_categorical_inputs=False, raw_labels=False)
classifier.train(X_train, y_train, training_config=training_config)

# Predict with Captum explainability
result = classifier.predict(
    np.array(["This product is amazing!"]),
    explain_with_captum=True,          # <-- enable Captum attribution
)

prediction  = result["prediction"][0][0].item()
confidence  = result["confidence"][0][0].item()
attributions = result["captum_attributions"][0][0]  # shape: (seq_len,)

print(f"Prediction: {'Positive' if prediction == 1 else 'Negative'}")
print(f"Confidence: {confidence:.4f}")
print(f"Attribution shape: {attributions.shape}")
```

### Output Dictionary

When `explain_with_captum=True`, the result contains additional keys:

```python
{
    "prediction": tensor,                # class predictions (decoded if ValueEncoder present)
    "confidence": tensor,                # confidence scores
    "captum_attributions": tensor,       # shape (batch_size, top_k, seq_len)
    "label_attention_attributions": None,
    "offset_mapping": list,              # character positions of each token
    "word_ids": list,                    # word index for each token
}
```

**Attribution values:**
- Higher positive values → stronger support for the predicted class
- Negative values → oppose the predicted class
- Near zero → neutral contribution

### Visualize Word Contributions

```python
def explain_with_captum(classifier, text):
    result = classifier.predict(
        np.array([text]),
        explain_with_captum=True
    )

    prediction  = result["prediction"][0][0].item()
    confidence  = result["confidence"][0][0].item()
    attributions = result["captum_attributions"][0][0]   # (seq_len,)
    offset_mapping = result["offset_mapping"][0]

    print(f"Text: '{text}'")
    print(f"Prediction: {prediction}  (confidence: {confidence:.4f})")

    # Map attributions to characters
    char_attrs = [0.0] * len(text)
    for (start, end), score in zip(offset_mapping, attributions.tolist()):
        for i in range(start, end):
            char_attrs[i] = score

    # Aggregate to words
    words = text.split()
    char_idx = 0
    print("\nWord Contributions:")
    print("-" * 50)
    for word in words:
        scores = char_attrs[char_idx : char_idx + len(word)]
        avg = sum(scores) / len(scores) if scores else 0.0
        bar = "█" * max(0, int(avg * 40))
        print(f"{word:>15} | {bar:<40} {avg:.4f}")
        char_idx += len(word) + 1  # +1 for space

explain_with_captum(classifier, "This product is amazing!")
```

---

## Method 2: Label Attention

Label attention is a **built-in architectural feature** that produces one sentence
embedding per class via a learnable cross-attention mechanism. It is:

- **Faster than Captum** at inference time (no gradient computation)
- **Class-specific**: shows which tokens matter for *each individual class*
- Enabled at model construction time via `n_heads_label_attention` in `ModelConfig`

### Enable Label Attention

```python
from torchTextClassifiers import ModelConfig, TrainingConfig, torchTextClassifiers
from torchTextClassifiers.model.components import AttentionConfig

model_config = ModelConfig(
    embedding_dim=96,
    num_classes=6,
    attention_config=AttentionConfig(   # self-attention (optional but recommended)
        n_layers=2,
        n_head=4,
        n_kv_head=4,
        sequence_len=50,
    ),
    n_heads_label_attention=4,          # <-- enables label attention with 4 heads
)

classifier = torchTextClassifiers(
    tokenizer=tokenizer,
    model_config=model_config,
)
classifier.train(X_train, y_train, training_config=training_config)
```

### Predict with Label Attention

```python
result = classifier.predict(
    X_test,
    explain_with_label_attention=True,   # <-- request attention weights
)

# Attention matrix: which tokens are important for each class
label_attention = result["label_attention_attributions"]
# Shape: (batch_size, n_head, num_classes, seq_len)
```

### Output Dictionary

```python
{
    "prediction": tensor,                      # top-k class predictions
    "confidence": tensor,                      # confidence scores
    "captum_attributions": None,
    "label_attention_attributions": tensor,    # (batch_size, n_head, num_classes, seq_len)
    "offset_mapping": list,
    "word_ids": list,
}
```

### Inspect Per-Class Token Importance

```python
import torch

# Average across heads for readability
# label_attention: (batch_size, n_head, num_classes, seq_len)
per_class_scores = label_attention[0].mean(dim=0)   # (num_classes, seq_len)

tokens = tokenizer.tokenize([text]).input_ids[0]
class_names = ["World", "Sports", "Business", "Sci/Tech"]  # example

print("Token importance by class:")
for class_idx, class_name in enumerate(class_names):
    scores = per_class_scores[class_idx]
    top_token_idx = scores.argmax().item()
    print(f"  [{class_name}] most attended token index: {top_token_idx} "
          f"(score: {scores[top_token_idx]:.4f})")
```

### Both Methods Together

You can combine both explainability methods in a single `predict` call:

```python
result = classifier.predict(
    X_test,
    explain_with_captum=True,
    explain_with_label_attention=True,
)

captum_attrs   = result["captum_attributions"]          # gradient-based
label_attrs    = result["label_attention_attributions"] # attention-based
```

---

## Choosing Between Methods

| | Captum | Label Attention |
|---|---|---|
| **Setup** | Requires `[explainability]` extra | Built into the model |
| **Speed** | Slower (gradient computation) | Fast (forward pass only) |
| **Granularity** | One attribution per token | One per (token, class) pair |
| **Works with any model** | Yes | Requires `n_heads_label_attention` set at training time |
| **Result key** | `captum_attributions` | `label_attention_attributions` |

**Rule of thumb:**
- Use Captum for a single overall attribution score per token.
- Use label attention when you want to understand how each *class* attends to different
  parts of the input (multi-class explainability).

---

## Debugging with Explainability

### Case 1: Model Ignores Negation

```python
explain_with_captum(classifier, "This product is not good")
# If 'not' has low attribution and 'good' is high → model misses negation
# Solution: add more negation examples to training data
```

### Case 2: Spurious Correlations

```python
explain_with_captum(classifier, "Product from Location X is excellent")
# If the location has high attribution → spurious correlation learned
# Solution: audit and balance the training set
```

### Case 3: Low Confidence

```python
result = classifier.predict(np.array(["Product arrived on time"]),
                             explain_with_captum=True)
# Low confidence + low attribution scores = text has no strong class signal
# This is expected and correct model behaviour
```

---

## Common Issues

### Issue 1: Captum Not Installed

**Error:** `ImportError: Captum is not installed`

**Solution:**
```bash
uv sync --extra explainability
```

### Issue 2: Label Attention Explainability Fails

**Error:** `RuntimeError: Label attention explainability is enabled, but the model was not configured with label attention`

**Solution:** Set `n_heads_label_attention` in `ModelConfig` **before training**:
```python
model_config = ModelConfig(
    embedding_dim=96,
    num_classes=4,
    n_heads_label_attention=4,
)
```
You cannot enable label attention on an already-trained model without retraining.

### Issue 3: All Attributions Near Zero

**Possible causes:**
- Model not well-trained
- Text has no discriminative features for that class

**Try:**
- Train longer or with more data
- Check prediction confidence first

---

## Summary

**Key takeaways:**
- Use `explain_with_captum=True` for gradient-based token attributions
- Use `explain_with_label_attention=True` for class-specific attention weights (requires `n_heads_label_attention` set at model init)
- Both methods return `offset_mapping` and `word_ids` for mapping token scores back to words
- Result keys: `captum_attributions` and `label_attention_attributions`

Ready for multilabel classification? Continue to {doc}`multilabel_classification`!
