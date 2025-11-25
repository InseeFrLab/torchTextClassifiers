# Model Explainability

Understand which words and characters drive your model's predictions using attribution analysis.

## Learning Objectives

By the end of this tutorial, you'll be able to:

- Generate explanations for individual predictions
- Visualize word-level and character-level contributions
- Identify the most influential tokens
- Use interactive explainability for debugging
- Understand Captum integration for attribution analysis

## Prerequisites

- Completed {doc}`basic_classification` tutorial
- Familiarity with model predictions
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

## Complete Example

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

y_train = np.array([1, 1, 1, 0, 0, 0])  # 1 = Positive, 0 = Negative

X_val = np.array([
    "Good product with decent quality",
    "Bad quality and poor service"
])
y_val = np.array([1, 0])

# Create and train tokenizer
tokenizer = WordPieceTokenizer(vocab_size=5000)
tokenizer.train(X_train.tolist())

# Create model
model_config = ModelConfig(
    embedding_dim=50,
    num_classes=2
)

classifier = torchTextClassifiers(
    tokenizer=tokenizer,
    model_config=model_config
)

# Train
training_config = TrainingConfig(
    num_epochs=25,
    batch_size=8,
    lr=1e-3
)

classifier.train(
    X_train, y_train, X_val, y_val,
    training_config=training_config
)

# Test with explainability
test_text = "This product is amazing!"

result = classifier.predict(
    np.array([test_text]),
    explain=True  # Enable explainability
)

# Extract results
prediction = result["prediction"][0][0].item()
confidence = result["confidence"][0][0].item()
attributions = result["attributions"][0][0]  # Token-level attributions

print(f"Prediction: {'Positive' if prediction == 1 else 'Negative'}")
print(f"Confidence: {confidence:.4f}")
print(f"Attribution shape: {attributions.shape}")
```

## Step-by-Step Walkthrough

### 1. Enable Explainability

Add `explain=True` to `predict()`:

```python
result = classifier.predict(
    X_test,
    explain=True  # Generate attribution scores
)
```

### 2. Understanding the Output

The result dictionary contains additional keys:

```python
{
    "prediction": tensor,           # Class predictions
    "confidence": tensor,           # Confidence scores
    "attributions": tensor,         # Token-level attribution scores
    "offset_mapping": list,         # Character positions of tokens
    "word_ids": list               # Word IDs for each token
}
```

**Attributions shape:** `(batch_size, top_k, sequence_length)`
- Higher values = stronger influence on prediction
- Positive values = supports predicted class
- Negative values = opposes predicted class

### 3. Visualize Word Contributions

Map token attributions to words:

```python
from torchTextClassifiers.utilities.plot_explainability import map_attributions_to_word

# Get attribution data
attributions = result["attributions"][0][0]  # Shape: (seq_len,)
word_ids = result["word_ids"][0]             # List of word IDs

# Map to words
words = test_text.split()
word_attributions = []

for word_idx in range(len(words)):
    # Find tokens belonging to this word
    token_mask = [wid == word_idx for wid in word_ids]
    token_attrs = attributions[token_mask]

    if len(token_attrs) > 0:
        word_attr = token_attrs.mean().item()
        word_attributions.append((words[word_idx], word_attr))

# Display results
print("\nWord-Level Contributions:")
print("-" * 50)
for word, score in word_attributions:
    print(f"{word:>15} | {'█' * int(score * 40)} {score:.4f}")
```

### 4. Character-Level Visualization

For finer-grained analysis:

```python
from torchTextClassifiers.utilities.plot_explainability import map_attributions_to_char

# Map token attributions to characters
char_attributions = map_attributions_to_char(
    attributions.unsqueeze(0),  # Add batch dimension
    result["offset_mapping"][0],
    test_text
)[0]

# Visualize
print("\nCharacter-Level Contributions:")
for i, char in enumerate(test_text):
    if i < len(char_attributions):
        score = char_attributions[i]
        bar = "█" * int(score * 20)
        print(f"{char} | {bar} {score:.4f}")
```

## Complete Visualization Example

Here's a complete function to visualize word importance:

```python
def explain_prediction(classifier, text):
    """Generate and visualize explanations for a prediction."""
    import numpy as np

    # Get prediction with explainability
    result = classifier.predict(
        np.array([text]),
        top_k=1,
        explain=True
    )

    # Extract prediction info
    prediction = result["prediction"][0][0].item()
    confidence = result["confidence"][0][0].item()
    sentiment = "Positive" if prediction == 1 else "Negative"

    print(f"Text: '{text}'")
    print(f"Prediction: {sentiment} (confidence: {confidence:.4f})")
    print("\n" + "="*60)

    # Get attributions
    attributions = result["attributions"][0][0]
    offset_mapping = result["offset_mapping"][0]

    # Map to characters
    from torchTextClassifiers.utilities.plot_explainability import map_attributions_to_char
    char_attrs = map_attributions_to_char(
        attributions.unsqueeze(0),
        offset_mapping,
        text
    )[0]

    # Group by words
    words = text.split()
    char_idx = 0
    word_scores = []

    for word in words:
        word_len = len(word)
        word_attrs = char_attrs[char_idx:char_idx + word_len]

        if len(word_attrs) > 0:
            avg_attr = sum(word_attrs) / len(word_attrs)
            word_scores.append((word, avg_attr))

        char_idx += word_len + 1  # +1 for space

    # Visualize
    max_score = max(score for _, score in word_scores) if word_scores else 1

    print("Word Contributions:")
    print("-" * 60)
    for word, score in word_scores:
        bar_length = int((score / max_score) * 40)
        bar = "█" * bar_length
        print(f"{word:>15} | {bar:<40} {score:.4f}")

    # Show top contributor
    if word_scores:
        top_word, top_score = max(word_scores, key=lambda x: x[1])
        print("-" * 60)
        print(f"Most influential: '{top_word}' (score: {top_score:.4f})")

# Use it
explain_prediction(classifier, "This product is amazing!")
explain_prediction(classifier, "Poor quality and terrible service")
```

## Interactive Explainability

Create an interactive tool for exploring predictions:

```python
def interactive_explainability(classifier):
    """Interactive mode for exploring model predictions."""
    print("\n" + "="*60)
    print("Interactive Explainability Mode")
    print("="*60)
    print("Enter text to see predictions and explanations!")
    print("Type 'quit' to exit.\n")

    while True:
        user_text = input("Enter text: ").strip()

        if user_text.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if not user_text:
            print("Please enter some text.")
            continue

        try:
            explain_prediction(classifier, user_text)
            print("\n" + "-"*60 + "\n")
        except Exception as e:
            print(f"Error: {e}")

# Use it
interactive_explainability(classifier)
```

## Understanding Attribution Scores

### What Do Scores Mean?

- **High positive scores**: Strong support for predicted class
- **Low/negative scores**: Opposition to predicted class
- **Zero scores**: Neutral contribution

### Example Interpretation

For positive sentiment prediction:

```
Word Contributions:
------------------------------------------------------------
           This | █████ 0.1234
        product | ████████████████ 0.4567
             is | ██ 0.0543
        amazing | ██████████████████████████████ 0.8901
              ! | ███ 0.0876
------------------------------------------------------------
Most influential: 'amazing' (score: 0.8901)
```

**Interpretation:**
- "amazing" strongly indicates positive sentiment (0.89)
- "product" moderately supports positive (0.46)
- "is" is nearly neutral (0.05)

## Debugging with Explainability

### Case 1: Unexpected Predictions

```python
test_text = "This product is not good"
explain_prediction(classifier, test_text)

# Output might show:
# Word Contributions:
#            not | ████ 0.12  <- Low attribution!
#           good | ██████████ 0.45  <- High attribution for "good"
```

**Problem**: Model ignores "not", focuses on "good"
**Solution**: Add more negation examples to training data

### Case 2: Correct Predictions, Wrong Reasons

```python
test_text = "Product from China is excellent"
explain_prediction(classifier, test_text)

# If "China" has high attribution, model may have learned spurious correlation
```

**Problem**: Model uses irrelevant features
**Solution**: Audit training data for bias, balance dataset

### Case 3: Low Confidence

```python
test_text = "Product arrived on time"
result = classifier.predict(np.array([test_text]), explain=True)
confidence = result["confidence"][0][0].item()  # Low confidence

explain_prediction(classifier, test_text)
# All words have similar low attribution scores
```

**Interpretation**: Text doesn't contain strong sentiment indicators
**This is correct behavior**: Model appropriately uncertain

## Advanced: Custom Attribution Methods

By default, torchTextClassifiers uses integrated gradients. For custom attribution:

```python
from torchTextClassifiers.utilities.plot_explainability import generate_attributions
from captum.attr import LayerIntegratedGradients

# Access the underlying model
model = classifier.model

# Create custom attribution method
attribution_method = LayerIntegratedGradients(
    model,
    model.text_embedder.embedding
)

# Generate attributions
attributions = generate_attributions(
    classifier,
    texts=["Your text here"],
    attribution_method=attribution_method
)
```

## Common Issues

### Issue 1: Explainability Fails

**Error:** "explain=True requires captum package"

**Solution:** Install explainability dependencies:
```bash
uv sync --extra explainability
```

### Issue 2: All Attributions Near Zero

**Possible causes:**
- Model not well-trained
- Text contains no discriminative features
- Attribution method sensitivity

**Try:**
- Train longer or with more data
- Check prediction confidence
- Verify model performance on test set

### Issue 3: Inconsistent Attributions

**Problem:** Same word has different attributions in different contexts

**This is expected!** Attribution considers:
- Surrounding context
- Position in sentence
- Interaction with other words

## Best Practices

1. **Always check confidence:** Low confidence = less reliable attributions
2. **Compare multiple examples:** Look for patterns across predictions
3. **Validate with domain knowledge:** Do highlighted words make sense?
4. **Use for debugging, not blind trust:** Attributions are approximations
5. **Check training data:** High attribution may reveal training biases

## Real-World Use Cases

### Sentiment Analysis

```python
positive_review = "Excellent product with amazing quality"
negative_review = "Terrible product with poor quality"

for review in [positive_review, negative_review]:
    explain_prediction(classifier, review)
    print("\n" + "="*60 + "\n")
```

Verify that sentiment words ("excellent", "terrible") have highest attribution.

### Spam Detection

```python
spam_text = "Click here for free money now!"
explain_prediction(spam_classifier, spam_text)
```

Check if "free", "click", "now" are highlighted (common spam indicators).

### Topic Classification

```python
sports_text = "The team won the championship game"
explain_prediction(topic_classifier, sports_text)
```

Verify "team", "championship", "game" drive sports prediction.

## Customization

### Batch Explainability

Explain multiple texts at once:

```python
test_texts = [
    "Great product",
    "Terrible experience",
    "Average quality"
]

result = classifier.predict(
    np.array(test_texts),
    explain=True
)

for i, text in enumerate(test_texts):
    print(f"\nText {i+1}: {text}")
    attributions = result["attributions"][i][0]
    print(f"Max attribution: {attributions.max():.4f}")
```

### Save Explanations

Export attributions for analysis:

```python
import json

explanations = []
for text in test_texts:
    result = classifier.predict(np.array([text]), explain=True)

    explanations.append({
        "text": text,
        "prediction": int(result["prediction"][0][0].item()),
        "confidence": float(result["confidence"][0][0].item()),
        "attributions": result["attributions"][0][0].tolist()
    })

# Save to JSON
with open("explanations.json", "w") as f:
    json.dump(explanations, f, indent=2)
```

## Summary

**Key takeaways:**
- Use `explain=True` to generate attribution scores
- Visualize word and character contributions
- High attribution = strong influence on prediction
- Use explainability for debugging and validation
- Check if model focuses on correct features

Ready for multilabel classification? Continue to {doc}`multilabel_classification`!
