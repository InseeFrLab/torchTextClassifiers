# Examples

This directory contains comprehensive examples demonstrating how to use the torchTextClassifiers package for various text classification tasks.

## 📁 Available Examples

### 1. [Basic Classification](basic_classification.py)
A simple binary sentiment classification example that covers:
- Creating a FastText classifier
- Preparing training and validation data
- Building and training the model
- Making predictions and evaluating performance
- Saving model configuration

**Run the example:**
```bash
cd examples
uv run python basic_classification.py
```

**What you'll learn:**
- Basic API usage
- Binary classification workflow
- Model evaluation
- Configuration persistence

### 2. [Multi-class Classification](multiclass_classification.py)
Demonstrates 3-class sentiment analysis (positive, negative, neutral):
- Multi-class data preparation
- Class distribution handling
- Detailed result analysis
- Configuration loading and validation

**Run the example:**
```bash
cd examples
uv run python multiclass_classification.py
```

**What you'll learn:**
- Multi-class classification setup
- Class imbalance considerations
- Advanced result interpretation
- Model serialization/deserialization

### 3. [Mixed Features](mixed_features.py)
Shows how to combine text and categorical features:
- Text + categorical data preparation
- Feature engineering for categorical variables
- Comparing mixed vs. text-only models
- Performance analysis with different feature types

**Run the example:**
```bash
cd examples
uv run python mixed_features.py
```

**What you'll learn:**
- Mixed feature classification
- Categorical feature configuration
- Feature importance analysis
- Model comparison techniques

### 4. [Advanced Training](advanced_training.py)
Explores advanced training configurations:
- Custom PyTorch Lightning trainer parameters
- Different hardware configurations (CPU/GPU)
- Training optimization techniques
- Model comparison and selection

**Run the example:**
```bash
cd examples
uv run python advanced_training.py
```

**What you'll learn:**
- Advanced training configurations
- Hardware-specific optimizations
- Training parameter tuning
- Model performance comparison

## 🚀 Quick Start

To run any example:

1. **Install dependencies:**
   ```bash
   uv sync
   ```

2. **Navigate to examples directory:**
   ```bash
   cd examples
   ```

3. **Run an example:**
   ```bash
   uv run python basic_classification.py
   ```

## 📊 Example Outputs

### Basic Classification
```
🚀 Basic Text Classification Example
==================================================
📝 Creating sample data...
Training samples: 10
Validation samples: 2
Test samples: 3

🏗️ Creating FastText classifier...
🔨 Building model...
✅ Model built successfully!

🎯 Training model...
✅ Training completed!

🔮 Making predictions...
Predictions: [1 0 1]
True labels: [1 0 1]
Test accuracy: 1.000

📊 Detailed Results:
----------------------------------------
1. ✅ Predicted: Positive
   Text: This is an amazing product with great features!...

2. ✅ Predicted: Negative
   Text: Completely disappointed with this purchase...

3. ✅ Predicted: Positive
   Text: Excellent build quality and works as expected...

💾 Saving model configuration...
✅ Configuration saved to 'basic_classifier_config.json'

🎉 Example completed successfully!
```

### Multi-class Classification
```
🎭 Multi-class Text Classification Example
==================================================
📝 Creating multi-class sentiment data...
Training samples: 15
Class distribution: Negative=5, Neutral=5, Positive=5

🏗️ Creating multi-class FastText classifier...
🔨 Building model...
✅ Model built successfully!

🎯 Training model...
✅ Training completed!

📊 Detailed Results:
------------------------------------------------------------
1. ✅ Predicted: Negative, True: Negative
   Text: This is absolutely horrible!

2. ✅ Predicted: Neutral, True: Neutral
   Text: It's an average product, nothing more.

3. ✅ Predicted: Positive, True: Positive
   Text: Fantastic! Love every aspect of it!

Final Accuracy: 3/6 = 0.500
```

## 🛠️ Customizing Examples

### Modify Data
You can easily adapt the examples to your own data:

```python
# Replace the example data with your own
X_train = np.array([
    "Your text sample 1",
    "Your text sample 2",
    # ... more samples
])
y_train = np.array([0, 1, ...])  # Your labels
```

### Adjust Parameters
Experiment with different model parameters:

```python
classifier = create_fasttext(
    embedding_dim=200,    # Increase for better representations
    num_tokens=20000,     # Increase for larger vocabularies
    min_count=3,          # Increase to filter rare words
    num_epochs=100,       # Increase for more training
    batch_size=64,        # Adjust based on your hardware
)
```

### Add Custom Features
Extend examples with your own categorical features:

```python
# Add your categorical features
categorical_features = np.array([
    [category1, category2, category3],
    # ... more feature vectors
])

X_mixed = np.column_stack([text_data, categorical_features])
```

## 📈 Performance Tips

### For Better Accuracy
1. **Increase embedding dimensions** for complex tasks
2. **Use more training data** when available
3. **Tune n-gram parameters** (min_n, max_n) for your domain
4. **Experiment with batch sizes** and learning rates
5. **Consider mixed features** if you have structured data

### For Faster Training
1. **Use sparse embeddings** for large vocabularies
2. **Increase batch size** (if memory allows)
3. **Reduce embedding dimensions** for faster convergence
4. **Use CPU training** for small datasets
5. **Adjust num_workers** for optimal data loading

### For Large Datasets
1. **Use gradient accumulation** with small batch sizes
2. **Enable mixed precision training** (precision=16)
3. **Implement data streaming** for very large datasets
4. **Use multiple GPUs** if available

## 🐛 Troubleshooting

### Common Issues

1. **Memory errors:**
   - Reduce batch_size
   - Use sparse=True
   - Reduce embedding_dim

2. **Slow training:**
   - Increase batch_size
   - Reduce num_workers
   - Use CPU for small datasets

3. **Poor accuracy:**
   - Increase training data
   - Tune hyperparameters
   - Check data quality
   - Increase num_epochs

4. **Import errors:**
   - Run `uv sync` to install dependencies
   - Check Python version compatibility

### Getting Help

If you encounter issues:

1. Check the [main README](../README.md) for setup instructions
2. Review the [API documentation](../docs/api_reference.md)
3. Look at similar examples for reference
4. Open an issue on GitHub with your specific problem

## 🔗 Related Documentation

- [Main README](../README.md) - Package overview and installation
- [API Reference](../docs/api_reference.md) - Complete API documentation
- [Developer Guide](../docs/developer_guide.md) - Adding new classifier types
- [Tests](../tests/) - Unit and integration tests for reference

## 🤝 Contributing Examples

We welcome new examples! If you have a use case that would benefit others:

1. Follow the existing example structure
2. Include comprehensive comments
3. Add error handling and validation
4. Test your example thoroughly
5. Update this README with your addition

Example template structure:
```python
"""
Your Example Title

Brief description of what this example demonstrates.
"""

import numpy as np
from torchTextClassifiers import create_fasttext

def main():
    print("🚀 Your Example Title")
    print("=" * 50)
    
    # Your implementation here
    
    print("🎉 Example completed successfully!")

if __name__ == "__main__":
    main()
```