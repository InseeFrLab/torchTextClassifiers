# Tutorials

Step-by-step guides to learn torchTextClassifiers through practical examples.

```{toctree}
:maxdepth: 2

basic_classification
multiclass_classification
mixed_features
explainability
multilabel_classification
```

## Overview

These tutorials guide you through common text classification tasks, from basic binary classification to advanced multiclass scenarios.

## Available Tutorials

### Getting Started

::::{grid} 1
:gutter: 3

:::{grid-item-card} {fas}`star` Binary Classification
:link: basic_classification
:link-type: doc

**Recommended first tutorial**

Build a sentiment classifier for product reviews. Learn the complete workflow from data preparation to evaluation.

**What you'll learn:**
- Creating and training tokenizers
- Configuring models
- Training with validation data
- Making predictions
- Evaluating performance

**Difficulty:** Beginner | **Time:** 15 minutes
:::

::::

### Intermediate Tutorials

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {fas}`layer-group` Multiclass Classification
:link: multiclass_classification
:link-type: doc

Classify text into 3+ categories with proper handling of class imbalance and evaluation metrics.

**What you'll learn:**
- Multiclass model configuration
- Class distribution analysis
- Reproducibility with seeds
- Confusion matrices
- Advanced evaluation metrics

**Difficulty:** Intermediate | **Time:** 20 minutes
:::

:::{grid-item-card} {fas}`puzzle-piece` Mixed Features
:link: mixed_features
:link-type: doc

Combine text with categorical variables for improved classification performance.

**What you'll learn:**
- Adding categorical features alongside text
- Configuring categorical embeddings
- Comparing performance improvements
- Feature combination strategies

**Difficulty:** Intermediate | **Time:** 25 minutes
:::

::::

### Advanced Tutorials

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {fas}`lightbulb` Explainability
:link: explainability
:link-type: doc

Understand which words and characters drive your model's predictions.

**What you'll learn:**
- Generating attribution scores with Captum
- Word-level and character-level visualizations
- Identifying influential tokens
- Interactive explainability mode

**Difficulty:** Advanced | **Time:** 30 minutes
:::

:::{grid-item-card} {fas}`tags` Multilabel Classification
:link: multilabel_classification
:link-type: doc

Assign multiple labels to each text sample for complex classification scenarios.

**What you'll learn:**
- Ragged lists vs. one-hot encoding
- Configuring BCEWithLogitsLoss
- Multilabel evaluation metrics
- Handling variable labels per sample

**Difficulty:** Advanced | **Time:** 30 minutes
:::

::::

## Learning Path

We recommend following this learning path:

```{mermaid}
graph LR
    A[Quick Start] --> B[Binary Classification]
    B --> C[Multiclass Classification]
    C --> D[Mixed Features]
    C --> F[Multilabel Classification]
    D --> E[Explainability]
    F --> E

    style A fill:#e3f2fd
    style B fill:#bbdefb
    style C fill:#90caf9
    style D fill:#64b5f6
    style E fill:#1976d2
    style F fill:#42a5f5
```

1. **Start with**: {doc}`../getting_started/quickstart` - Get familiar with the basics
2. **Then**: {doc}`basic_classification` - Understand the complete workflow
3. **Next**: {doc}`multiclass_classification` - Handle multiple classes
4. **Branch out**: {doc}`mixed_features` for categorical features OR {doc}`multilabel_classification` for multiple labels
5. **Master**: {doc}`explainability` - Understand your model's predictions

## Tutorial Format

Each tutorial follows a consistent structure:

**Learning Objectives**
: What you'll be able to do after completing the tutorial

**Prerequisites**
: What you need to know before starting

**Complete Code**
: Full working example you can copy and run

**Step-by-Step Walkthrough**
: Detailed explanation of each step

**Customization**
: How to adapt the code to your needs

**Common Issues**
: Troubleshooting tips and solutions

**Next Steps**
: Where to go after finishing

## Tips for Learning

### Run the Code

Don't just read - run the examples! Modify them to see what happens:

```python
# Try different values
model_config = ModelConfig(
    embedding_dim=128,  # Was 64 - what changes?
    num_classes=2
)
```

### Start Simple

Begin with the Quick Start, then move to Binary Classification. Don't skip ahead!

### Use Your Own Data

Once you understand the examples, try them with your own text data:

```python
# Your data
my_texts = ["your", "text", "samples"]
my_labels = [0, 1, 0]

# Same workflow
classifier.train(my_texts, my_labels, training_config)
```

### Experiment

- Try different tokenizers (WordPiece vs NGram)
- Adjust hyperparameters (learning rate, embedding dim)
- Compare model sizes
- Test different batch sizes

### Read the Errors

Error messages are helpful! They often tell you exactly what's wrong:

```python
# Error: num_classes=2 but got label 3
# Solution: Check your labels - should be 0, 1 (not 1, 2, 3)
```

## Getting Help

Stuck on a tutorial? Here's how to get help:

1. **Check Common Issues**: Each tutorial has a troubleshooting section
2. **Read the API docs**: {doc}`../api/index` for detailed parameter descriptions
3. **Review architecture**: {doc}`../architecture/overview` for how components work
4. **Ask questions**: [GitHub Discussions](https://github.com/InseeFrLab/torchTextClassifiers/discussions)
5. **Report bugs**: [GitHub Issues](https://github.com/InseeFrLab/torchTextClassifiers/issues)

## Additional Resources

### Example Scripts

All tutorials are based on runnable examples in the repository:

- [examples/basic_classification.py](https://github.com/InseeFrLab/torchTextClassifiers/blob/main/examples/basic_classification.py)
- [examples/multiclass_classification.py](https://github.com/InseeFrLab/torchTextClassifiers/blob/main/examples/multiclass_classification.py)
- [examples/using_additional_features.py](https://github.com/InseeFrLab/torchTextClassifiers/blob/main/examples/using_additional_features.py)
- [examples/advanced_training.py](https://github.com/InseeFrLab/torchTextClassifiers/blob/main/examples/advanced_training.py)
- [examples/simple_explainability_example.py](https://github.com/InseeFrLab/torchTextClassifiers/blob/main/examples/simple_explainability_example.py)

### Jupyter Notebooks

Interactive notebooks for hands-on learning:

- [Basic example notebook](https://github.com/InseeFrLab/torchTextClassifiers/blob/main/notebooks/example.ipynb)
- [Multilabel classification notebook](https://github.com/InseeFrLab/torchTextClassifiers/blob/main/notebooks/multilabel_classification.ipynb)

## Contributing

Want to contribute a tutorial? We welcome:

- New use cases
- Alternative approaches
- Real-world examples
- Performance tips

See our [contributing guidelines](https://github.com/InseeFrLab/torchTextClassifiers/blob/main/CONTRIBUTING.md) to get started!

## What's Next?

Ready to start? Choose your path:

- **New to text classification?** Start with {doc}`../getting_started/quickstart`
- **Want to dive deeper?** Begin with {doc}`basic_classification`
- **Ready for multiclass?** Jump to {doc}`multiclass_classification`
- **Need API details?** Check {doc}`../api/index`
