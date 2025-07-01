Architecture Overview
====================

This section provides a detailed overview of the torchTextClassifiers architecture,
including design principles, component relationships, and extension points.

Framework Design
----------------

torchTextClassifiers follows a modular, plugin-based architecture that separates
concerns and enables easy extension with new classifier types.

.. code-block:: text

    ┌─────────────────────────────────────────────────────┐
    │                User Interface                       │
    │  create_fasttext(), build(), train(), predict()    │
    └─────────────────────┬───────────────────────────────┘
                          │
    ┌─────────────────────▼───────────────────────────────┐
    │              torchTextClassifiers                   │
    │         (Main Classifier Interface)                 │
    │                                                     │
    │  ┌─────────────────┐    ┌─────────────────────────┐ │
    │  │ ClassifierType  │    │  ClassifierFactory      │ │
    │  │   (Enum)        │    │    (Registry)           │ │
    │  └─────────────────┘    └─────────────────────────┘ │
    └─────────────────────┬───────────────────────────────┘
                          │
    ┌─────────────────────▼───────────────────────────────┐
    │           Classifier Implementations                │
    │                                                     │
    │  ┌─────────────────────────────────────────────────┐ │
    │  │              FastText                           │ │
    │  │                                                 │ │
    │  │ ┌─────────────┐ ┌──────────────┐ ┌────────────┐ │ │
    │  │ │   Config    │ │   Wrapper    │ │   Model    │ │ │
    │  │ │             │ │              │ │            │ │ │
    │  │ └─────────────┘ └──────────────┘ └────────────┘ │ │
    │  │                       │                        │ │
    │  │ ┌─────────────┐ ┌──────▼──────┐ ┌────────────┐ │ │
    │  │ │ Tokenizer   │ │  Lightning  │ │  Dataset   │ │ │
    │  │ │             │ │   Module    │ │            │ │ │
    │  │ └─────────────┘ └─────────────┘ └────────────┘ │ │
    │  └─────────────────────────────────────────────────┘ │
    └─────────────────────────────────────────────────────┘

Core Components
---------------

1. **torchTextClassifiers**: Main classifier interface that provides a unified API
2. **ClassifierType**: Enum defining available classifier types  
3. **ClassifierFactory**: Registry for classifier implementations
4. **BaseClassifierWrapper**: Abstract base class for classifier implementations
5. **BaseClassifierConfig**: Configuration base class

FastText Implementation
-----------------------

The FastText classifier demonstrates the framework's capabilities with a complete
implementation including:

Architecture Flow
~~~~~~~~~~~~~~~~~

.. code-block:: text

    Input Text: "Hello world example"
           │
           ▼
    ┌─────────────────────────────────────────────────────┐
    │              NGramTokenizer                         │
    │  "Hello" → [hel, ell, llo] + [Hello] + [wor, ord,   │
    │  "world" → [rld] + [world] + [exa, xam, amp, mpl,   │
    │  "example" → [ple, ple] + [example]                 │
    └─────────────────────────────────────────────────────┘
           │
           ▼ (Token IDs: [234, 567, 123, ...])
    ┌─────────────────────────────────────────────────────┐
    │              Embedding Layer                        │
    │  - Learnable embedding matrix: [vocab_size, emb_dim]│
    │  - Maps token IDs to dense vectors                  │
    │  - Supports sparse embeddings for memory efficiency │
    └─────────────────────────────────────────────────────┘
           │
           ▼ (Embeddings: [seq_len, emb_dim])
    ┌─────────────────────────────────────────────────────┐
    │              Pooling Layer                          │
    │  - Average pooling across sequence dimension        │
    │  - Result: [batch_size, emb_dim]                    │
    └─────────────────────────────────────────────────────┘
           │
           ▼ (Pooled features: [batch_size, emb_dim])
    ┌─────────────────────────────────────────────────────┐
    │           Classification Head                       │
    │  - Linear layer: [emb_dim] → [num_classes]          │
    │  - No activation (logits output)                    │
    └─────────────────────────────────────────────────────┘
           │
           ▼ (Logits: [batch_size, num_classes])
        Output Predictions

Mathematical Formulation
~~~~~~~~~~~~~~~~~~~~~~~~~

For text input x = [x₁, x₂, ..., xₙ] and categorical features c = [c₁, c₂, ..., cₘ]:

1. **Token Embeddings**: E(x) = [e₁, e₂, ..., eₙ] where eᵢ ∈ ℝᵈ
2. **Text Representation**: h_text = (1/n) ∑ᵢ eᵢ  
3. **Categorical Embeddings**: h_cat = [E_cat₁(c₁), E_cat₂(c₂), ..., E_catₘ(cₘ)]
4. **Combined Representation**: h = [h_text; h_cat] (concatenation)
5. **Output Logits**: y = W·h + b where W ∈ ℝᶜˣᵈ, b ∈ ℝᶜ

Extension Points
----------------

Adding New Classifiers
~~~~~~~~~~~~~~~~~~~~~~~

To add a new classifier type:

1. Create a new classifier type in the ClassifierType enum
2. Implement BaseClassifierWrapper for your classifier
3. Create a configuration class extending BaseClassifierConfig
4. Register your classifier with ClassifierFactory

Example:

.. code-block:: python

   # 1. Add to ClassifierType enum
   class ClassifierType(Enum):
       FASTTEXT = "fasttext"
       BERT = "bert"  # New classifier type

   # 2. Implement wrapper
   class BertWrapper(BaseClassifierWrapper):
       def __init__(self, config: BertConfig):
           super().__init__(config)
           # Implementation...

   # 3. Register with factory
   ClassifierFactory.register_classifier(ClassifierType.BERT, BertWrapper)

Design Principles
-----------------

1. **Separation of Concerns**: Each component has a single responsibility
2. **Dependency Injection**: Components receive dependencies rather than creating them
3. **Configuration-Driven**: Behavior controlled through configuration objects  
4. **Plugin Architecture**: Easy to add new classifier types
5. **PyTorch Lightning Integration**: Leverage battle-tested training infrastructure
6. **Type Safety**: Strong typing throughout the codebase

Performance Considerations
--------------------------

Memory Management
~~~~~~~~~~~~~~~~~

- Sparse embeddings for large vocabularies
- Lazy loading of model components
- Efficient batch processing
- Memory-mapped dataset loading

Training Optimization
~~~~~~~~~~~~~~~~~~~~~

- Automatic mixed precision support
- Multi-GPU training with PyTorch Lightning
- Gradient accumulation for large batches
- Learning rate scheduling and early stopping

Inference Optimization
~~~~~~~~~~~~~~~~~~~~~~

- Model quantization support
- Batch prediction optimization
- CPU/GPU automatic selection
- Caching for repeated predictions
