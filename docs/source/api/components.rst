Model Components
================

Modular torch.nn.Module components for building custom architectures.

.. currentmodule:: torchTextClassifiers.model.components

Text Embedding
--------------

Text embedding is split into two composable stages:

1. **TokenEmbedder** — maps each token to a dense vector (with optional self-attention). Output: ``(batch, seq_len, embedding_dim)``.
2. **SentenceEmbedder** — aggregates token vectors into a sentence embedding. Output: ``(batch, embedding_dim)`` or ``(batch, num_classes, embedding_dim)`` with label attention.

TokenEmbedder
~~~~~~~~~~~~~

Embeds tokenized text with optional self-attention.

.. autoclass:: torchTextClassifiers.model.components.text_embedder.TokenEmbedder
   :members:
   :undoc-members:
   :show-inheritance:

TokenEmbedderConfig
~~~~~~~~~~~~~~~~~~~

Configuration for TokenEmbedder.

.. autoclass:: torchTextClassifiers.model.components.text_embedder.TokenEmbedderConfig
   :members:
   :undoc-members:
   :show-inheritance:

Example:

.. code-block:: python

   from torchTextClassifiers.model.components import (
       TokenEmbedder, TokenEmbedderConfig, AttentionConfig,
   )

   # Simple token embedder (no self-attention)
   config = TokenEmbedderConfig(
       vocab_size=5000,
       embedding_dim=128,
       padding_idx=0,
   )
   token_embedder = TokenEmbedder(config)
   out = token_embedder(input_ids, attention_mask)
   # out["token_embeddings"]: (batch, seq_len, 128)

   # With self-attention
   attention_config = AttentionConfig(
       n_layers=2,
       n_head=4,
       n_kv_head=4,
       positional_encoding=False,
   )
   config = TokenEmbedderConfig(
       vocab_size=5000,
       embedding_dim=128,
       padding_idx=0,
       attention_config=attention_config,
   )
   token_embedder = TokenEmbedder(config)

SentenceEmbedder
~~~~~~~~~~~~~~~~

Aggregates per-token embeddings into a single sentence embedding.

.. autoclass:: torchTextClassifiers.model.components.text_embedder.SentenceEmbedder
   :members:
   :undoc-members:
   :show-inheritance:

SentenceEmbedderConfig
~~~~~~~~~~~~~~~~~~~~~~

Configuration for SentenceEmbedder.

.. autoclass:: torchTextClassifiers.model.components.text_embedder.SentenceEmbedderConfig
   :members:
   :undoc-members:
   :show-inheritance:

LabelAttentionConfig
~~~~~~~~~~~~~~~~~~~~

Configuration for the label-attention aggregation mode.

.. autoclass:: torchTextClassifiers.model.components.text_embedder.LabelAttentionConfig
   :members:
   :undoc-members:
   :show-inheritance:

Example:

.. code-block:: python

   from torchTextClassifiers.model.components import (
       SentenceEmbedder, SentenceEmbedderConfig,
       LabelAttentionConfig,
   )

   # Mean-pooling (default)
   sentence_embedder = SentenceEmbedder(SentenceEmbedderConfig(aggregation_method="mean"))
   out = sentence_embedder(token_embeddings, attention_mask)
   # out["sentence_embedding"]: (batch, 128)

   # Label attention — one embedding per class
   sentence_embedder = SentenceEmbedder(SentenceEmbedderConfig(
       aggregation_method=None,
       label_attention_config=LabelAttentionConfig(
           n_head=4,
           num_classes=6,
           embedding_dim=128,
       ),
   ))
   out = sentence_embedder(token_embeddings, attention_mask)
   # out["sentence_embedding"]: (batch, num_classes, 128)

Categorical Features
--------------------

CategoricalVariableNet
~~~~~~~~~~~~~~~~~~~~~~

Handles categorical features alongside text.

.. autoclass:: torchTextClassifiers.model.components.categorical_var_net.CategoricalVariableNet
   :members:
   :undoc-members:
   :show-inheritance:

CategoricalForwardType
~~~~~~~~~~~~~~~~~~~~~~

Enum for categorical feature combination strategies.

.. autoclass:: torchTextClassifiers.model.components.categorical_var_net.CategoricalForwardType
   :members:
   :undoc-members:
   :show-inheritance:

   .. attribute:: SUM_TO_TEXT

      Sum categorical embeddings, concatenate with text.

   .. attribute:: AVERAGE_AND_CONCAT

      Average categorical embeddings, concatenate with text.

   .. attribute:: CONCATENATE_ALL

      Concatenate all embeddings (text + each categorical).

Example:

.. code-block:: python

   from torchTextClassifiers.model.components import (
       CategoricalVariableNet,
       CategoricalForwardType
   )

   # 3 categorical variables with different vocab sizes
   cat_net = CategoricalVariableNet(
       vocabulary_sizes=[10, 5, 20],
       embedding_dims=[8, 4, 16],
       forward_type=CategoricalForwardType.AVERAGE_AND_CONCAT
   )

   # Forward pass
   cat_embeddings = cat_net(categorical_data)

Classification Head
-------------------

ClassificationHead
~~~~~~~~~~~~~~~~~~

Linear classification layer(s).

.. autoclass:: torchTextClassifiers.model.components.classification_head.ClassificationHead
   :members:
   :undoc-members:
   :show-inheritance:

Example:

.. code-block:: python

   from torchTextClassifiers.model.components import ClassificationHead

   # Simple linear classifier
   head = ClassificationHead(
       input_dim=128,
       num_classes=5
   )

   # Custom classifier with nested nn.Module
   import torch.nn as nn

   custom_head_module = nn.Sequential(
       nn.Linear(128, 64),
       nn.ReLU(),
       nn.Dropout(0.2),
       nn.Linear(64, 5)
   )

   head = ClassificationHead(net=custom_head_module)

Attention Mechanism
-------------------

AttentionConfig
~~~~~~~~~~~~~~~

Configuration for transformer-style self-attention.

.. autoclass:: torchTextClassifiers.model.components.attention.AttentionConfig
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Attributes

   .. attribute:: n_embd
      :type: int

      Embedding dimension.

   .. attribute:: n_head
      :type: int

      Number of attention heads.

   .. attribute:: n_layer
      :type: int

      Number of transformer blocks.

   .. attribute:: dropout
      :type: float

      Dropout rate (default: 0.0).

   .. attribute:: bias
      :type: bool

      Use bias in linear layers (default: False).

Block
~~~~~

Single transformer block with self-attention + MLP.

.. autoclass:: torchTextClassifiers.model.components.attention.Block
   :members:
   :undoc-members:
   :show-inheritance:

SelfAttentionLayer
~~~~~~~~~~~~~~~~~~

Multi-head self-attention layer.

.. autoclass:: torchTextClassifiers.model.components.attention.SelfAttentionLayer
   :members:
   :undoc-members:
   :show-inheritance:

MLP
~~~

Feed-forward network.

.. autoclass:: torchTextClassifiers.model.components.attention.MLP
   :members:
   :undoc-members:
   :show-inheritance:

Example:

.. code-block:: python

   from torchTextClassifiers.model.components import AttentionConfig, Block

   # Configure attention
   config = AttentionConfig(
       n_embd=128,
       n_head=4,
       n_layer=3,
       dropout=0.1
   )

   # Create transformer block
   block = Block(config)

   # Forward pass (requires rotary embeddings cos, sin)
   output = block(embeddings, cos, sin)

Composing Components
--------------------

Components can be composed to create custom architectures:

.. code-block:: python

   import torch
   import torch.nn as nn
   from torchTextClassifiers.model.components import (
       TokenEmbedder, TokenEmbedderConfig,
       SentenceEmbedder, SentenceEmbedderConfig,
       CategoricalVariableNet, ClassificationHead,
   )

   class CustomModel(nn.Module):
       def __init__(self):
           super().__init__()
           self.token_embedder = TokenEmbedder(TokenEmbedderConfig(
               vocab_size=5000, embedding_dim=128, padding_idx=0,
           ))
           self.sentence_embedder = SentenceEmbedder(SentenceEmbedderConfig())
           self.cat_net = CategoricalVariableNet(...)
           self.head = ClassificationHead(...)

       def forward(self, input_ids, attention_mask, categorical_data):
           token_out = self.token_embedder(input_ids, attention_mask)
           sent_out = self.sentence_embedder(
               token_out["token_embeddings"], token_out["attention_mask"]
           )
           cat_features = self.cat_net(categorical_data)
           combined = torch.cat([sent_out["sentence_embedding"], cat_features], dim=1)
           return self.head(combined)

See Also
--------

* :doc:`model` - How components are used in models
* :doc:`../architecture/overview` - Architecture explanation
* :doc:`configs` - ModelConfig for component configuration
