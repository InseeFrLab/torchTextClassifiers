Model Components
================

Modular torch.nn.Module components for building custom architectures.

.. currentmodule:: torchTextClassifiers.model.components

Text Embedding
--------------

TextEmbedder
~~~~~~~~~~~~

Embeds text tokens with optional self-attention.

.. autoclass:: torchTextClassifiers.model.components.text_embedder.TextEmbedder
   :members:
   :undoc-members:
   :show-inheritance:

TextEmbedderConfig
~~~~~~~~~~~~~~~~~~

Configuration for TextEmbedder.

.. autoclass:: torchTextClassifiers.model.components.text_embedder.TextEmbedderConfig
   :members:
   :undoc-members:
   :show-inheritance:

Example:

.. code-block:: python

   from torchTextClassifiers.model.components import TextEmbedder, TextEmbedderConfig

   # Simple text embedder
   config = TextEmbedderConfig(
       vocab_size=5000,
       embedding_dim=128,
       attention_config=None
   )
   embedder = TextEmbedder(config)

   # With self-attention
   from torchTextClassifiers.model.components import AttentionConfig

   attention_config = AttentionConfig(
       n_embd=128,
       n_head=4,
       n_layer=2,
       dropout=0.1
   )
   config = TextEmbedderConfig(
       vocab_size=5000,
       embedding_dim=128,
       attention_config=attention_config
   )
   embedder = TextEmbedder(config)

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

   import torch.nn as nn
   from torchTextClassifiers.model.components import (
       TextEmbedder, CategoricalVariableNet, ClassificationHead
   )

   class CustomModel(nn.Module):
       def __init__(self):
           super().__init__()
           self.text_embedder = TextEmbedder(text_config)
           self.cat_net = CategoricalVariableNet(...)
           self.head = ClassificationHead(...)

       def forward(self, input_ids, categorical_data):
           text_features = self.text_embedder(input_ids)
           cat_features = self.cat_net(categorical_data)
           combined = torch.cat([text_features, cat_features], dim=1)
           return self.head(combined)

See Also
--------

* :doc:`model` - How components are used in models
* :doc:`../architecture/overview` - Architecture explanation
* :doc:`configs` - ModelConfig for component configuration

