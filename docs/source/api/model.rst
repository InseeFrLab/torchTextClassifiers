Core Models
===========

Core PyTorch and PyTorch Lightning models.

.. currentmodule:: torchTextClassifiers.model

PyTorch Model
-------------

TextClassificationModel
~~~~~~~~~~~~~~~~~~~~~~~

Core PyTorch nn.Module combining all components.

.. autoclass:: torchTextClassifiers.model.model.TextClassificationModel
   :members:
   :undoc-members:
   :show-inheritance:

   **Architecture:**

   The model combines four main components:

   1. **TokenEmbedder**: Maps each token to a dense vector (with optional self-attention)
   2. **SentenceEmbedder**: Aggregates token vectors into a sentence representation
   3. **CategoricalVariableNet** (optional): Handles categorical features
   4. **ClassificationHead**: Produces class logits

Example:

.. code-block:: python

   from torchTextClassifiers.model import TextClassificationModel
   from torchTextClassifiers.model.components import (
       TokenEmbedder, TokenEmbedderConfig,
       SentenceEmbedder, SentenceEmbedderConfig,
       CategoricalVariableNet,
       ClassificationHead,
   )

   # Create components
   token_embedder = TokenEmbedder(TokenEmbedderConfig(
       vocab_size=5000,
       embedding_dim=128,
       padding_idx=0,
   ))
   sentence_embedder = SentenceEmbedder(SentenceEmbedderConfig(aggregation_method="mean"))

   cat_net = CategoricalVariableNet(
       categorical_vocabulary_sizes=[10, 20],
       categorical_embedding_dims=[8, 16],
   )

   classification_head = ClassificationHead(
       input_dim=128 + 24,  # text_dim + cat_dim
       num_classes=5,
   )

   # Combine into model
   model = TextClassificationModel(
       token_embedder=token_embedder,
       sentence_embedder=sentence_embedder,
       categorical_variable_net=cat_net,
       classification_head=classification_head,
   )

   # Forward pass
   logits = model(input_ids, attention_mask, categorical_data)

PyTorch Lightning Module
-------------------------

TextClassificationModule
~~~~~~~~~~~~~~~~~~~~~~~~

PyTorch Lightning LightningModule for automated training.

.. autoclass:: torchTextClassifiers.model.lightning.TextClassificationModule
   :members:
   :undoc-members:
   :show-inheritance:

   **Features:**

   - Automated training/validation/test steps
   - Metrics tracking (accuracy)
   - Optimizer and scheduler management
   - Logging integration
   - PyTorch Lightning callbacks support

Example:

.. code-block:: python

   from torchTextClassifiers.model import (
       TextClassificationModel,
       TextClassificationModule
   )
   import torch.nn as nn
   import torch.optim as optim
   from pytorch_lightning import Trainer

   # Create PyTorch model
   model = TextClassificationModel(...)

   # Wrap in Lightning module
   lightning_module = TextClassificationModule(
       model=model,
       loss=nn.CrossEntropyLoss(),
       optimizer=optim.Adam,
       lr=1e-3,
       scheduler=optim.lr_scheduler.StepLR,
       scheduler_params={"step_size": 10, "gamma": 0.1}
   )

   # Train with Lightning Trainer
   trainer = Trainer(
       max_epochs=20,
       accelerator="auto",
       devices=1
   )

   trainer.fit(
       lightning_module,
       train_dataloaders=train_dataloader,
       val_dataloaders=val_dataloader
   )

   # Test
   trainer.test(lightning_module, dataloaders=test_dataloader)

Training Steps
--------------

The TextClassificationModule implements standard training/validation/test steps:

**Training Step:**

.. code-block:: python

   def training_step(self, batch, batch_idx):
       input_ids, cat_features, labels = batch
       logits = self.model(input_ids, cat_features)
       loss = self.loss(logits, labels)
       acc = self.compute_accuracy(logits, labels)

       self.log("train_loss", loss)
       self.log("train_acc", acc)

       return loss

**Validation Step:**

.. code-block:: python

   def validation_step(self, batch, batch_idx):
       input_ids, cat_features, labels = batch
       logits = self.model(input_ids, cat_features)
       loss = self.loss(logits, labels)
       acc = self.compute_accuracy(logits, labels)

       self.log("val_loss", loss)
       self.log("val_acc", acc)

Custom Training
---------------

For custom training loops, use the PyTorch model directly:

.. code-block:: python

   from torchTextClassifiers.model import TextClassificationModel
   import torch.nn as nn
   import torch.optim as optim

   model = TextClassificationModel(...)
   loss_fn = nn.CrossEntropyLoss()
   optimizer = optim.Adam(model.parameters(), lr=1e-3)

   # Custom training loop
   for epoch in range(num_epochs):
       for batch in dataloader:
           input_ids, cat_features, labels = batch

           # Forward pass
           logits = model(input_ids, cat_features)
           loss = loss_fn(logits, labels)

           # Backward pass
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()

           print(f"Epoch {epoch}, Loss: {loss.item()}")

See Also
--------

* :doc:`components` - Model components
* :doc:`wrapper` - High-level wrapper using these models
* :doc:`dataset` - Data loading for models
* :doc:`configs` - Model and training configuration
