Configuration Classes
=====================

Configuration dataclasses for model and training setup.

.. currentmodule:: torchTextClassifiers.torchTextClassifiers

ModelConfig
-----------

Configuration for model architecture.

.. autoclass:: ModelConfig
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Attributes

   .. attribute:: embedding_dim
      :type: int

      Dimension of text embeddings.

   .. attribute:: categorical_vocabulary_sizes
      :type: Optional[List[int]]

      Vocabulary sizes for categorical variables (optional).

   .. attribute:: categorical_embedding_dims
      :type: Optional[Union[List[int], int]]

      Embedding dimensions for categorical variables (optional).

   .. attribute:: num_classes
      :type: Optional[int]

      Number of output classes (optional, inferred from data if not provided).

   .. attribute:: attention_config
      :type: Optional[AttentionConfig]

      Configuration for attention mechanism (optional).

Example
~~~~~~~

.. code-block:: python

   from torchTextClassifiers import ModelConfig
   from torchTextClassifiers.model.components import AttentionConfig

   # Simple configuration
   config = ModelConfig(
       embedding_dim=128,
       num_classes=3
   )

   # With categorical features
   config = ModelConfig(
       embedding_dim=128,
       num_classes=5,
       categorical_vocabulary_sizes=[10, 20, 5],  # 3 categorical variables
       categorical_embedding_dims=[8, 16, 4]      # Their embedding dimensions
   )

   # With attention
   attention_config = AttentionConfig(
       n_embd=128,
       n_head=4,
       n_layer=2,
       dropout=0.1
   )
   config = ModelConfig(
       embedding_dim=128,
       num_classes=2,
       attention_config=attention_config
   )

TrainingConfig
--------------

Configuration for training process.

.. autoclass:: TrainingConfig
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Attributes

   .. attribute:: num_epochs
      :type: int

      Number of training epochs.

   .. attribute:: batch_size
      :type: int

      Batch size for training.

   .. attribute:: lr
      :type: float

      Learning rate.

   .. attribute:: loss
      :type: torch.nn.Module

      Loss function (default: CrossEntropyLoss).

   .. attribute:: optimizer
      :type: Type[torch.optim.Optimizer]

      Optimizer class (default: Adam).

   .. attribute:: scheduler
      :type: Optional[Type[torch.optim.lr_scheduler._LRScheduler]]

      Learning rate scheduler class (optional). Schedulers are stepped once per
      epoch. ``ReduceLROnPlateau`` reduces the learning rate when the metric set by
      :attr:`scheduler_monitor` stops improving.

   .. attribute:: accelerator
      :type: str

      Accelerator type: "auto", "cpu", "gpu", or "mps" (default: "auto").

   .. attribute:: num_workers
      :type: Optional[int]

      Number of data loading workers. Default (``None``): number of CPUs actually
      available to the process minus one (taking cgroup CPU quotas and CPU affinity
      into account, so containers are sized correctly), capped at 8.

   .. attribute:: patience_early_stopping
      :type: int

      Early stopping patience in epochs (default: 3). Early stopping and model
      checkpointing monitor ``"val_loss"`` when a validation set is given,
      ``"train_loss"`` otherwise.

      With ``ReduceLROnPlateau``, its ``patience`` (in :attr:`scheduler_params`)
      must be lower than this value, otherwise training stops before the learning
      rate is ever reduced. A warning is logged in that case. Note that PyTorch's
      default ``patience`` for ``ReduceLROnPlateau`` is 10, which is higher than
      the default early stopping patience of 3.

   .. attribute:: dataloader_params
      :type: Optional[dict]

      Additional DataLoader parameters (optional).

   .. attribute:: trainer_params
      :type: Optional[dict]

      Additional PyTorch Lightning Trainer parameters (optional).

   .. attribute:: optimizer_params
      :type: Optional[dict]

      Additional optimizer parameters (optional).

   .. attribute:: scheduler_params
      :type: Optional[dict]

      Additional scheduler parameters (optional).

   .. attribute:: scheduler_monitor
      :type: Optional[str]

      Metric monitored by ``ReduceLROnPlateau`` (ignored by other schedulers).
      Default (``None``): the same metric as early stopping and checkpointing,
      i.e. ``"val_loss"`` when a validation set is given, ``"train_loss"`` otherwise.
      Set to ``"train_loss"`` to reduce the learning rate on the training loss even
      when a validation set is available.

Example
~~~~~~~

.. code-block:: python

   from torchTextClassifiers import TrainingConfig
   import torch.nn as nn
   import torch.optim as optim

   # Basic configuration
   config = TrainingConfig(
       num_epochs=20,
       batch_size=32,
       lr=1e-3
   )

   # Advanced configuration
   config = TrainingConfig(
       num_epochs=50,
       batch_size=64,
       lr=5e-4,
       loss=nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 1.5])),
       optimizer=optim.AdamW,
       scheduler=optim.lr_scheduler.CosineAnnealingLR,
       accelerator="gpu",
       patience_early_stopping=10,
       optimizer_params={"weight_decay": 0.01},
       scheduler_params={"T_max": 50}
   )

   # Reduce the learning rate when the validation loss plateaus
   config = TrainingConfig(
       num_epochs=50,
       batch_size=64,
       lr=1e-3,
       scheduler=optim.lr_scheduler.ReduceLROnPlateau,
       scheduler_params={"patience": 2, "factor": 0.5},  # patience < early stopping
       patience_early_stopping=5,
       # scheduler_monitor="train_loss",  # uncomment to watch the training loss instead
   )

See Also
--------

* :doc:`wrapper` - Using configurations with the wrapper
* :doc:`components` - AttentionConfig for attention mechanism
* :doc:`model` - How configurations affect the model
