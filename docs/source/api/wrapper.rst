torchTextClassifiers Wrapper
=============================

The main wrapper class for text classification tasks.

.. currentmodule:: torchTextClassifiers.torchTextClassifiers

Main Class
----------

.. autoclass:: torchTextClassifiers
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :special-members: __init__

   .. rubric:: Methods

   .. autosummary::
      :nosignatures:

      ~torchTextClassifiers.train
      ~torchTextClassifiers.predict
      ~torchTextClassifiers.predict_proba
      ~torchTextClassifiers.get_explanations
      ~torchTextClassifiers.save
      ~torchTextClassifiers.load
      ~torchTextClassifiers.save_pretrained
      ~torchTextClassifiers.from_pretrained
      ~torchTextClassifiers.push_to_hub

Usage Example
-------------

.. code-block:: python

   from torchTextClassifiers import torchTextClassifiers, ModelConfig, TrainingConfig
   from torchTextClassifiers.tokenizers import WordPieceTokenizer

   # Create tokenizer
   tokenizer = WordPieceTokenizer()
   tokenizer.train(texts, vocab_size=1000)

   # Configure model
   model_config = ModelConfig(embedding_dim=64, num_classes=2)
   training_config = TrainingConfig(num_epochs=10, batch_size=16, lr=1e-3)

   # Create and train classifier
   classifier = torchTextClassifiers(tokenizer=tokenizer, model_config=model_config)
   classifier.train(X_text=texts, y=labels, training_config=training_config)

   # Make predictions
   predictions = classifier.predict(new_texts)
   probabilities = classifier.predict_proba(new_texts)

Hugging Face Hub
----------------

With the ``hub`` extra (``pip install "torchTextClassifiers[hub]"``), a classifier can be saved
and shared using ``save_pretrained``, ``from_pretrained`` and ``push_to_hub``. Files are written
as ``model.safetensors`` plus JSON (no pickle). Currently only models built from a
``ModelConfig`` with an ``NGramTokenizer`` are supported. See :doc:`../tutorials/hub_sharing`.

.. code-block:: python

   classifier.save_pretrained("my_model/")
   classifier.push_to_hub("my-org/my-model", model_card_kwargs={"license": "mit"})
   classifier = torchTextClassifiers.from_pretrained("my-org/my-model")

See Also
--------

* :doc:`configs` - Configuration classes
* :doc:`tokenizers` - Tokenizer options
* :doc:`model` - Underlying PyTorch models
