Tokenizers
==========

Text tokenization classes for converting text to numerical tokens.

.. currentmodule:: torchTextClassifiers.tokenizers

Base Classes
------------

BaseTokenizer
~~~~~~~~~~~~~

Abstract base class for all tokenizers.

.. autoclass:: torchTextClassifiers.tokenizers.base.BaseTokenizer
   :members:
   :undoc-members:
   :show-inheritance:

TokenizerOutput
~~~~~~~~~~~~~~~

Output dataclass from tokenization.

.. autoclass:: torchTextClassifiers.tokenizers.base.TokenizerOutput
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Attributes

   .. attribute:: input_ids
      :type: torch.Tensor

      Token indices (batch_size, seq_len).

   .. attribute:: attention_mask
      :type: torch.Tensor

      Attention mask tensor (batch_size, seq_len).

   .. attribute:: offset_mapping
      :type: Optional[List[List[Tuple[int, int]]]]

      Byte offsets for each token (optional, for explainability).

   .. attribute:: word_ids
      :type: Optional[List[List[Optional[int]]]]

      Word-level indices for each token (optional).

Concrete Tokenizers
-------------------

NGramTokenizer
~~~~~~~~~~~~~~

FastText-style character n-gram tokenizer.

.. autoclass:: torchTextClassifiers.tokenizers.ngram.NGramTokenizer
   :members:
   :undoc-members:
   :show-inheritance:

   **Features:**

   - Character n-gram generation (customizable min/max n)
   - Subword caching for performance
   - Text cleaning and normalization (FastText style)
   - Hash-based tokenization
   - Support for special tokens, padding, truncation

Example:

.. code-block:: python

   from torchTextClassifiers.tokenizers import NGramTokenizer

   # Create tokenizer
   tokenizer = NGramTokenizer(
       vocab_size=10000,
       min_n=3,  # Minimum n-gram size
       max_n=6,  # Maximum n-gram size
       output_dim=128
   )

   # Train on corpus
   tokenizer.train(training_texts)

   # Tokenize
   output = tokenizer(["Hello world!", "Text classification"])

WordPieceTokenizer
~~~~~~~~~~~~~~~~~~

WordPiece subword tokenization.

.. autoclass:: torchTextClassifiers.tokenizers.WordPiece.WordPieceTokenizer
   :members:
   :undoc-members:
   :show-inheritance:

   **Features:**

   - Subword tokenization strategy
   - Vocabulary learning from corpus
   - Handles unknown words gracefully
   - Efficient encoding/decoding

Example:

.. code-block:: python

   from torchTextClassifiers.tokenizers import WordPieceTokenizer

   # Create tokenizer
   tokenizer = WordPieceTokenizer(
       vocab_size=5000,
       output_dim=128
   )

   # Train on corpus
   tokenizer.train(training_texts)

   # Tokenize
   output = tokenizer(["Hello world!", "Text classification"])

HuggingFaceTokenizer
~~~~~~~~~~~~~~~~~~~~

Wrapper for HuggingFace tokenizers.

.. autoclass:: torchTextClassifiers.tokenizers.base.HuggingFaceTokenizer
   :members:
   :undoc-members:
   :show-inheritance:

   **Features:**

   - Access to HuggingFace pre-trained tokenizers
   - Compatible with transformer models
   - Support for special tokens

Example:

.. code-block:: python

   from torchTextClassifiers.tokenizers import HuggingFaceTokenizer
   from transformers import AutoTokenizer

   # Load pre-trained tokenizer
   hf_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

   # Wrap in our interface
   tokenizer = HuggingFaceTokenizer(
       tokenizer=hf_tokenizer,
       output_dim=128
   )

   # Tokenize
   output = tokenizer(["Hello world!", "Text classification"])

Choosing a Tokenizer
---------------------

**NGramTokenizer (FastText-style)**

Use when:

* You want character-level features
* Your text has many misspellings or variations
* You need fast training
* You have limited vocabulary

**WordPieceTokenizer**

Use when:

* You want subword-level features
* Your vocabulary is large but manageable
* You need good coverage with reasonable vocab size
* You're doing standard text classification

**HuggingFaceTokenizer**

Use when:

* You want to use pre-trained tokenizers
* You're working with transformer models
* You need specific language support
* You want to fine-tune on top of BERT/RoBERTa/etc.

Tokenizer Comparison
--------------------

.. list-table::
   :widths: 25 25 25 25
   :header-rows: 1

   * - Feature
     - NGramTokenizer
     - WordPieceTokenizer
     - HuggingFaceTokenizer
   * - Granularity
     - Character n-grams
     - Subwords
     - Subwords/Words
   * - Training Speed
     - Fast
     - Medium
     - Pre-trained
   * - Vocab Size
     - Configurable
     - Configurable
     - Pre-defined
   * - OOV Handling
     - Excellent (char-level)
     - Good (subwords)
     - Good (subwords)
   * - Memory
     - Efficient
     - Medium
     - Larger

See Also
--------

* :doc:`wrapper` - Using tokenizers with the wrapper
* :doc:`dataset` - How tokenizers are used in datasets
* :doc:`../tutorials/basic_classification` - Tokenizer tutorial
