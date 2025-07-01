How to Use Custom Tokenizers
=============================

This guide shows you how to work with custom tokenizers, including creating your own tokenizers 
and using pre-trained tokenizers with torchTextClassifiers.

Understanding the Built-in NGramTokenizer
------------------------------------------

The Default Tokenizer
~~~~~~~~~~~~~~~~~~~~~~

torchTextClassifiers uses an ``NGramTokenizer`` by default, which implements FastText's tokenization strategy:

.. code-block:: python

   from torchTextClassifiers.classifiers.fasttext.tokenizer import NGramTokenizer
   import numpy as np

   # Sample training data
   training_texts = np.array([
       "This is a sample text for training",
       "Another example sentence",
       "Text classification example"
   ])

   # Create tokenizer with custom parameters
   tokenizer = NGramTokenizer(
       min_count=1,          # Minimum frequency for vocabulary inclusion
       min_n=3,              # Minimum character n-gram length
       max_n=6,              # Maximum character n-gram length
       num_tokens=10000,     # Vocabulary size (hash table size)
       len_word_ngrams=2,    # Word n-gram length
       training_text=training_texts.tolist()
   )

   # Tokenize new text
   new_texts = ["Hello world example"]
   tokens, indices, id_to_token, token_to_id = tokenizer.tokenize(new_texts)
   
   print(f"Tokens: {tokens[0]}")  # List of token strings
   print(f"Indices: {indices[0]}")  # Corresponding token IDs

Tokenizer Configuration Options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Character n-gram settings
   tokenizer = NGramTokenizer(
       min_n=2,              # Shorter n-grams for broader matching
       max_n=8,              # Longer n-grams for more specificity (max 6)
       min_count=5,          # Higher threshold filters rare tokens
       num_tokens=50000,     # Larger vocabulary for complex domains
       len_word_ngrams=3,    # Capture longer phrases
       training_text=training_data
   )

   # For different use cases:
   
   # Memory-efficient configuration
   efficient_config = {
       'min_n': 3,
       'max_n': 5,
       'min_count': 10,
       'num_tokens': 20000,
       'len_word_ngrams': 2
   }
   
   # High-precision configuration  
   precision_config = {
       'min_n': 2,
       'max_n': 6,
       'min_count': 1,
       'num_tokens': 100000,
       'len_word_ngrams': 3
   }

Building FastText from Pre-trained Tokenizer
---------------------------------------------

Using Existing Tokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~

If you have a pre-trained tokenizer, you can build a FastText classifier directly from it:

.. code-block:: python

   from torchTextClassifiers import build_fasttext_from_tokenizer

   # Assume you have a pre-trained tokenizer
   # (This could be saved from a previous training session)
   
   # Create classifier from existing tokenizer
   classifier = build_fasttext_from_tokenizer(
       tokenizer=existing_tokenizer,
       embedding_dim=100,
       num_classes=3,
       sparse=False
   )

   # The tokenizer is already built, so you can start training immediately
   classifier.train(X_train, y_train, X_val, y_val, num_epochs=20)

Saving and Loading Tokenizers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import json
   import pickle

   # Method 1: Save tokenizer configuration (for NGramTokenizer)
   def save_tokenizer_config(tokenizer, filepath):
       """Save tokenizer configuration to JSON"""
       config = {
           'min_count': tokenizer.min_count,
           'min_n': tokenizer.min_n,
           'max_n': tokenizer.max_n,
           'num_tokens': tokenizer.num_tokens,
           'len_word_ngrams': tokenizer.word_ngrams,
           'word_id_mapping': tokenizer.word_id_mapping,
           'nwords': tokenizer.nwords
       }
       
       with open(filepath, 'w') as f:
           json.dump(config, f, indent=2)

   def load_tokenizer_config(filepath, training_text):
       """Load tokenizer configuration from JSON"""
       with open(filepath, 'r') as f:
           config = json.load(f)
       
       # Recreate tokenizer (you'll need the original training text)
       return NGramTokenizer(
           min_count=config['min_count'],
           min_n=config['min_n'],
           max_n=config['max_n'],
           num_tokens=config['num_tokens'],
           len_word_ngrams=config['len_word_ngrams'],
           training_text=training_text
       )

   # Method 2: Save complete tokenizer object
   def save_complete_tokenizer(tokenizer, filepath):
       """Save complete tokenizer object"""
       with open(filepath, 'wb') as f:
           pickle.dump(tokenizer, f)

   def load_complete_tokenizer(filepath):
       """Load complete tokenizer object"""
       with open(filepath, 'rb') as f:
           return pickle.load(f)

   # Usage examples
   save_tokenizer_config(tokenizer, 'tokenizer_config.json')
   save_complete_tokenizer(tokenizer, 'tokenizer_complete.pkl')

   # Load later
   loaded_tokenizer = load_complete_tokenizer('tokenizer_complete.pkl')

Creating Custom Tokenizers
---------------------------

Custom Tokenizer Interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To create a custom tokenizer, implement the following interface:

.. code-block:: python

   from abc import ABC, abstractmethod
   import torch
   from typing import List, Dict, Tuple

   class BaseTokenizer(ABC):
       """Base class for custom tokenizers"""
       
       def __init__(self, training_text: List[str]):
           self.training_text = training_text
           self.vocab_size = 0
           self.padding_index = 0
       
       @abstractmethod
       def tokenize(self, texts: List[str]) -> Tuple[List[List[str]], torch.Tensor, Dict, Dict]:
           """
           Tokenize input texts
           
           Args:
               texts: List of input texts
               
           Returns:
               Tuple of (token_strings, token_indices, id_to_token, token_to_id)
           """
           pass
       
       @abstractmethod
       def get_vocab_size(self) -> int:
           """Return vocabulary size"""
           pass
       
       @abstractmethod
       def get_padding_index(self) -> int:
           """Return padding index"""
           pass

Simple Word-Level Tokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from collections import Counter
   import torch

   class SimpleWordTokenizer(BaseTokenizer):
       """Simple word-level tokenizer"""
       
       def __init__(self, training_text: List[str], min_count: int = 1, max_vocab: int = 10000):
           super().__init__(training_text)
           self.min_count = min_count
           self.max_vocab = max_vocab
           self.word_to_id = {}
           self.id_to_word = {}
           self.padding_index = 0
           
           self._build_vocab()
       
       def _build_vocab(self):
           """Build vocabulary from training text"""
           # Count word frequencies
           word_counts = Counter()
           for text in self.training_text:
               words = text.lower().split()
               word_counts.update(words)
           
           # Build vocabulary
           self.word_to_id = {'<PAD>': 0, '<UNK>': 1}
           vocab_id = 2
           
           # Add words above minimum count threshold
           for word, count in word_counts.most_common():
               if count >= self.min_count and vocab_id < self.max_vocab:
                   self.word_to_id[word] = vocab_id
                   vocab_id += 1
           
           # Create reverse mapping
           self.id_to_word = {v: k for k, v in self.word_to_id.items()}
           self.vocab_size = len(self.word_to_id)
       
       def tokenize(self, texts: List[str]) -> Tuple[List[List[str]], torch.Tensor, Dict, Dict]:
           """Tokenize input texts"""
           token_strings = []
           token_indices_list = []
           
           for text in texts:
               words = text.lower().split()
               tokens = []
               indices = []
               
               for word in words:
                   if word in self.word_to_id:
                       tokens.append(word)
                       indices.append(self.word_to_id[word])
                   else:
                       tokens.append('<UNK>')
                       indices.append(self.word_to_id['<UNK>'])
               
               token_strings.append(tokens)
               token_indices_list.append(torch.tensor(indices, dtype=torch.long))
           
           return token_strings, token_indices_list, self.id_to_word, self.word_to_id
       
       def get_vocab_size(self) -> int:
           return self.vocab_size
       
       def get_padding_index(self) -> int:
           return self.padding_index

   # Usage
   simple_tokenizer = SimpleWordTokenizer(
       training_text=["hello world", "world peace", "hello there"],
       min_count=1,
       max_vocab=1000
   )
   
   tokens, indices, id_to_word, word_to_id = simple_tokenizer.tokenize(["hello world"])
   print(f"Tokens: {tokens[0]}")
   print(f"Indices: {indices[0]}")

Subword Tokenizer with SentencePiece
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   try:
       import sentencepiece as spm
       SENTENCEPIECE_AVAILABLE = True
   except ImportError:
       SENTENCEPIECE_AVAILABLE = False
       print("SentencePiece not available. Install with: pip install sentencepiece")

   if SENTENCEPIECE_AVAILABLE:
       class SentencePieceTokenizer(BaseTokenizer):
           """SentencePiece-based tokenizer"""
           
           def __init__(self, training_text: List[str], vocab_size: int = 8000, model_type: str = 'bpe'):
               super().__init__(training_text)
               self.vocab_size = vocab_size
               self.model_type = model_type
               self.sp = spm.SentencePieceProcessor()
               self._train_sentencepiece()
           
           def _train_sentencepiece(self):
               """Train SentencePiece model"""
               import tempfile
               import os
               
               # Write training data to temporary file
               with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
                   for text in self.training_text:
                       f.write(text + '\n')
                   temp_file = f.name
               
               # Train SentencePiece model
               model_prefix = tempfile.mktemp()
               spm.SentencePieceTrainer.train(
                   input=temp_file,
                   model_prefix=model_prefix,
                   vocab_size=self.vocab_size,
                   model_type=self.model_type,
                   character_coverage=0.9995,
                   input_sentence_size=len(self.training_text),
                   shuffle_input_sentence=True
               )
               
               # Load trained model
               self.sp.load(f'{model_prefix}.model')
               self.padding_index = self.sp.pad_id()
               
               # Cleanup
               os.unlink(temp_file)
               os.unlink(f'{model_prefix}.model')
               os.unlink(f'{model_prefix}.vocab')
           
           def tokenize(self, texts: List[str]) -> Tuple[List[List[str]], torch.Tensor, Dict, Dict]:
               """Tokenize using SentencePiece"""
               token_strings = []
               token_indices_list = []
               
               for text in texts:
                   # Tokenize with SentencePiece
                   tokens = self.sp.encode_as_pieces(text)
                   indices = self.sp.encode_as_ids(text)
                   
                   token_strings.append(tokens)
                   token_indices_list.append(torch.tensor(indices, dtype=torch.long))
               
               # Create mappings
               id_to_token = {i: self.sp.decode_ids([i]) for i in range(self.sp.get_piece_size())}
               token_to_id = {v: k for k, v in id_to_token.items()}
               
               return token_strings, token_indices_list, id_to_token, token_to_id
           
           def get_vocab_size(self) -> int:
               return self.sp.get_piece_size()
           
           def get_padding_index(self) -> int:
               return self.padding_index

       # Usage
       sp_tokenizer = SentencePieceTokenizer(
           training_text=large_training_corpus,
           vocab_size=8000,
           model_type='bpe'  # or 'unigram'
       )

Using Hugging Face Tokenizers
-----------------------------

Integration with Transformers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   try:
       from transformers import AutoTokenizer
       TRANSFORMERS_AVAILABLE = True
   except ImportError:
       TRANSFORMERS_AVAILABLE = False
       print("Transformers not available. Install with: pip install transformers")

   if TRANSFORMERS_AVAILABLE:
       class HuggingFaceTokenizer(BaseTokenizer):
           """Wrapper for Hugging Face tokenizers"""
           
           def __init__(self, model_name: str = 'bert-base-uncased', max_length: int = 512):
               self.tokenizer = AutoTokenizer.from_pretrained(model_name)
               self.max_length = max_length
               self.padding_index = self.tokenizer.pad_token_id
           
           def tokenize(self, texts: List[str]) -> Tuple[List[List[str]], torch.Tensor, Dict, Dict]:
               """Tokenize using Hugging Face tokenizer"""
               token_strings = []
               token_indices_list = []
               
               for text in texts:
                   # Tokenize with HF tokenizer
                   encoded = self.tokenizer(
                       text,
                       padding='max_length',
                       truncation=True,
                       max_length=self.max_length,
                       return_tensors='pt'
                   )
                   
                   # Get tokens and indices
                   tokens = self.tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])
                   indices = encoded['input_ids'][0]
                   
                   token_strings.append(tokens)
                   token_indices_list.append(indices)
               
               # Create mappings
               vocab = self.tokenizer.get_vocab()
               token_to_id = vocab
               id_to_token = {v: k for k, v in vocab.items()}
               
               return token_strings, token_indices_list, id_to_token, token_to_id
           
           def get_vocab_size(self) -> int:
               return self.tokenizer.vocab_size
           
           def get_padding_index(self) -> int:
               return self.padding_index

       # Usage
       hf_tokenizer = HuggingFaceTokenizer(
           model_name='distilbert-base-uncased',
           max_length=256
       )

Advanced Tokenizer Customization
---------------------------------

Preprocessing Pipeline
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import re
   from typing import Callable

   class PreprocessingTokenizer(NGramTokenizer):
       """NGramTokenizer with custom preprocessing"""
       
       def __init__(self, preprocessing_fn: Callable[[str], str] = None, **kwargs):
           self.preprocessing_fn = preprocessing_fn or self._default_preprocessing
           super().__init__(**kwargs)
       
       def _default_preprocessing(self, text: str) -> str:
           """Default preprocessing pipeline"""
           # Convert to lowercase
           text = text.lower()
           
           # Remove extra whitespace
           text = re.sub(r'\s+', ' ', text)
           
           # Remove special characters (keep basic punctuation)
           text = re.sub(r'[^\w\s.,!?-]', '', text)
           
           # Strip leading/trailing whitespace
           text = text.strip()
           
           return text
       
       def tokenize(self, texts: List[str], **kwargs):
           """Tokenize with preprocessing"""
           # Apply preprocessing
           preprocessed_texts = [self.preprocessing_fn(text) for text in texts]
           
           # Call parent tokenize method
           return super().tokenize(preprocessed_texts, **kwargs)

   # Custom preprocessing function
   def domain_specific_preprocessing(text: str) -> str:
       """Preprocessing for specific domain (e.g., social media)"""
       # Remove URLs
       text = re.sub(r'http\S+|www\S+', '<URL>', text)
       
       # Replace @mentions with placeholder
       text = re.sub(r'@\w+', '<MENTION>', text)
       
       # Replace hashtags
       text = re.sub(r'#(\w+)', r'\1', text)
       
       # Normalize repeated characters
       text = re.sub(r'(.)\1{2,}', r'\1\1', text)
       
       return text

   # Usage
   custom_tokenizer = PreprocessingTokenizer(
       preprocessing_fn=domain_specific_preprocessing,
       min_count=2,
       min_n=3,
       max_n=6,
       num_tokens=10000,
       len_word_ngrams=2,
       training_text=training_texts
   )

Multi-Language Support
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class MultiLanguageTokenizer(NGramTokenizer):
       """Tokenizer with multi-language support"""
       
       def __init__(self, languages: List[str] = None, **kwargs):
           self.languages = languages or ['en']
           super().__init__(**kwargs)
       
       def _language_specific_preprocessing(self, text: str, language: str) -> str:
           """Apply language-specific preprocessing"""
           if language == 'en':
               # English-specific preprocessing
               return text.lower()
           elif language == 'de':
               # German-specific preprocessing (handle umlauts, etc.)
               replacements = {'ä': 'ae', 'ö': 'oe', 'ü': 'ue', 'ß': 'ss'}
               for old, new in replacements.items():
                   text = text.replace(old, new)
               return text.lower()
           elif language == 'fr':
               # French-specific preprocessing
               text = text.lower()
               # Remove accents (simplified example)
               text = re.sub(r'[àáâãäå]', 'a', text)
               text = re.sub(r'[èéêë]', 'e', text)
               return text
           else:
               return text.lower()
       
       def tokenize(self, texts: List[str], language: str = 'en', **kwargs):
           """Tokenize with language-specific preprocessing"""
           if language not in self.languages:
               print(f"Warning: Language '{language}' not in trained languages {self.languages}")
           
           # Apply language-specific preprocessing
           preprocessed_texts = [
               self._language_specific_preprocessing(text, language) 
               for text in texts
           ]
           
           return super().tokenize(preprocessed_texts, **kwargs)

Performance Considerations
--------------------------

Tokenizer Caching
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from functools import lru_cache

   class CachedTokenizer:
       """Tokenizer with caching for repeated inputs"""
       
       def __init__(self, base_tokenizer, cache_size=1000):
           self.tokenizer = base_tokenizer
           self.cache_size = cache_size
           
           # Create cached version of tokenize method
           self._cached_tokenize_single = lru_cache(maxsize=cache_size)(
               self._tokenize_single
           )
       
       def _tokenize_single(self, text: str):
           """Tokenize single text (cacheable)"""
           tokens, indices, id_to_token, token_to_id = self.tokenizer.tokenize([text])
           return tokens[0], indices[0], id_to_token, token_to_id
       
       def tokenize(self, texts: List[str]):
           """Tokenize with caching"""
           results = [self._cached_tokenize_single(text) for text in texts]
           
           # Separate results
           token_strings = [r[0] for r in results]
           token_indices = [r[1] for r in results]
           id_to_token_dicts = [r[2] for r in results]
           token_to_id_dicts = [r[3] for r in results]
           
           return token_strings, token_indices, id_to_token_dicts, token_to_id_dicts
       
       def clear_cache(self):
           """Clear tokenization cache"""
           self._cached_tokenize_single.cache_clear()

Parallel Tokenization
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from multiprocessing import Pool
   from functools import partial

   def parallel_tokenize(tokenizer, texts: List[str], n_processes: int = 4):
       """Tokenize texts in parallel"""
       
       def tokenize_chunk(chunk_texts, tokenizer):
           return tokenizer.tokenize(chunk_texts)
       
       # Split texts into chunks
       chunk_size = len(texts) // n_processes + 1
       chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
       
       # Process chunks in parallel
       tokenize_func = partial(tokenize_chunk, tokenizer=tokenizer)
       
       with Pool(n_processes) as pool:
           results = pool.map(tokenize_func, chunks)
       
       # Combine results
       all_token_strings = []
       all_token_indices = []
       all_id_to_token = []
       all_token_to_id = []
       
       for token_strings, token_indices, id_to_token, token_to_id in results:
           all_token_strings.extend(token_strings)
           all_token_indices.extend(token_indices)
           all_id_to_token.extend(id_to_token)
           all_token_to_id.extend(token_to_id)
       
       return all_token_strings, all_token_indices, all_id_to_token, all_token_to_id

Best Practices
--------------

Tokenizer Selection Guidelines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Guidelines for choosing tokenizers:

   # 1. For small datasets (< 10K samples)
   # Use simple word-level tokenizer
   simple_tokenizer = SimpleWordTokenizer(training_text, min_count=1)

   # 2. For medium datasets (10K - 100K samples)  
   # Use NGramTokenizer with moderate settings
   medium_tokenizer = NGramTokenizer(
       min_count=2, min_n=3, max_n=6, 
       num_tokens=20000, len_word_ngrams=2,
       training_text=training_text
   )

   # 3. For large datasets (> 100K samples)
   # Use NGramTokenizer with aggressive filtering
   large_tokenizer = NGramTokenizer(
       min_count=5, min_n=3, max_n=5,
       num_tokens=50000, len_word_ngrams=2,
       training_text=training_text
   )

   # 4. For multilingual tasks
   # Use SentencePiece or Hugging Face tokenizers
   multilingual_tokenizer = SentencePieceTokenizer(
       training_text=multilingual_corpus,
       vocab_size=32000,
       model_type='unigram'
   )

.. note::
   When creating custom tokenizers, ensure they implement the required interface 
   and handle edge cases like empty texts and unknown tokens properly.

.. tip::
   Profile your tokenizer performance with large datasets and consider using 
   caching or parallel processing for production use cases.