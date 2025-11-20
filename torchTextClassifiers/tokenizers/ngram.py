import json
import re
import unicodedata
from functools import lru_cache
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from torchTextClassifiers.tokenizers import BaseTokenizer, TokenizerOutput

# ============================================================================
#                        Optimized normalization
# ============================================================================

_fasttext_non_alnum = re.compile(r"[^a-z0-9]+")
_fasttext_multi_space = re.compile(r"\s+")

# Pre-compile translation table for faster character removal
_COMBINING_MARKS = {c: None for c in range(0x0300, 0x0370)}


@lru_cache(maxsize=10000)
def _clean_single_text_cached(text: str) -> str:
    """Cached version of text cleaning - major speedup for repeated texts."""
    t = text.lower()
    t = unicodedata.normalize("NFKD", t)
    # Faster: use translate() instead of list comprehension
    t = t.translate(_COMBINING_MARKS)
    t = _fasttext_non_alnum.sub(" ", t)
    t = _fasttext_multi_space.sub(" ", t)
    return t.strip()


def clean_text_feature(texts: List[str]) -> List[str]:
    """Vectorized text cleaning with caching."""
    return [_clean_single_text_cached(t) for t in texts]


# ============================================================================
#                   Optimized hash function
# ============================================================================


def fast_hash(s: str) -> int:
    """FNV-1a hash - simple and fast."""
    h = 2166136261
    for c in s:
        h ^= ord(c)
        h = (h * 16777619) & 0xFFFFFFFF
    return h


# ============================================================================
#                   Pre-computed subword cache
# ============================================================================


class SubwordCache:
    """Aggressive pre-computation cache for subwords."""

    def __init__(
        self,
        word_to_id: dict,
        min_n: int,
        max_n: int,
        num_tokens: int,
        nwords: int,
        unk_token_id: int,
    ):
        self.cache = {}
        self.word_to_id = word_to_id
        self.min_n = min_n
        self.max_n = max_n
        self.num_tokens = num_tokens
        self.nwords = nwords
        self.unk_token_id = unk_token_id

        # Pre-compute for all vocabulary words
        self._precompute_vocab()

    def _precompute_vocab(self):
        """Pre-compute subwords for entire vocabulary."""
        for word, word_id in self.word_to_id.items():
            self.cache[word] = self._compute_subwords(word, word_id)

    def _compute_subwords(self, word: str, word_id: Optional[int] = None) -> List[int]:
        """Compute subword indices for a word."""
        indices = []

        # Add word token if in vocab
        if word_id is not None:
            indices.append(word_id)

        # Extract character n-grams
        word_tagged = f"<{word}>"
        L = len(word_tagged)

        for n in range(self.min_n, self.max_n + 1):
            for i in range(L - n + 1):
                ngram = word_tagged[i : i + n]
                if ngram != word and ngram != word_tagged:
                    bucket_idx = fast_hash(ngram) % self.num_tokens
                    indices.append(3 + self.nwords + bucket_idx)

        return indices if indices else [self.unk_token_id]

    def get(self, word: str) -> List[int]:
        """Get subwords with on-demand computation for OOV words."""
        if word not in self.cache:
            word_id = self.word_to_id.get(word)
            self.cache[word] = self._compute_subwords(word, word_id)
        return self.cache[word]


# ============================================================================
#                   Vectorized encoding with optional metadata
# ============================================================================


def encode_batch_vectorized(
    sentences: List[str],
    subword_cache: SubwordCache,
    eos_token_id: int,
    pad_token_id: int,
    max_length: Optional[int] = None,
    truncation: bool = False,
    return_offsets_mapping: bool = False,
    return_word_ids: bool = False,
    force_max_length: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[List], Optional[List]]:
    """
    Vectorized batch encoding - processes all sentences together.
    Returns padded tensors directly, with optional offset mappings and word IDs.

    Args:
        force_max_length: If True and max_length is set, always return tensors of size max_length
    """
    all_ids = []
    all_offsets = [] if return_offsets_mapping else None
    all_word_ids = [] if return_word_ids else None
    max_len = 0

    # First pass: encode all sentences
    for sentence in sentences:
        ids = []
        offsets = [] if return_offsets_mapping else None
        word_ids = [] if return_word_ids else None

        words = sentence.split()
        char_offset = 0

        for word_idx, word in enumerate(words):
            # Find the actual position of this word in the original sentence
            word_start = sentence.find(word, char_offset)
            word_end = word_start + len(word)
            char_offset = word_end

            # Get subword tokens for this word
            subword_tokens = subword_cache.get(word)

            for token_id in subword_tokens:
                ids.append(token_id)

                if return_offsets_mapping:
                    # All subword tokens of a word map to the word's character span
                    offsets.append((word_start, word_end))

                if return_word_ids:
                    # All subword tokens of a word get the same word_id
                    word_ids.append(word_idx)

        # Add EOS token
        ids.append(eos_token_id)
        if return_offsets_mapping:
            offsets.append((len(sentence), len(sentence)))  # EOS has no span
        if return_word_ids:
            word_ids.append(None)  # EOS is not part of any word

        # Truncate if needed
        if truncation and max_length and len(ids) > max_length:
            ids = ids[:max_length]
            if return_offsets_mapping:
                offsets = offsets[:max_length]
            if return_word_ids:
                word_ids = word_ids[:max_length]

        all_ids.append(ids)
        if return_offsets_mapping:
            all_offsets.append(offsets)
        if return_word_ids:
            all_word_ids.append(word_ids)
        max_len = max(max_len, len(ids))

    # Determine final sequence length
    if force_max_length and max_length:
        # Always use max_length when force_max_length is True
        seq_len = max_length
    elif max_length and not truncation:
        seq_len = min(max_len, max_length)
    elif max_length:
        seq_len = max_length
    else:
        seq_len = max_len

    # Pre-allocate tensors
    batch_size = len(sentences)
    input_ids = torch.full((batch_size, seq_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, seq_len), dtype=torch.long)

    # Fill tensors and pad metadata
    for i, ids in enumerate(all_ids):
        length = min(len(ids), seq_len)
        input_ids[i, :length] = torch.tensor(ids[:length], dtype=torch.long)
        attention_mask[i, :length] = 1

        # Pad offsets and word_ids to match sequence length
        if return_offsets_mapping:
            # Pad with (0, 0) for padding tokens
            all_offsets[i] = all_offsets[i][:length] + [(0, 0)] * (seq_len - length)

        if return_word_ids:
            # Pad with None for padding tokens
            all_word_ids[i] = all_word_ids[i][:length] + [None] * (seq_len - length)

    return input_ids, attention_mask, all_offsets, all_word_ids


# ============================================================================
#                           NGramTokenizer - Optimized
# ============================================================================


class NGramTokenizer(BaseTokenizer):
    """
    Heavily optimized FastText N-gram tokenizer with:
    - Pre-computed subword cache for entire vocabulary
    - Vectorized batch encoding
    - Cached text normalization
    - Direct tensor operations
    - Optional offset mapping and word ID tracking
    """

    PAD_TOKEN = "[PAD]"
    UNK_TOKEN = "[UNK]"
    EOS_TOKEN = "</s>"

    def __init__(
        self,
        min_count: int,
        min_n: int,
        max_n: int,
        num_tokens: int,
        len_word_ngrams: int,
        training_text: Optional[List[str]] = None,
        preprocess: bool = True,
        output_dim: Optional[int] = None,
        **kwargs,
    ):
        if min_n < 2:
            raise ValueError("min_n must be >= 2")
        if max_n > 6:
            raise ValueError("max_n must be <= 6")

        self.min_count = min_count
        self.min_n = min_n
        self.max_n = max_n
        self.num_tokens = num_tokens
        self.word_ngrams = len_word_ngrams
        self.preprocess = preprocess

        self.pad_token_id = 0
        self.unk_token_id = 1
        self.eos_token_id = 2

        if training_text is not None:
            self.train(training_text)
        else:
            self.word_to_id = {}
            self.id_to_word = {}
            self.nwords = 0
            self.subword_cache = None

        self.vocab_size = 3 + self.nwords + self.num_tokens

        super().__init__(
            vocab_size=self.vocab_size, padding_idx=self.pad_token_id, output_dim=output_dim
        )

    def train(self, training_text: List[str]):
        """Build vocabulary from training text."""
        word_counts = {}
        for sent in training_text:
            for w in sent.split():
                word_counts[w] = word_counts.get(w, 0) + 1

        self.word_to_id = {}
        idx = 3
        for w, c in word_counts.items():
            if c >= self.min_count:
                self.word_to_id[w] = idx
                idx += 1

        self.nwords = len(self.word_to_id)
        self.vocab_size = 3 + self.nwords + self.num_tokens

        # Create reverse mapping
        self.id_to_word = {v: k for k, v in self.word_to_id.items()}
        self.id_to_word[self.pad_token_id] = self.PAD_TOKEN
        self.id_to_word[self.unk_token_id] = self.UNK_TOKEN
        self.id_to_word[self.eos_token_id] = self.EOS_TOKEN

        # Pre-compute all subwords for vocabulary
        print(f"Pre-computing subwords for {self.nwords} vocabulary words...")
        self.subword_cache = SubwordCache(
            self.word_to_id, self.min_n, self.max_n, self.num_tokens, self.nwords, self.unk_token_id
        )
        print("✓ Subword cache built")

    def tokenize(
        self,
        text: Union[str, List[str]],
        return_offsets_mapping: bool = False,
        return_word_ids: bool = False,
        **kwargs,
    ) -> TokenizerOutput:
        """
        Optimized tokenization with vectorized operations.

        Args:
            text: Single string or list of strings to tokenize
            padding: Padding strategy ('longest' or 'max_length')
            max_length: Maximum sequence length
            truncation: Whether to truncate sequences exceeding max_length
            return_offsets_mapping: If True, return character offsets for each token
            return_word_ids: If True, return word indices for each token

        Returns:
            TokenizerOutput with input_ids, attention_mask, and optionally
            offset_mapping and word_ids
        """
        is_single = isinstance(text, str)
        if is_single:
            text = [text]

        # Fast cached text cleaning
        if self.preprocess:
            text = clean_text_feature(text)

        if self.output_dim is not None:
            max_length = self.output_dim
            truncation = True
        else:
            max_length = None
            truncation = False

        # Vectorized encoding
        input_ids, attention_mask, offsets, word_ids = encode_batch_vectorized(
            text,
            self.subword_cache,
            self.eos_token_id,
            self.pad_token_id,
            max_length=max_length,
            truncation=truncation,
            return_offsets_mapping=return_offsets_mapping,
            return_word_ids=return_word_ids,
        )

        offsets = torch.tensor(offsets) if return_offsets_mapping else None
        word_ids = np.array(word_ids) if return_word_ids else None

        return TokenizerOutput(
            input_ids=input_ids,
            attention_mask=attention_mask,
            word_ids=word_ids,
            offset_mapping=offsets,
        )

    def decode(
        self, token_ids: Union[List[int], torch.Tensor], skip_special_tokens: bool = True
    ) -> str:
        """Decode token IDs back to text."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        tokens = []
        for id_ in token_ids:
            if id_ == self.pad_token_id and skip_special_tokens:
                continue

            if id_ == self.eos_token_id:
                if not skip_special_tokens:
                    tokens.append(self.EOS_TOKEN)
                continue

            if id_ in self.id_to_word:
                tokens.append(self.id_to_word[id_])
            elif not skip_special_tokens:
                tokens.append(f"[ID:{id_}]")

        return " ".join(tokens)

    def batch_decode(
        self, sequences: Union[List[List[int]], torch.Tensor], skip_special_tokens: bool = True
    ) -> List[str]:
        """Decode multiple sequences."""
        if isinstance(sequences, torch.Tensor):
            sequences = sequences.tolist()
        return [self.decode(seq, skip_special_tokens) for seq in sequences]

    def save_pretrained(self, save_directory: str):
        """Save tokenizer configuration and vocabulary."""
        import os

        os.makedirs(save_directory, exist_ok=True)

        config = {
            "min_count": self.min_count,
            "min_n": self.min_n,
            "max_n": self.max_n,
            "num_tokens": self.num_tokens,
            "len_word_ngrams": self.word_ngrams,
            "word_to_id": self.word_to_id,
            "preprocess": self.preprocess,
            "vocab_size": self.vocab_size,
            "nwords": self.nwords,
        }

        with open(f"{save_directory}/tokenizer.json", "w") as f:
            json.dump(config, f, indent=2)

        print(f"✓ Tokenizer saved to {save_directory}")

    @classmethod
    def from_pretrained(cls, directory: str):
        """Load tokenizer from saved configuration."""
        with open(f"{directory}/tokenizer.json", "r") as f:
            config = json.load(f)

        tokenizer = cls(
            min_count=config["min_count"],
            min_n=config["min_n"],
            max_n=config["max_n"],
            num_tokens=config["num_tokens"],
            len_word_ngrams=config["len_word_ngrams"],
            preprocess=config["preprocess"],
            training_text=None,
        )

        tokenizer.word_to_id = config["word_to_id"]
        tokenizer.nwords = config["nwords"]
        tokenizer.vocab_size = config["vocab_size"]

        tokenizer.id_to_word = {v: k for k, v in tokenizer.word_to_id.items()}
        tokenizer.id_to_word[tokenizer.pad_token_id] = cls.PAD_TOKEN
        tokenizer.id_to_word[tokenizer.unk_token_id] = cls.UNK_TOKEN
        tokenizer.id_to_word[tokenizer.eos_token_id] = cls.EOS_TOKEN

        # Rebuild subword cache
        print("Rebuilding subword cache...")
        tokenizer.subword_cache = SubwordCache(
            tokenizer.word_to_id,
            tokenizer.min_n,
            tokenizer.max_n,
            tokenizer.num_tokens,
            tokenizer.nwords,
            tokenizer.unk_token_id,
        )
        print("✓ Subword cache built")

        print(f"✓ Tokenizer loaded from {directory}")
        return tokenizer
