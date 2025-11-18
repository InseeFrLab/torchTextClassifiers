"""
tests/test_tokenizer_benchmarks.py

Pytest integration for tokenizer benchmarks.
Run with: pytest tests/test_tokenizer_benchmarks.py --benchmark
"""

from pathlib import Path

import pytest

from tests.benchmark_suite import (
    compare_tokenizers,
    generate_test_data,
    plot_comparison,
)
from torchTextClassifiers.tokenizers.ngram import NGramTokenizer
from torchTextClassifiers.tokenizers.WordPiece import WordPieceTokenizer


@pytest.fixture(scope="module")
def training_data():
    """Generate training data once for all tests."""
    return generate_test_data(1000, avg_length=30)


@pytest.fixture(scope="module")
def ngram_tokenizer(training_data):
    """Create and train NGram tokenizer."""
    tokenizer = NGramTokenizer(
        min_count=2,
        min_n=2,
        max_n=4,
        num_tokens=10000,
        len_word_ngrams=2,
        training_text=training_data,
    )
    return tokenizer


@pytest.fixture(scope="module")
def wordpiece_tokenizer(training_data):
    """Create and train WordPiece tokenizer."""
    wp = WordPieceTokenizer(vocab_size=10000)
    wp.train(training_corpus=training_data)
    return wp


# ============================================================================
#                           Regular Tests (Always Run)
# ============================================================================


def test_ngram_tokenizer_basic(ngram_tokenizer):
    """Basic sanity test for NGram tokenizer."""
    test_text = ["hello world", "machine learning is awesome"]
    result = ngram_tokenizer.tokenize(test_text)

    assert result.input_ids is not None
    assert result.attention_mask is not None
    assert result.input_ids.shape[0] == len(test_text)


def test_wordpiece_tokenizer_basic(wordpiece_tokenizer):
    """Basic sanity test for WordPiece tokenizer."""
    test_text = ["hello world", "machine learning is awesome"]
    result = wordpiece_tokenizer.tokenize(test_text)

    assert result.input_ids is not None
    assert result.attention_mask is not None
    assert result.input_ids.shape[0] == len(test_text)


# ============================================================================
#                    Benchmark Tests (Run with --benchmark flag)
# ============================================================================


def test_tokenizer_comparison_small(ngram_tokenizer, wordpiece_tokenizer):
    """Compare tokenizers on small batch (CI-friendly)."""
    tokenizers = {
        "NGram": ngram_tokenizer,
        "WordPiece": wordpiece_tokenizer,
    }

    # Small batch sizes for CI
    results = compare_tokenizers(tokenizers, batch_sizes=[100, 500])

    # Ensure results were generated
    assert len(results) == 2
    for name, data in results.items():
        assert len(data) > 0, f"{name} produced no results"


def test_tokenizer_comparison_full(ngram_tokenizer, wordpiece_tokenizer):
    """Full benchmark comparison (for local testing)."""
    tokenizers = {
        "NGram": ngram_tokenizer,
        "WordPiece": wordpiece_tokenizer,
    }

    # Full benchmark
    results = compare_tokenizers(tokenizers, batch_sizes=[100, 500, 1000])

    # Save results
    output_dir = Path("benchmark_results")
    output_dir.mkdir(exist_ok=True)

    # Save plot
    plot_comparison(results, save_path=str(output_dir / "comparison.png"))

    # Save JSON results
    results_json = {}
    for name, data in results.items():
        results_json[name] = [
            {
                "batch_size": d["throughput"] / d["time"] * 1000,
                "time": d["time"],
                "throughput": d["throughput"],
            }
            for d in data
        ]

    print(f"\n✓ Results: {results_json}/")
