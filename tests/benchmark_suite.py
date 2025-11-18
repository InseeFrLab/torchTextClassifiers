"""
Simplified benchmark suite for comparing tokenizers
"""

import random
import time
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

from torchTextClassifiers.tokenizers.ngram import NGramTokenizer
from torchTextClassifiers.tokenizers.WordPiece import WordPieceTokenizer

# ============================================================================
#                           Test Data Generation
# ============================================================================


def generate_test_data(num_samples: int, avg_length: int = 50) -> List[str]:
    """Generate synthetic text data."""
    words = [
        "the",
        "quick",
        "brown",
        "fox",
        "jumps",
        "over",
        "lazy",
        "dog",
        "machine",
        "learning",
        "artificial",
        "intelligence",
        "neural",
        "network",
        "tokenizer",
        "optimization",
        "performance",
        "benchmark",
        "testing",
        "python",
        "pytorch",
        "numpy",
        "data",
        "processing",
        "model",
    ]

    sentences = []
    for _ in range(num_samples):
        length = max(5, int(np.random.normal(avg_length, avg_length // 4)))
        sentence = " ".join(random.choices(words, k=length))
        sentences.append(sentence)

    return sentences


# ============================================================================
#                           Simple Benchmark
# ============================================================================


def benchmark_tokenizer(tokenizer, data: List[str], name: str, runs: int = 3) -> Dict:
    """Benchmark a single tokenizer on data."""

    # Warmup
    _ = tokenizer.tokenize(data[:10])

    # Benchmark
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        _ = tokenizer.tokenize(data)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    mean_time = np.mean(times)
    throughput = len(data) / mean_time

    return {
        "name": name,
        "time": mean_time,
        "std": np.std(times),
        "throughput": throughput,
        "times": times,
    }


def compare_tokenizers(tokenizers: Dict[str, Any], batch_sizes: List[int] = None):
    """
    Compare multiple tokenizers across different batch sizes.

    Args:
        tokenizers: Dict with {name: tokenizer_instance}
        batch_sizes: List of batch sizes to test
    """

    if batch_sizes is None:
        batch_sizes = [100, 500, 1000, 2000]

    print("=" * 80)
    print("TOKENIZER COMPARISON")
    print("=" * 80)

    results = {name: [] for name in tokenizers.keys()}

    for batch_size in batch_sizes:
        print(f"\n--- Batch Size: {batch_size} ---")
        test_data = generate_test_data(batch_size)

        batch_results = []
        for name, tokenizer in tokenizers.items():
            try:
                result = benchmark_tokenizer(tokenizer, test_data, name)
                results[name].append(result)

                print(
                    f"{name:20s}: {result['time']:.3f}s ± {result['std']:.3f}s "
                    f"({result['throughput']:.0f} samples/sec)"
                )
                batch_results.append(result)

            except Exception as e:
                print(f"{name:20s}: FAILED - {e}")

        # Show speedup
        if len(batch_results) > 1:
            fastest = min(batch_results, key=lambda x: x["time"])
            slowest = max(batch_results, key=lambda x: x["time"])
            speedup = slowest["time"] / fastest["time"]
            print(f"\n  → {fastest['name']} is {speedup:.2f}x faster than {slowest['name']}")

    return results


def plot_comparison(results: Dict[str, List[Dict]], save_path: str = "comparison.png"):
    """Plot comparison results."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Throughput vs Batch Size
    for name, data in results.items():
        if not data:
            continue
        batch_sizes = [d["throughput"] / d["time"] * 1000 for d in data]  # rough estimate
        throughputs = [d["throughput"] for d in data]
        ax1.plot(batch_sizes, throughputs, marker="o", label=name, linewidth=2)

    ax1.set_xlabel("Batch Size")
    ax1.set_ylabel("Throughput (samples/sec)")
    ax1.set_title("Throughput Comparison")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Time comparison (last batch size)
    names = []
    times = []
    colors = []

    for i, (name, data) in enumerate(results.items()):
        if data:
            names.append(name)
            times.append(data[-1]["time"])
            colors.append(f"C{i}")

    if times:
        bars = ax2.barh(range(len(names)), times, color=colors, alpha=0.7)
        ax2.set_yticks(range(len(names)))
        ax2.set_yticklabels(names)
        ax2.set_xlabel("Time (seconds)")
        ax2.set_title("Processing Time Comparison")
        ax2.grid(True, alpha=0.3, axis="x")

        # Add value labels
        for i, (bar, t) in enumerate(zip(bars, times)):
            ax2.text(t + 0.01, i, f"{t:.3f}s", va="center")

        # Mark fastest
        fastest_idx = times.index(min(times))
        ax2.get_yticklabels()[fastest_idx].set_weight("bold")
        ax2.get_yticklabels()[fastest_idx].set_color("green")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n✓ Plot saved to {save_path}")
    plt.close()


def print_summary(results: Dict[str, List[Dict]]):
    """Print summary statistics."""

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Get last batch results (largest)
    last_batch = {name: data[-1] for name, data in results.items() if data}

    if not last_batch:
        print("No results to summarize")
        return

    fastest = min(last_batch.items(), key=lambda x: x[1]["time"])
    slowest = max(last_batch.items(), key=lambda x: x[1]["time"])

    print(f"\n🏆 Winner: {fastest[0]}")
    print(f"   Time: {fastest[1]['time']:.3f}s")
    print(f"   Throughput: {fastest[1]['throughput']:.0f} samples/sec")

    if len(last_batch) > 1:
        speedup = slowest[1]["time"] / fastest[1]["time"]
        print(f"\n   {speedup:.2f}x faster than {slowest[0]}")

    print("\n" + "-" * 80)
    print("All tokenizers (sorted by speed):")
    for name, result in sorted(last_batch.items(), key=lambda x: x[1]["time"]):
        speedup = slowest[1]["time"] / result["time"]
        print(f"  {name:20s}: {result['time']:.3f}s ({speedup:.2f}x)")


if __name__ == "__main__":
    """
    Simple usage example:

    1. Train your tokenizers
    2. Put them in a dict
    3. Run comparison
    """

    print("Training tokenizers...")
    training_data = generate_test_data(1000, avg_length=30)

    # Create tokenizers
    tokenizers = {}

    # NGram tokenizer
    tokenizers["NGram"] = NGramTokenizer(
        min_count=2,
        min_n=2,
        max_n=4,
        num_tokens=10000,
        len_word_ngrams=2,
        training_text=training_data,
    )

    # WordPiece tokenizer
    wp = WordPieceTokenizer(vocab_size=10000)
    wp.train(training_corpus=training_data)
    tokenizers["WordPiece"] = wp

    print(f"\n✓ Trained {len(tokenizers)} tokenizers\n")

    # Run comparison
    results = compare_tokenizers(tokenizers, batch_sizes=[100, 500, 1000])

    # Plot results
    plot_comparison(results)

    # Print summary
    print_summary(results)

    print("\n✓ Benchmark complete!")
