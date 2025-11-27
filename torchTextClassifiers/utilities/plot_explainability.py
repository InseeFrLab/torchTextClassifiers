from typing import List, Optional

import numpy as np
import torch

try:
    from matplotlib import pyplot as plt

    HAS_PYPLOT = True
except ImportError:
    HAS_PYPLOT = False


def map_attributions_to_char(attributions, offsets, text):
    """
    Maps token-level attributions to character-level attributions based on token offsets.
    Args:
        attributions (np.ndarray): Array of shape (top_k, seq_len) or (seq_len,) containing token-level attributions.
               Output from:
                >>> ttc.predict(X, top_k=top_k, explain=True)["attributions"]
        offsets (list of tuples): List of (start, end) offsets for each token in the original text.
                Output from:
                >>> ttc.predict(X, top_k=top_k, explain=True)["offset_mapping"]
                Also from:
                >>> ttc.tokenizer.tokenize(text, return_offsets_mapping=True)["offset_mapping"]
        text (str): The original input text.

    Returns:
        np.ndarray: Array of shape (top_k, text_len) containing character-level attributions.
            text_len is the number of characters in the original text.

    """

    if isinstance(text, list):
        raise ValueError("text must be a single string, not a list of strings.")

    assert isinstance(text, str), "text must be a string."

    if isinstance(attributions, torch.Tensor):
        attributions = attributions.cpu().numpy()

    if attributions.ndim == 1:
        attributions = attributions[None, :]

    attributions_per_char = np.zeros((attributions.shape[0], len(text)))  # top_k, text_len

    for token_idx, (start, end) in enumerate(offsets):
        if start == end:  # skip special tokens
            continue
        attributions_per_char[:, start:end] = attributions[:, token_idx][:, None]

    return np.exp(attributions_per_char) / np.sum(
        np.exp(attributions_per_char), axis=1, keepdims=True
    )  # softmax normalization

def get_id_to_word(text, word_ids, offsets):
    words = {}
    for idx, word_id in enumerate(word_ids):
        if word_id is None:
            continue
        start, end = offsets[idx]
        words[int(word_id)] = text[start:end]
    
    return words


def map_attributions_to_word(attributions, text, word_ids, offsets):
    """
    Maps token-level attributions to word-level attributions based on word IDs.
    Args:
        attributions (np.ndarray): Array of shape (top_k, seq_len) or (seq_len,) containing token-level attributions.
               Output from:
                >>> ttc.predict(X, top_k=top_k, explain=True)["attributions"]
        word_ids (list of int or None): List of word IDs for each token in the original text.
                Output from:
                >>> ttc.predict(X, top_k=top_k, explain=True)["word_ids"]

    Returns:
        np.ndarray: Array of shape (top_k, num_words) containing word-level attributions.
            num_words is the number of unique words in the original text.
    """
    
    word_ids = np.array(word_ids)
    words = get_id_to_word(text, word_ids, offsets)

    # Convert None to -1 for easier processing (PAD tokens)
    word_ids_int = np.array([x if x is not None else -1 for x in word_ids], dtype=int)

    # Filter out PAD tokens from attributions and word_ids
    attributions = attributions[
        torch.arange(attributions.shape[0])[:, None],
        torch.tensor(np.where(word_ids_int != -1)[0])[None, :],
    ]
    word_ids_int = word_ids_int[word_ids_int != -1]
    unique_word_ids = np.unique(word_ids_int)
    num_unique_words = len(unique_word_ids)

    top_k = attributions.shape[0]
    attr_with_word_id = np.concat(
        (attributions[:, :, None], np.tile(word_ids_int[None, :], reps=(top_k, 1))[:, :, None]),
        axis=-1,
    )  # top_k, seq_len, 2
    # last dim is 2: 0 is the attribution of the token, 1 is the word_id the token is associated to

    word_attributions = np.zeros((top_k, num_unique_words))
    for word_id in unique_word_ids:
        mask = attr_with_word_id[:, :, 1] == word_id  # top_k, seq_len
        word_attributions[:, word_id] = (attr_with_word_id[:, :, 0] * mask).sum(
            axis=1
        )  # zero-out non-matching tokens and sum attributions for all tokens belonging to the same word

    # assert word_attributions.sum(axis=1) == attributions.sum(axis=1), "Sum of word attributions per top_k must equal sum of token attributions per top_k."
    return words, np.exp(word_attributions) / np.sum(
        np.exp(word_attributions), axis=1, keepdims=True
    )  # softmax normalization


def plot_attributions_at_char(
    text: str,
    attributions_per_char: np.ndarray,
    figsize=(10, 2),
    titles: Optional[List[str]] = None,
):
    """
    Plots character-level attributions as a heatmap.
    Args:
        text (str): The original input text.
        attributions_per_char (np.ndarray): Array of shape (top_k, text_len) containing character-level attributions.
               Output from map_attributions_to_char function.
        title (str): Title of the plot.
        figsize (tuple): Figure size for the plot.
    """

    if not HAS_PYPLOT:
        raise ImportError(
            "matplotlib is required for plotting. Please install it to use this function."
        )
    top_k = attributions_per_char.shape[0]

    all_plots = []
    for i in range(top_k):
        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(range(len(text)), attributions_per_char[i])
        ax.set_xticks(np.arange(len(text)))
        ax.set_xticklabels(list(text), rotation=45)
        title = titles[i] if titles is not None else f"Attributions for Top {i+1} Prediction"
        ax.set_title(title)
        ax.set_xlabel("Characters in Text")
        ax.set_ylabel("Top Predictions")
        all_plots.append(fig)

    return all_plots


def plot_attributions_at_word(
    text, words, attributions_per_word, figsize=(10, 2), titles: Optional[List[str]] = None
):
    """
    Plots word-level attributions as a heatmap.
    Args:
        text (str): The original input text.
        attributions_per_word (np.ndarray): Array of shape (top_k, num_words) containing word-level attributions.
               Output from map_attributions_to_word function.
        title (str): Title of the plot.
        figsize (tuple): Figure size for the plot.
    """

    if not HAS_PYPLOT:
        raise ImportError(
            "matplotlib is required for plotting. Please install it to use this function."
        )

    top_k = attributions_per_word.shape[0]
    all_plots = []
    for i in range(top_k):
        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(range(len(words)), attributions_per_word[i])
        ax.set_xticks(np.arange(len(words)))
        ax.set_xticklabels(words, rotation=45)
        title = titles[i] if titles is not None else f"Attributions for Top {i+1} Prediction"
        ax.set_title(title)
        ax.set_xlabel("Words in Text")
        ax.set_ylabel("Attributions")
        all_plots.append(fig)

    return all_plots


def figshow(figure):
    # https://stackoverflow.com/questions/53088212/create-multiple-figures-in-pyplot-but-only-show-one
    for i in plt.get_fignums():
        if figure != plt.figure(i):
            plt.close(plt.figure(i))
    plt.show()
