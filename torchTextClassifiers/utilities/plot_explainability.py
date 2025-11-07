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

    attributions_per_char = np.empty((attributions.shape[0], len(text)))  # top_k, text_len

    for token_idx, (start, end) in enumerate(offsets):
        if start == end:
            continue
        attributions_per_char[:, start:end] = attributions[:, token_idx][:, None]

    return attributions_per_char


def map_attributions_to_word(attributions, word_ids):
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

    # Convert None to -1 for easier processing (PAD tokens)
    word_ids_int = np.array([x if x is not None else -1 for x in word_ids], dtype=int)

    # Consider only tokens that belong to actual words (non-PAD)
    unique_word_ids = np.unique(word_ids_int)
    unique_word_ids = unique_word_ids[unique_word_ids != -1]

    top_k = attributions.shape[0]
    attr_with_word_id = np.concat(
        (attributions[:, :, None], np.tile(word_ids_int[None, :], reps=(top_k, 1))[:, :, None]),
        axis=-1,
    )  # top_k, seq_len, 2
    # last dim is 2: 0 is the attribution of the token, 1 is the word_id the token is associated to

    word_attributions = np.zeros((top_k, len(word_ids_int)))
    for word_id in unique_word_ids:
        mask = attr_with_word_id[:, :, 1] == word_id  # top_k, seq_len
        word_attributions[:, word_id] = (attr_with_word_id[:, :, 0] * mask).sum(
            axis=1
        )  # zero-out non-matching tokens and sum attributions for all tokens belonging to the same word

    return word_attributions


def plot_attributions_at_char(text, attributions_per_char, title="Attributions", figsize=(10, 2)):
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

    plt.figure(figsize=figsize)
    plt.imshow(attributions_per_char, aspect="auto", cmap="viridis")
    plt.colorbar(label="Attribution Score")
    plt.yticks(
        ticks=np.arange(attributions_per_char.shape[0]),
        labels=[f"Top {i+1}" for i in range(attributions_per_char.shape[0])],
    )
    plt.xticks(ticks=np.arange(len(text)), labels=list(text), rotation=90)
    plt.title(title)
    plt.xlabel("Characters in Text")
    plt.ylabel("Top Predictions")
    plt.tight_layout()
    plt.show()


def plot_attributions_at_word(text, attributions_per_word, title="Attributions", figsize=(10, 2)):
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

    words = text.split()
    plt.figure(figsize=figsize)
    plt.imshow(attributions_per_word, aspect="auto", cmap="viridis")
    plt.colorbar(label="Attribution Score")
    plt.yticks(
        ticks=np.arange(attributions_per_word.shape[0]),
        labels=[f"Top {i+1}" for i in range(attributions_per_word.shape[0])],
    )
    plt.xticks(ticks=np.arange(len(words)), labels=words, rotation=90)
    plt.title(title)
    plt.xlabel("Words in Text")
    plt.ylabel("Top Predictions")
    plt.tight_layout()
    plt.show()
