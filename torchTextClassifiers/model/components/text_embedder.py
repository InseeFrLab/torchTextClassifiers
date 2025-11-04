import torch
from torch import nn


class TextEmbedder(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, padding_idx: int):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        self.embedding_layer = nn.Embedding(
            embedding_dim=embedding_dim,
            num_embeddings=vocab_size,
            padding_idx=self.padding_idx,
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Converts input token IDs to their corresponding embeddings."""

        encoded_text = input_ids  # clearer name
        if encoded_text.dtype != torch.long:
            encoded_text = encoded_text.to(torch.long)

        batch_size, seq_len = encoded_text.shape
        batch_size_check, seq_len_check = attention_mask.shape

        if batch_size != batch_size_check or seq_len != seq_len_check:
            raise ValueError(
                f"Input IDs and attention mask must have the same batch size and sequence length. "
                f"Got input_ids shape {encoded_text.shape} and attention_mask shape {attention_mask.shape}."
            )

        token_embeddings = self.embedding_layer(
            encoded_text
        )  # (batch_size, seq_len, embedding_dim)

        text_embedding = self._get_sentence_embedding(
            token_embeddings=token_embeddings, attention_mask=attention_mask
        )

        return text_embedding

    def _get_sentence_embedding(
        self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute sentence embedding from embedded tokens - "remove" second dimension.

        Args (output from dataset collate_fn):
            token_embeddings (torch.Tensor[Long]), shape (batch_size, seq_len, embedding_dim): Tokenized + padded text
            attention_mask (torch.Tensor[Long]), shape (batch_size, seq_len): Attention mask indicating non-pad tokens
        Returns:
            torch.Tensor: Sentence embeddings, shape (batch_size, embedding_dim)
        """

        # average over non-pad token embeddings
        # attention mask has 1 for non-pad tokens and 0 for pad token positions
        # TODO: add attention logic at some point

        # mask pad-tokens
        mask = attention_mask.unsqueeze(-1).float()  # (batch_size, seq_len, 1)
        masked_embeddings = token_embeddings * mask  # (batch_size, seq_len, embedding_dim)

        sentence_embedding = masked_embeddings.sum(dim=1) / mask.sum(dim=1).clamp(
            min=1.0
        )  # avoid division by zero

        sentence_embedding = torch.nan_to_num(sentence_embedding, 0.0)

        return sentence_embedding

    def __call__(self, *args, **kwargs):
        out = super().__call__(*args, **kwargs)
        if out.dim() != 2:
            raise ValueError(
                f"Output of {self.__class__.__name__}.forward must be 2D "
                f"(got shape {tuple(out.shape)})"
            )
        return out
