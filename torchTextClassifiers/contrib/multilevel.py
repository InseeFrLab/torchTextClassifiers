"""Multi-level (multi-task) classification architecture.

Provides a custom model and a matching loss function for tasks that require
predicting several classification targets simultaneously from the same text
input — for example, hierarchical category codes (level 1, level 2, level 3)
predicted in a single forward pass.

These classes are compatible with ``torchTextClassifiers.from_model`` and can
serve as a starting point for your own multi-task architectures.

Example usage::

    from torchTextClassifiers import torchTextClassifiers
    from torchTextClassifiers.contrib import (
        MultiLevelTextClassificationModel,
        MultiLevelCrossEntropyLoss,
    )
    from torchTextClassifiers.model import TextClassificationModule

    model = MultiLevelTextClassificationModel(
        token_embedder=token_embedder,
        sentence_embedders=sentence_embedders,   # one per level
        classification_heads=classification_heads,  # one per level
        categorical_variable_net=categorical_var_net,
    )

    # Train with PyTorch Lightning directly
    module = TextClassificationModule(
        model=model,
        loss=MultiLevelCrossEntropyLoss(),
        optimizer=torch.optim.Adam,
        optimizer_params={"lr": 1e-3},
    )

    # Or wrap with the high-level API for predict() / save() / load()
    classifier = torchTextClassifiers.from_model(
        tokenizer=tokenizer,
        pytorch_model=model,
    )
"""

from typing import Optional

import torch
from torch import nn

from torchTextClassifiers.model.components import (
    CategoricalForwardType,
    CategoricalVariableNet,
    ClassificationHead,
    SentenceEmbedder,
    TokenEmbedder,
)


class MultiLevelTextClassificationModel(nn.Module):
    """Multi-task text classifier that predicts several classes in one forward pass.

    Each classification level has its own ``SentenceEmbedder`` and
    ``ClassificationHead`` but they all share the same ``TokenEmbedder``, so
    the token-level representations are computed only once.

    Attributes:
        num_classes: List of class counts, one entry per level.  Required by
            ``torchTextClassifiers.from_model``.
        categorical_variable_net: The categorical embedding module (may be
            ``None``).  Required by ``torchTextClassifiers.from_model``.

    Args:
        token_embedder: Shared token embedding module.
        sentence_embedders: One ``SentenceEmbedder`` per classification level.
        classification_heads: One ``ClassificationHead`` per level.
        categorical_variable_net: Categorical feature embedding module.
    """

    def __init__(
        self,
        token_embedder: TokenEmbedder,
        sentence_embedders: list[SentenceEmbedder],
        classification_heads: list[ClassificationHead],
        categorical_variable_net: CategoricalVariableNet,
    ):
        super().__init__()
        self.token_embedder = token_embedder
        self.sentence_embedders = nn.ModuleList(sentence_embedders)
        self.classification_heads = nn.ModuleList(classification_heads)
        self.categorical_variable_net = categorical_variable_net
        self.num_classes: list[int] = [
            se.label_attention_config.num_classes
            if se.label_attention_config is not None
            else ch.num_classes
            for se, ch in zip(sentence_embedders, classification_heads)
        ]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        categorical_vars: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> list[torch.Tensor]:
        """Run a forward pass and return one logit tensor per level.

        Args:
            input_ids: Tokenised text, shape ``(batch, seq_len)``.
            attention_mask: Padding mask, shape ``(batch, seq_len)``.
            categorical_vars: Integer-encoded categorical features,
                shape ``(batch, n_cats)``.  May be ``None`` if no categorical
                features are used.

        Returns:
            List of raw logit tensors, one per classification level.
            Each tensor has shape ``(batch, num_classes_at_level)``.
        """
        token_embed_output = self.token_embedder(input_ids, attention_mask)
        x_token = token_embed_output["token_embeddings"]
        x_cat = self.categorical_variable_net(categorical_vars)

        outputs = []
        for sentence_embedder, classification_head in zip(
            self.sentence_embedders, self.classification_heads
        ):
            if sentence_embedder.label_attention_config is not None:
                num_cls = sentence_embedder.label_attention_config.num_classes
                x_cat_level = x_cat.unsqueeze(1).expand(-1, num_cls, -1)
            else:
                x_cat_level = x_cat

            sentence_embedding = sentence_embedder(
                token_embeddings=x_token, attention_mask=attention_mask
            )["sentence_embedding"]

            fwd = self.categorical_variable_net.forward_type
            if fwd in (
                CategoricalForwardType.AVERAGE_AND_CONCAT,
                CategoricalForwardType.CONCATENATE_ALL,
            ):
                x_combined = torch.cat((sentence_embedding, x_cat_level), dim=-1)
            else:
                assert fwd == CategoricalForwardType.SUM_TO_TEXT
                x_combined = sentence_embedding + x_cat_level

            outputs.append(classification_head(x_combined).squeeze(-1))

        return outputs


class MultiLevelCrossEntropyLoss(nn.Module):
    """Weighted cross-entropy loss across multiple classification levels.

    Averages the per-level cross-entropy losses, optionally weighting each
    level by its number of classes so that finer-grained levels contribute
    more to the total gradient.

    Args:
        num_classes: If provided, level ``i`` is weighted by
            ``num_classes[i] / sum(num_classes)``.  If ``None``, all levels
            are weighted equally.

    Example::

        loss_fn = MultiLevelCrossEntropyLoss(num_classes=[5, 20, 100])
        # or unweighted:
        loss_fn = MultiLevelCrossEntropyLoss()
    """

    def __init__(self, num_classes: Optional[list[int]] = None):
        super().__init__()
        self.num_classes = num_classes
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, outputs: list[torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
        """Compute the weighted average loss.

        Args:
            outputs: List of logit tensors ``(batch, num_classes_i)`` returned
                by ``MultiLevelTextClassificationModel``.
            labels: Integer label tensor of shape ``(batch, n_levels)``.
                Column ``i`` contains the ground-truth label for level ``i``.

        Returns:
            Scalar loss tensor.
        """
        total_loss = torch.tensor(0.0, device=outputs[0].device)
        for idx, output in enumerate(outputs):
            label = labels[:, idx]
            weight = self.num_classes[idx] if self.num_classes is not None else 1
            total_loss = total_loss + self.loss_fn(output.squeeze(), label) * weight

        total_weight = sum(self.num_classes) if self.num_classes is not None else len(outputs)
        return total_loss / total_weight
