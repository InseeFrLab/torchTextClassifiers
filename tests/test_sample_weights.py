import numpy as np
import pytest
import torch

from torchTextClassifiers import ModelConfig, TrainingConfig, torchTextClassifiers
from torchTextClassifiers.contrib import MultiLevelCrossEntropyLoss
from torchTextClassifiers.dataset import TextClassificationDataset
from torchTextClassifiers.model import TextClassificationModule
from torchTextClassifiers.tokenizers import NGramTokenizer


def _trained_ngram_tokenizer(texts):
    tokenizer = NGramTokenizer(
        min_count=1, min_n=2, max_n=4, num_tokens=50, len_word_ngrams=2, output_dim=20
    )
    tokenizer.train(list(texts))
    return tokenizer


class DummyClassificationModel(torch.nn.Module):
    """Bypasses tokenization/embedding: forwards pre-computed logits straight through."""

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.categorical_variable_net = None

    def forward(self, input_ids, attention_mask=None, categorical_vars=None, **kwargs):
        return input_ids


class TestDatasetSampleWeights:
    def test_default_sample_weights_are_ones(self, sample_text_data, sample_labels):
        tokenizer = _trained_ngram_tokenizer(sample_text_data)
        dataset = TextClassificationDataset(
            texts=sample_text_data.tolist(),
            categorical_variables=None,
            tokenizer=tokenizer,
            labels=sample_labels.tolist(),
        )
        dataloader = dataset.create_dataloader(
            batch_size=len(sample_text_data), shuffle=False, num_workers=0
        )
        batch = next(iter(dataloader))

        assert torch.allclose(batch["sample_weights"], torch.ones(len(sample_text_data)))

    def test_custom_sample_weights_flow_through_batch(self, sample_text_data, sample_labels):
        weights = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32)
        tokenizer = _trained_ngram_tokenizer(sample_text_data)
        dataset = TextClassificationDataset(
            texts=sample_text_data.tolist(),
            categorical_variables=None,
            tokenizer=tokenizer,
            labels=sample_labels.tolist(),
            sample_weights=weights,
        )
        dataloader = dataset.create_dataloader(
            batch_size=len(sample_text_data), shuffle=False, num_workers=0
        )
        batch = next(iter(dataloader))

        assert torch.allclose(batch["sample_weights"], torch.tensor(weights))


class TestLightningModuleSampleWeights:
    def _build_module(self, loss, num_classes=3):
        return TextClassificationModule(
            model=DummyClassificationModel(num_classes=num_classes),
            loss=loss,
            optimizer=torch.optim.Adam,
            optimizer_params={"lr": 1e-3},
            scheduler=None,
            scheduler_params=None,
        )

    @staticmethod
    def _make_batch(logits, targets, sample_weights=None):
        batch = {
            "input_ids": logits,
            "attention_mask": None,
            "categorical_vars": None,
            "labels": targets,
        }
        if sample_weights is not None:
            batch["sample_weights"] = sample_weights
        return batch

    def test_default_loss_reduction_switched_to_none(self):
        module = self._build_module(torch.nn.CrossEntropyLoss())
        assert module.loss.reduction == "none"

    def test_uniform_weights_match_unweighted_loss(self):
        torch.manual_seed(0)
        logits = torch.randn(5, 3)
        targets = torch.tensor([0, 1, 2, 1, 0])

        module = self._build_module(torch.nn.CrossEntropyLoss())
        batch = self._make_batch(logits, targets, torch.ones(5))
        loss, _ = module.step(batch)

        expected = torch.nn.functional.cross_entropy(logits, targets)
        assert torch.allclose(loss, expected, atol=1e-6)

    def test_missing_sample_weights_defaults_to_ones(self):
        torch.manual_seed(0)
        logits = torch.randn(5, 3)
        targets = torch.tensor([0, 1, 2, 1, 0])

        module = self._build_module(torch.nn.CrossEntropyLoss())
        batch = self._make_batch(logits, targets)
        loss, _ = module.step(batch)

        expected = torch.nn.functional.cross_entropy(logits, targets)
        assert torch.allclose(loss, expected, atol=1e-6)

    def test_weighted_loss_matches_manual_computation(self):
        torch.manual_seed(0)
        logits = torch.randn(4, 3)
        targets = torch.tensor([0, 1, 2, 1])
        weights = torch.tensor([1.0, 0.0, 2.0, 1.0])

        module = self._build_module(torch.nn.CrossEntropyLoss())
        batch = self._make_batch(logits, targets, weights)
        loss, _ = module.step(batch)

        per_sample = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
        expected = (per_sample * weights).sum() / weights.sum()
        assert torch.allclose(loss, expected, atol=1e-6)

    def test_zero_weight_equivalent_to_excluding_sample(self):
        torch.manual_seed(0)
        logits = torch.randn(3, 3)
        targets = torch.tensor([0, 1, 2])
        weights = torch.tensor([1.0, 0.0, 1.0])

        module = self._build_module(torch.nn.CrossEntropyLoss())
        batch_with_zero = self._make_batch(logits, targets, weights)
        loss_with_zero, _ = module.step(batch_with_zero)

        kept = [0, 2]
        batch_excluded = self._make_batch(logits[kept], targets[kept], torch.ones(2))
        loss_excluded, _ = module.step(batch_excluded)

        assert torch.allclose(loss_with_zero, loss_excluded, atol=1e-6)


class TestMultiLevelCrossEntropyLossSampleWeights:
    def test_weighted_matches_manual_computation(self):
        torch.manual_seed(0)
        outputs = [torch.randn(4, 3), torch.randn(4, 2)]
        labels = torch.stack([torch.tensor([0, 1, 2, 1]), torch.tensor([0, 1, 0, 1])], dim=1)
        weights = torch.tensor([1.0, 0.0, 2.0, 1.0])

        loss_fn = MultiLevelCrossEntropyLoss()
        loss = loss_fn(outputs, labels, sample_weights=weights)

        per_level_losses = []
        for i, out in enumerate(outputs):
            per_sample = torch.nn.functional.cross_entropy(out, labels[:, i], reduction="none")
            per_level_losses.append((per_sample * weights).sum() / weights.sum())
        expected = sum(per_level_losses) / len(outputs)

        assert torch.allclose(loss, expected, atol=1e-6)

    def test_none_sample_weights_matches_unweighted(self):
        torch.manual_seed(0)
        outputs = [torch.randn(4, 3)]
        labels = torch.tensor([0, 1, 2, 1]).unsqueeze(1)

        loss_fn = MultiLevelCrossEntropyLoss()
        weighted = loss_fn(outputs, labels)
        expected = torch.nn.functional.cross_entropy(outputs[0], labels[:, 0])

        assert torch.allclose(weighted, expected, atol=1e-6)


class TestWrapperSampleWeightsValidation:
    def test_check_sample_weights_none_passthrough(self):
        assert torchTextClassifiers._check_sample_weights(None, 5) is None

    def test_check_sample_weights_valid(self):
        weights = [0.1, 0.2, 0.3]
        checked = torchTextClassifiers._check_sample_weights(weights, 3)
        assert isinstance(checked, np.ndarray)
        assert checked.shape == (3,)

    def test_check_sample_weights_wrong_length_raises(self):
        with pytest.raises(ValueError):
            torchTextClassifiers._check_sample_weights([0.1, 0.2], 3)

    def test_check_sample_weights_negative_raises(self):
        with pytest.raises(AssertionError):
            torchTextClassifiers._check_sample_weights([-0.1, 0.2, 0.3], 3)


class TestTrainWithSampleWeights:
    def test_train_runs_with_sample_weights(self, sample_text_data, sample_labels):
        tokenizer = _trained_ngram_tokenizer(sample_text_data)
        model_config = ModelConfig(embedding_dim=8, num_classes=2)
        ttc = torchTextClassifiers(tokenizer=tokenizer, model_config=model_config)

        training_config = TrainingConfig(
            num_epochs=1,
            batch_size=4,
            lr=1e-3,
            num_workers=0,
            raw_labels=False,
        )

        sample_weights = np.linspace(0.5, 1.5, num=len(sample_text_data))
        val_sample_weights = np.linspace(0.5, 1.5, num=len(sample_text_data))

        ttc.train(
            X_train=sample_text_data,
            y_train=sample_labels,
            X_val=sample_text_data,
            y_val=sample_labels,
            sample_weights=sample_weights,
            val_sample_weights=val_sample_weights,
            training_config=training_config,
        )

        assert ttc.save_path is not None
