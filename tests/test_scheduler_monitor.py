"""ReduceLROnPlateau must monitor the same metric as early stopping / checkpointing."""

import logging

import pytest
import torch

from torchTextClassifiers import ModelConfig, TrainingConfig, torchTextClassifiers
from torchTextClassifiers.model import TextClassificationModule
from torchTextClassifiers.tokenizers import NGramTokenizer


class DummyClassificationModel(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.linear = torch.nn.Linear(2, num_classes)

    def forward(self, input_ids, attention_mask=None, categorical_vars=None, **kwargs):
        return self.linear(input_ids)


def _build_module(scheduler, **kwargs):
    return TextClassificationModule(
        model=DummyClassificationModel(num_classes=2),
        loss=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam,
        optimizer_params={"lr": 1e-3},
        scheduler=scheduler,
        scheduler_params={},
        **kwargs,
    )


class TestConfigureOptimizers:
    def test_plateau_defaults_to_train_loss(self):
        _, [config] = _build_module(
            torch.optim.lr_scheduler.ReduceLROnPlateau
        ).configure_optimizers()
        assert config["monitor"] == "train_loss"

    def test_plateau_uses_given_monitor(self):
        module = _build_module(
            torch.optim.lr_scheduler.ReduceLROnPlateau, scheduler_monitor="val_loss"
        )
        _, [config] = module.configure_optimizers()
        assert config["monitor"] == "val_loss"

    def test_non_plateau_scheduler_has_no_monitor(self):
        module = TextClassificationModule(
            model=DummyClassificationModel(num_classes=2),
            loss=torch.nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam,
            optimizer_params={"lr": 1e-3},
            scheduler=torch.optim.lr_scheduler.StepLR,
            scheduler_params={"step_size": 1},
            scheduler_monitor="val_loss",
        )
        _, [scheduler] = module.configure_optimizers()
        assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)


def _classifier(texts):
    tokenizer = NGramTokenizer(
        min_count=1, min_n=2, max_n=4, num_tokens=50, len_word_ngrams=2, output_dim=20
    )
    tokenizer.train(list(texts))
    return torchTextClassifiers(
        tokenizer=tokenizer, model_config=ModelConfig(embedding_dim=8, num_classes=2)
    )


def _training_config(tmp_path, scheduler_patience=1, early_stopping_patience=3):
    return TrainingConfig(
        num_epochs=2,
        batch_size=4,
        lr=1e-3,
        num_workers=0,
        raw_labels=False,
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
        scheduler_params={"patience": scheduler_patience},
        patience_early_stopping=early_stopping_patience,
        trainer_params={"default_root_dir": str(tmp_path), "enable_progress_bar": False},
    )


class TestTrainSchedulerMonitor:
    def test_monitors_val_loss_with_validation_set(self, tmp_path, sample_text_data, sample_labels):
        ttc = _classifier(sample_text_data)
        ttc.train(
            X_train=sample_text_data,
            y_train=sample_labels,
            X_val=sample_text_data,
            y_val=sample_labels,
            training_config=_training_config(tmp_path),
        )
        assert ttc.lightning_module.scheduler_monitor == "val_loss"

    def test_falls_back_to_train_loss_without_validation_set(
        self, tmp_path, sample_text_data, sample_labels
    ):
        ttc = _classifier(sample_text_data)
        ttc.train(
            X_train=sample_text_data,
            y_train=sample_labels,
            training_config=_training_config(tmp_path),
        )
        assert ttc.lightning_module.scheduler_monitor == "train_loss"

    def test_user_can_keep_train_loss_with_validation_set(
        self, tmp_path, sample_text_data, sample_labels
    ):
        ttc = _classifier(sample_text_data)
        config = _training_config(tmp_path)
        config.scheduler_monitor = "train_loss"
        ttc.train(
            X_train=sample_text_data,
            y_train=sample_labels,
            X_val=sample_text_data,
            y_val=sample_labels,
            training_config=config,
        )
        assert ttc.lightning_module.scheduler_monitor == "train_loss"

    def test_val_monitor_without_validation_set_raises(
        self, tmp_path, sample_text_data, sample_labels
    ):
        ttc = _classifier(sample_text_data)
        config = _training_config(tmp_path)
        config.scheduler_monitor = "val_loss"
        with pytest.raises(ValueError, match="requires a validation set"):
            ttc.train(X_train=sample_text_data, y_train=sample_labels, training_config=config)

    @pytest.mark.parametrize("scheduler_patience, warns", [(3, True), (2, False)])
    def test_warns_when_scheduler_patience_not_below_early_stopping(
        self, tmp_path, caplog, sample_text_data, sample_labels, scheduler_patience, warns
    ):
        ttc = _classifier(sample_text_data)
        with caplog.at_level(logging.WARNING, logger="torchTextClassifiers"):
            ttc.train(
                X_train=sample_text_data,
                y_train=sample_labels,
                X_val=sample_text_data,
                y_val=sample_labels,
                training_config=_training_config(
                    tmp_path, scheduler_patience=scheduler_patience, early_stopping_patience=3
                ),
            )
        assert any("ReduceLROnPlateau patience" in r.message for r in caplog.records) is warns

    def test_warns_with_default_scheduler_patience(
        self, tmp_path, caplog, sample_text_data, sample_labels
    ):
        # No 'patience' given: PyTorch's default (10) is not below early stopping (3)
        ttc = _classifier(sample_text_data)
        config = _training_config(tmp_path)
        config.scheduler_params = None
        with caplog.at_level(logging.WARNING, logger="torchTextClassifiers"):
            ttc.train(
                X_train=sample_text_data,
                y_train=sample_labels,
                X_val=sample_text_data,
                y_val=sample_labels,
                training_config=config,
            )
        assert any("ReduceLROnPlateau patience (10)" in r.message for r in caplog.records)
