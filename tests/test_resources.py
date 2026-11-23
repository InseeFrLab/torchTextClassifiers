"""Tests for CPU-quota-aware sizing of DataLoader workers.

These tests only fake cgroup files and monkeypatch CPU counts: they never spawn
DataLoader worker processes.
"""

import os

import numpy as np
import pytest

from torchTextClassifiers.utilities import resources
from torchTextClassifiers.utilities.resources import (
    _cgroup_cpu_limit,
    available_cpu_count,
    default_num_workers,
)


@pytest.fixture
def big_host(monkeypatch):
    """Simulate a 128-CPU host with no affinity restriction."""
    monkeypatch.setattr(os, "cpu_count", lambda: 128)
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(128)), raising=False)


def test_cgroup_v2_quota(tmp_path):
    (tmp_path / "cpu.max").write_text("1500000 100000\n")
    assert _cgroup_cpu_limit(tmp_path) == 15


def test_cgroup_v2_fractional_quota_rounds_up(tmp_path):
    (tmp_path / "cpu.max").write_text("50000 100000\n")
    assert _cgroup_cpu_limit(tmp_path) == 1


def test_cgroup_v2_unlimited(tmp_path):
    (tmp_path / "cpu.max").write_text("max 100000\n")
    assert _cgroup_cpu_limit(tmp_path) is None


def test_cgroup_v1_quota(tmp_path):
    cpu = tmp_path / "cpu,cpuacct"
    cpu.mkdir()
    (cpu / "cpu.cfs_quota_us").write_text("400000\n")
    (cpu / "cpu.cfs_period_us").write_text("100000\n")
    assert _cgroup_cpu_limit(tmp_path) == 4


def test_cgroup_v1_unlimited(tmp_path):
    cpu = tmp_path / "cpu"
    cpu.mkdir()
    (cpu / "cpu.cfs_quota_us").write_text("-1\n")
    (cpu / "cpu.cfs_period_us").write_text("100000\n")
    assert _cgroup_cpu_limit(tmp_path) is None


def test_no_cgroup_files(tmp_path):
    assert _cgroup_cpu_limit(tmp_path) is None


def test_available_cpu_count_uses_cgroup_quota(tmp_path, big_host):
    (tmp_path / "cpu.max").write_text("1500000 100000\n")
    assert available_cpu_count(tmp_path) == 15


def test_available_cpu_count_uses_affinity(tmp_path, big_host, monkeypatch):
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1, 2}, raising=False)
    assert available_cpu_count(tmp_path) == 3


def test_available_cpu_count_falls_back_to_cpu_count(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: None)
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    assert available_cpu_count(tmp_path) == 1


@pytest.mark.parametrize("cpus, expected", [(1, 0), (2, 1), (4, 3), (15, 8), (128, 8)])
def test_default_num_workers(monkeypatch, cpus, expected):
    monkeypatch.setattr(resources, "available_cpu_count", lambda: cpus)
    assert default_num_workers() == expected


def test_create_dataloader_default_is_quota_aware(monkeypatch, sample_text_data):
    """The DataLoader default no longer derives from os.cpu_count()."""
    from torchTextClassifiers.dataset import dataset as dataset_module
    from torchTextClassifiers.dataset.dataset import TextClassificationDataset
    from torchTextClassifiers.tokenizers import NGramTokenizer

    monkeypatch.setattr(os, "cpu_count", lambda: 128)
    monkeypatch.setattr(dataset_module, "default_num_workers", lambda: 0)

    tokenizer = NGramTokenizer(
        min_count=1, min_n=2, max_n=4, num_tokens=50, len_word_ngrams=2, output_dim=20
    )
    tokenizer.train(list(sample_text_data))
    ds = TextClassificationDataset(
        texts=list(sample_text_data),
        categorical_variables=None,
        tokenizer=tokenizer,
        labels=np.zeros(len(sample_text_data), dtype=int).tolist(),
    )
    loader = ds.create_dataloader(batch_size=2)  # not iterated: no worker is started
    assert loader.num_workers == 0
    assert loader.persistent_workers is False


def test_training_config_default_num_workers_is_none():
    from torchTextClassifiers import TrainingConfig

    assert TrainingConfig(num_epochs=1, batch_size=2, lr=1e-3).num_workers is None
