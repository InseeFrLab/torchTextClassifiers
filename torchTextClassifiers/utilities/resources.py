"""Helpers to size parallelism from the CPUs actually available to the process.

``os.cpu_count()`` reports the CPUs of the whole host. In containers (Kubernetes,
Docker, Onyxia, SLURM jobs...) the process is usually restricted by a cgroup CPU
quota or a CPU affinity mask, so using ``os.cpu_count()`` to size DataLoader
workers can spawn far more processes than the container can afford (each worker
being a full Python process, this can exhaust the container memory).
"""

import math
import os
from pathlib import Path
from typing import Optional, Union

# Each DataLoader worker is a separate process holding its own copy of the
# dataset/tokenizer: keep the default bounded, even on large machines.
MAX_DEFAULT_NUM_WORKERS = 8

_CGROUP_ROOT = Path("/sys/fs/cgroup")


def _read(path: Path) -> Optional[str]:
    try:
        return path.read_text().strip()
    except (OSError, ValueError):
        return None


def _cgroup_cpu_limit(cgroup_root: Union[str, Path] = _CGROUP_ROOT) -> Optional[int]:
    """CPU limit from the cgroup quota (v2 then v1), or None if unlimited/unknown."""
    root = Path(cgroup_root)

    # cgroup v2: "<quota> <period>" or "max <period>"
    content = _read(root / "cpu.max")
    if content:
        parts = content.split()
        if len(parts) == 2 and parts[0] != "max":
            try:
                quota, period = int(parts[0]), int(parts[1])
                if quota > 0 and period > 0:
                    return max(1, math.ceil(quota / period))
            except ValueError:
                pass

    # cgroup v1: quota is -1 when unlimited
    for cpu_dir in ("cpu", "cpu,cpuacct"):
        quota = _read(root / cpu_dir / "cpu.cfs_quota_us")
        period = _read(root / cpu_dir / "cpu.cfs_period_us")
        if quota and period:
            try:
                q, p = int(quota), int(period)
            except ValueError:
                continue
            if q > 0 and p > 0:
                return max(1, math.ceil(q / p))

    return None


def available_cpu_count(cgroup_root: Union[str, Path] = _CGROUP_ROOT) -> int:
    """Number of CPUs this process can actually use.

    Takes the minimum of the host CPU count, the CPU affinity mask and the cgroup
    CPU quota (v1 or v2). Always returns at least 1.
    """
    counts = [os.cpu_count() or 1]

    if hasattr(os, "sched_getaffinity"):
        try:
            counts.append(len(os.sched_getaffinity(0)))
        except OSError:
            pass

    cgroup_limit = _cgroup_cpu_limit(cgroup_root)
    if cgroup_limit is not None:
        counts.append(cgroup_limit)

    return max(1, min(counts))


def default_num_workers(max_workers: int = MAX_DEFAULT_NUM_WORKERS) -> int:
    """Default number of DataLoader workers: available CPUs - 1, capped at ``max_workers``."""
    return max(0, min(available_cpu_count() - 1, max_workers))
