"""Paths, seeding and provenance for every reported number.

Every result file written by the research pipeline carries the git commit, a
hash of the configuration that produced it, the dataset hash and the library
versions, so that each table cell in the manuscript can be traced to a run.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
KAGGLE_DIR = REPO_ROOT / "data" / "external" / "kaggle_provider_fraud"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def stable_hash(obj: Any) -> str:
    """SHA-256 of a JSON-serialisable object with sorted keys (first 16 hex chars)."""
    payload = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def frame_hash(df) -> str:
    """Order-sensitive content hash of a DataFrame."""
    import pandas as pd

    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()[:16]


def git_state() -> dict:
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return "unknown"

    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain", "--", "research"))}


def environment() -> dict:
    versions = {}
    for mod in ["numpy", "pandas", "scipy", "sklearn", "torch", "torch_geometric"]:
        try:
            versions[mod] = __import__(mod).__version__
        except ImportError:
            versions[mod] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        **versions,
    }


def provenance(config: dict, **extra: Any) -> dict:
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_state(),
        "config_hash": stable_hash(config),
        "config": config,
        "environment": environment(),
        **extra,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    tmp.replace(path)


def _json_default(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serialisable: {type(value)}")
