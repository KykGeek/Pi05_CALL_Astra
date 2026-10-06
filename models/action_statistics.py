"""Frozen π0.5 action-chunk statistics used by the trained CALL_ASTRA head."""
from __future__ import annotations

from typing import Any

import numpy as np


def action_statistics(action_chunk: Any) -> np.ndarray:
    """Match the exact 36-value auxiliary representation used during training."""
    action = np.asarray(action_chunk, dtype=np.float32)
    if action.shape != (10, 7):
        raise ValueError("action_chunk_must_be_10_by_7")
    if not np.isfinite(action).all():
        raise ValueError("action_chunk_must_be_finite")
    mean = action.mean(axis=-2)
    std = action.std(axis=-2)
    norm = np.linalg.norm(action.reshape(-1))
    first = action[0]
    last = action[-1]
    delta = last - first
    stats = np.concatenate([mean, std, np.asarray([norm]), first, last, delta]).astype(
        np.float32
    )
    if stats.shape != (36,) or not np.isfinite(stats).all():
        raise ValueError("action_statistics_malformed")
    return stats
