"""Guard the CALL_ASTRA V1 runtime to the verified OpenPI LIBERO checkpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any


OFFICIAL_PI05_LIBERO_CHECKPOINT_NAME = "pi05_libero"
OFFICIAL_PI05_LIBERO_SOURCE_URI = "gs://openpi-assets/checkpoints/pi05_libero"
OFFICIAL_PI05_LIBERO_CONFIG = "pi05_libero"


def is_official_pi05_libero_path(path: str | Path) -> bool:
    return Path(path).expanduser().name == OFFICIAL_PI05_LIBERO_CHECKPOINT_NAME


def validate_official_pi05_libero_checkpoint(
    path: str | Path,
    *,
    require_exists: bool = True,
) -> Path:
    """Validate a user-supplied location for the compatible LIBERO checkpoint."""
    if not is_official_pi05_libero_path(path):
        raise ValueError(
            "CALL_ASTRA V1 expects the compatible OpenPI LIBERO checkpoint directory "
            f"to be named {OFFICIAL_PI05_LIBERO_CHECKPOINT_NAME!r}; received {path}"
        )
    resolved = Path(path).expanduser()
    if require_exists and not resolved.is_dir():
        raise FileNotFoundError(f"pi05_libero checkpoint is missing: {resolved}")
    return resolved.resolve()


def official_pi05_libero_metadata(checkpoint_path: str | Path) -> dict[str, Any]:
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_id": "openpi_pi05_libero",
        "policy_config": OFFICIAL_PI05_LIBERO_CONFIG,
        "checkpoint_source_uri": OFFICIAL_PI05_LIBERO_SOURCE_URI,
        "weight_role": "official_pi0.5_LIBERO_30k_finetuned_checkpoint",
    }
