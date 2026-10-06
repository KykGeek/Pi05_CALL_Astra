"""Generic handoff contract and non-recovery backends for CALL_ASTRA."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping, Protocol

import numpy as np

from .recovery import RecoveryExecutor, RecoveryOutcome


@dataclass(frozen=True)
class HandoffRequest:
    task_instruction: str
    episode_id: str
    suite: str
    policy_query_idx: int
    call_step: int
    observation: Mapping[str, Any]
    robot_state: Any
    simulator_state: Any
    pi05_action: Any
    pi05_features: Mapping[str, Any]
    p_help: float
    threshold: float | None
    hard_threshold: float | None
    recent_scores: list[float]
    reason: str
    decision_rule: str
    bddl_path: str | None = None
    qpos: Any = None
    qvel: Any = None
    observation_id: str | None = None
    intervention_id: str | None = None


@dataclass(frozen=True)
class HandoffResponse:
    status: str
    detail: str
    snapshot_path: str | None = None
    recovery: RecoveryOutcome | None = None


class HandoffHandler(Protocol):
    def on_call(self, request: HandoffRequest) -> HandoffResponse:
        """Receive one CALL event. Implementations must not imply recovery success."""


class StopOnCallHandler:
    """Stop autonomous control without attempting recovery."""

    def on_call(self, request: HandoffRequest) -> HandoffResponse:
        return HandoffResponse(
            status="stopped",
            detail="π0.5 autonomous control stopped; no recovery action was executed.",
        )


class MockAstraHandler:
    """Acknowledge the handoff contract without calling Astra or acting."""

    def on_call(self, request: HandoffRequest) -> HandoffResponse:
        return HandoffResponse(
            status="acknowledged",
            detail="Mock handoff acknowledged; no Astra inference was run and no recovery action was executed.",
        )


def _as_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    else:
        try:
            value = np.asarray(value)
        except (TypeError, ValueError):
            return None
    array = np.asarray(value)
    if array.dtype.kind in "OUSV":
        return None
    return array


def _safe_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return name or "unknown"


def _validate_complete_snapshot(request: HandoffRequest) -> None:
    if not request.task_instruction.strip():
        raise ValueError("CALL snapshot is missing the task instruction")
    if not request.episode_id.strip() or not request.suite.strip():
        raise ValueError("CALL snapshot is missing suite or episode identity")
    if not request.bddl_path:
        raise ValueError("CALL snapshot is missing the task BDDL path")
    if request.robot_state is None or request.pi05_action is None:
        raise ValueError("CALL snapshot is missing robot state or the planned π0.5 action")
    if request.simulator_state is None and (request.qpos is None or request.qvel is None):
        raise ValueError("CALL snapshot needs simulator state or both qpos and qvel")

    observation_keys = {str(key).lower() for key in request.observation}
    image_aliases = {
        "observation/image",
        "observation.images.image",
        "agentview_image",
        "image",
    }
    wrist_aliases = {
        "observation/wrist_image",
        "observation.images.wrist_image",
        "wrist_image",
        "robot0_eye_in_hand_image",
    }
    if not observation_keys.intersection(image_aliases):
        raise ValueError("CALL snapshot is missing the external RGB image")
    if not observation_keys.intersection(wrist_aliases):
        raise ValueError("CALL snapshot is missing the wrist RGB image")

    feature_keys = {str(key).lower() for key in request.pi05_features}
    if not {"h_sem", "h_act"}.issubset(feature_keys):
        raise ValueError("CALL snapshot must include both h_sem and h_act")


def _collect_arrays(prefix: str, value: Any, arrays: dict[str, np.ndarray]) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _collect_arrays(f"{prefix}__{key}", nested, arrays)
        return
    if isinstance(value, (list, tuple)) and value and any(
        isinstance(item, (Mapping, list, tuple)) for item in value
    ):
        for index, nested in enumerate(value):
            _collect_arrays(f"{prefix}__{index}", nested, arrays)
        return
    array = _as_numpy(value)
    if array is not None:
        arrays[_safe_name(prefix)] = array


class LoggingHandoffHandler:
    """Persist a complete, immutable CALL snapshot and stop control."""

    def __init__(self, snapshot_root: str | Path) -> None:
        self.snapshot_root = Path(snapshot_root)

    def on_call(self, request: HandoffRequest) -> HandoffResponse:
        _validate_complete_snapshot(request)
        self.snapshot_root.mkdir(parents=True, exist_ok=True)
        call_name = _safe_name(
            f"{request.suite}_{request.episode_id}_q{request.policy_query_idx:05d}"
        )
        destination = self.snapshot_root / call_name
        if destination.exists():
            raise FileExistsError(f"CALL snapshot already exists: {destination}")

        arrays: dict[str, np.ndarray] = {}
        _collect_arrays("observation", request.observation, arrays)
        _collect_arrays("robot_state", request.robot_state, arrays)
        _collect_arrays("simulator_state", request.simulator_state, arrays)
        _collect_arrays("qpos", request.qpos, arrays)
        _collect_arrays("qvel", request.qvel, arrays)
        _collect_arrays("pi05_action", request.pi05_action, arrays)
        _collect_arrays("pi05_features", request.pi05_features, arrays)

        metadata = {
            "task_instruction": request.task_instruction,
            "suite": request.suite,
            "episode_id": request.episode_id,
            "policy_query_idx": int(request.policy_query_idx),
            "call_step": int(request.call_step),
            "observation_id": request.observation_id,
            "intervention_id": request.intervention_id,
            "bddl_path": request.bddl_path,
            "p_help": float(request.p_help),
            "threshold": None if request.threshold is None else float(request.threshold),
            "hard_threshold": (
                None if request.hard_threshold is None else float(request.hard_threshold)
            ),
            "recent_scores": [float(value) for value in request.recent_scores],
            "reason": request.reason,
            "decision_rule": request.decision_rule,
            "snapshot_arrays": "snapshot_arrays.npz",
            "array_keys": sorted(arrays),
            "observation_keys": sorted(str(key) for key in request.observation),
            "feature_keys": sorted(str(key) for key in request.pi05_features),
            "simulator_state_included": request.simulator_state is not None,
            "recovery_executed": False,
            "astra_called": False,
        }

        temporary = Path(
            tempfile.mkdtemp(prefix=f".{call_name}.", dir=self.snapshot_root)
        )
        try:
            np.savez_compressed(temporary / "snapshot_arrays.npz", **arrays)
            (temporary / "metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(temporary, destination)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

        return HandoffResponse(
            status="snapshot_saved",
            detail="CALL snapshot saved; autonomous control stopped.",
            snapshot_path=str(destination),
        )


class ResumableHandoffHandler:
    """Save the CALL snapshot, run a recovery executor, and return its checkpoint.

    The executor is deliberately injected.  This keeps the current benchmark
    safe: the default runner still uses ``LoggingHandoffHandler`` and never
    invokes Astra.  A real Astra adapter can be supplied later without
    changing the π0.5 controller or the snapshot contract.
    """

    def __init__(
        self,
        snapshot_root: str | Path,
        recovery_executor: RecoveryExecutor,
    ) -> None:
        self.snapshot_handler = LoggingHandoffHandler(snapshot_root)
        self.recovery_executor = recovery_executor

    @staticmethod
    def _record_recovery_metadata(
        snapshot_path: str | None, outcome: RecoveryOutcome
    ) -> None:
        if snapshot_path is None:
            raise ValueError("recovery result has no CALL snapshot path")
        outcome_path = Path(snapshot_path) / "recovery_outcome.json"
        outcome_record = {
                "astra_called": bool(outcome.astra_called),
                "recovery_executed": bool(outcome.recovery_executed),
                "recovery_status": outcome.status,
                "recovery_detail": outcome.detail,
                "intervention_start_step": int(outcome.intervention_start_step),
                "intervention_end_step": (
                    None
                    if outcome.intervention_end_step is None
                    else int(outcome.intervention_end_step)
                ),
                "pi05_resume_step": (
                    None if outcome.resume_step is None else int(outcome.resume_step)
                ),
                "recovery_checkpoint_id": outcome.checkpoint_id,
                "recovery_can_resume": bool(outcome.can_resume),
                "task_succeeded_during_recovery": bool(outcome.task_succeeded),
                "episode_id": outcome.episode_id,
                "intervention_id": outcome.intervention_id,
                "resume_observation_id": outcome.resume_observation_id,
                "execution_uncertain": bool(outcome.execution_uncertain),
                "model_response_received": bool(outcome.model_response_received),
                "resume_requested": bool(outcome.resume_requested),
                "audit_path": outcome.audit_path,
                "end_reason": outcome.end_reason,
                "model": outcome.model,
                "reasoning_effort": outcome.reasoning_effort,
                "codex_cli_version": outcome.codex_cli_version,
            }
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=outcome_path.parent,
            suffix=".outcome.tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(outcome_record, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, outcome_path)

    @staticmethod
    def _record_executor_error(snapshot_path: str | None, error: Exception) -> str:
        error_id = os.urandom(12).hex()
        if snapshot_path is not None:
            outcome_path = Path(snapshot_path) / "recovery_outcome.json"
            record = {
                "recovery_status": "failed",
                "astra_called": False,
                "recovery_executed": False,
                "error_type": type(error).__name__,
                "error_id": error_id,
                "can_resume": False,
                "execution_uncertain": False,
            }
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=outcome_path.parent,
                suffix=".outcome.tmp", delete=False
            ) as handle:
                temporary = Path(handle.name)
                json.dump(record, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, outcome_path)
        return error_id

    def on_call(self, request: HandoffRequest) -> HandoffResponse:
        snapshot = self.snapshot_handler.on_call(request)
        try:
            outcome = self.recovery_executor.intervene(
                request, snapshot.snapshot_path
            )
        except Exception as error:
            error_id = self._record_executor_error(snapshot.snapshot_path, error)
            return HandoffResponse(
                status="recovery_failed",
                detail=(
                    f"Astra/recovery executor failed after snapshot save: "
                    f"{type(error).__name__}; error_id={error_id}"
                ),
                snapshot_path=snapshot.snapshot_path,
            )
        if not isinstance(outcome, RecoveryOutcome):
            raise TypeError("recovery executor must return RecoveryOutcome")
        try:
            self._record_recovery_metadata(snapshot.snapshot_path, outcome)
        except Exception as error:
            # Never resume from an intervention whose durable checkpoint log
            # could not be updated.
            blocked = replace(outcome, can_resume=False)
            return HandoffResponse(
                status="recovery_metadata_failed",
                detail=(
                    f"recovery completed but checkpoint metadata could not be "
                    f"updated: {type(error).__name__}: {error}"
                ),
                snapshot_path=snapshot.snapshot_path,
                recovery=blocked,
            )
        if outcome.status == "completed":
            status = "recovery_completed"
        elif outcome.task_succeeded:
            status = "task_succeeded"
        else:
            status = f"recovery_{outcome.status}"
        return HandoffResponse(
            status=status,
            detail=outcome.detail,
            snapshot_path=snapshot.snapshot_path,
            recovery=outcome,
        )
