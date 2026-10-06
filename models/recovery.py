"""Recovery-to-π0.5 re-entry contract.

The CALL_ASTRA controller stops π0.5 before handing control to an external
recovery executor.  This module describes the result of that intervention and
implements the conservative gate that decides whether π0.5 may take control
back.  The gate intentionally uses only the returned checkpoint, images, and
robot state; it does not inspect object-state or simulator privileged labels.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Protocol

import numpy as np


@dataclass(frozen=True)
class RecoveryOutcome:
    """One completed or terminal Astra intervention."""

    status: str
    detail: str
    intervention_start_step: int
    intervention_end_step: int | None
    resume_step: int | None
    resume_observation: Mapping[str, Any] | None = None
    resume_robot_state: Any = None
    can_resume: bool = False
    task_succeeded: bool = False
    checkpoint_id: str | None = None
    astra_called: bool = True
    recovery_executed: bool = True
    episode_id: str | None = None
    intervention_id: str | None = None
    resume_observation_id: str | None = None
    execution_uncertain: bool = False
    model_response_received: bool = False
    resume_requested: bool = False
    audit_path: str | None = None
    end_reason: str | None = None
    model: str | None = None
    reasoning_effort: str | None = None
    codex_cli_version: str | None = None


@dataclass(frozen=True)
class RecoveryReentryDecision:
    """Auditable result of the π0.5 re-entry gate."""

    allowed: bool
    reason: str
    intervention_steps: int | None = None
    resume_step: int | None = None


class RecoveryExecutor(Protocol):
    """Adapter for the real Astra/recovery implementation."""

    def intervene(self, request: Any, snapshot_path: str | None) -> RecoveryOutcome:
        """Run recovery and return the exact checkpoint at which it ended."""


def _has_required_visuals(observation: Mapping[str, Any] | None) -> bool:
    if not isinstance(observation, Mapping):
        return False
    keys = {str(key).lower() for key in observation}
    raw_views = {"agentview_image", "robot0_eye_in_hand_image"}
    policy_views = {"observation/image", "observation/wrist_image"}
    return raw_views.issubset(keys) or policy_views.issubset(keys)


def _has_finite_robot_state(value: Any) -> bool:
    if value is None:
        return False
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return False
    return array.size > 0 and bool(np.isfinite(array).all())


class RecoveryReentryGate:
    """Allow π0.5 to resume only from a complete, synchronized checkpoint."""

    def evaluate(
        self,
        outcome: RecoveryOutcome | None,
        *,
        call_step: int | None = None,
    ) -> RecoveryReentryDecision:
        if outcome is None:
            return RecoveryReentryDecision(False, "missing_recovery_outcome")
        if outcome.status != "completed":
            return RecoveryReentryDecision(False, f"recovery_status_{outcome.status}")
        if outcome.task_succeeded:
            return RecoveryReentryDecision(False, "task_already_succeeded")
        if not outcome.can_resume:
            return RecoveryReentryDecision(False, "recovery_did_not_mark_resume_safe")

        start = int(outcome.intervention_start_step)
        end = outcome.intervention_end_step
        resume = outcome.resume_step
        if start < 0:
            return RecoveryReentryDecision(False, "invalid_intervention_start_step")
        if call_step is not None and start != int(call_step):
            return RecoveryReentryDecision(False, "intervention_start_does_not_match_call_step")
        if end is None or int(end) < start:
            return RecoveryReentryDecision(False, "invalid_intervention_end_step")
        if resume is None or int(resume) != int(end):
            return RecoveryReentryDecision(False, "resume_step_is_not_recovery_end_step")
        if not _has_required_visuals(outcome.resume_observation):
            return RecoveryReentryDecision(False, "resume_observation_missing_required_views")
        if not _has_finite_robot_state(outcome.resume_robot_state):
            return RecoveryReentryDecision(False, "resume_robot_state_missing_or_nonfinite")

        return RecoveryReentryDecision(
            True,
            "reentry_allowed",
            intervention_steps=int(end) - start,
            resume_step=int(resume),
        )


class MockRecoveryExecutor:
    """Small test adapter; it never calls Astra or changes an environment."""

    def intervene(self, request: Any, snapshot_path: str | None) -> RecoveryOutcome:
        end_step = int(request.call_step) + 5
        return RecoveryOutcome(
            status="completed",
            detail="mock recovery completed",
            intervention_start_step=int(request.call_step),
            intervention_end_step=end_step,
            resume_step=end_step,
            resume_observation=request.observation,
            resume_robot_state=request.robot_state,
            can_resume=True,
            checkpoint_id="mock-recovery-checkpoint",
            astra_called=False,
            recovery_executed=False,
        )
