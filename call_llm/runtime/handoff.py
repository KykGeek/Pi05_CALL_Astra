"""Exclusive ownership transition from π0.5 to the Astra executor."""
from __future__ import annotations

from dataclasses import replace
from typing import Any
from uuid import uuid4

import numpy as np

from ..handoff import HandoffRequest, HandoffResponse, ResumableHandoffHandler


class ExclusiveAstraHandoffHandler:
    """Clear queued π0.5 actions and transfer one live env to Astra exactly once."""

    def __init__(self, *, snapshot_root: str, executor: Any, broker: Any,
                 action_queue: Any) -> None:
        self.broker = broker
        self.action_queue = action_queue
        self.inner = ResumableHandoffHandler(snapshot_root, executor)

    def on_call(self, request: HandoffRequest) -> HandoffResponse:
        checkpoint = self.broker.current_checkpoint()
        if self.broker.owner != "pi05" or self.broker.uncertain:
            raise RuntimeError("pi05_control_owner_not_confirmed_at_CALL")
        if int(request.call_step) != int(checkpoint.env_step):
            raise RuntimeError("call_step_does_not_match_host_checkpoint")
        _verify_public_observation(request.observation, checkpoint.raw)
        # No action from the already-inferred chunk may survive this point.
        self.action_queue.clear()
        bound_request = replace(
            request,
            observation_id=str(checkpoint.observation_id),
            intervention_id="astra-" + str(uuid4()),
        )
        self.broker.transfer("pi05", "astra")
        try:
            response = self.inner.on_call(bound_request)
        except BaseException:
            self.broker.stop("handoff_handler_exception")
            raise
        recovery = response.recovery
        if recovery is None:
            self.broker.stop("recovery_outcome_missing")
        elif (
            recovery.episode_id != str(bound_request.episode_id)
            or recovery.intervention_id != str(bound_request.intervention_id)
            or recovery.intervention_start_step != int(bound_request.call_step)
        ):
            recovery = replace(
                recovery,
                status="unsafe",
                detail="recovery_identity_or_start_step_mismatch",
                can_resume=False,
                resume_requested=False,
            )
            response = replace(response, status="recovery_unsafe", recovery=recovery)
            self.broker.stop("recovery_identity_or_start_step_mismatch")
        elif recovery.execution_uncertain or not recovery.can_resume:
            if not recovery.task_succeeded:
                self.broker.stop("recovery_not_resumable")
        return response


def _verify_public_observation(request_observation: Any, host_observation: Any) -> None:
    if not isinstance(request_observation, dict) or not isinstance(host_observation, dict):
        raise RuntimeError("handoff_observation_invalid")
    for key in ("agentview_image", "robot0_eye_in_hand_image", "robot0_eef_pos",
                "robot0_eef_quat", "robot0_gripper_qpos"):
        if key not in request_observation or key not in host_observation:
            raise RuntimeError("handoff_public_observation_missing_" + key)
        left = np.asarray(request_observation[key])
        right = np.asarray(host_observation[key])
        if left.shape != right.shape or left.dtype != right.dtype or not np.array_equal(left, right):
            raise RuntimeError("handoff_observation_not_synchronized_" + key)
