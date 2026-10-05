"""Closed-loop π0.5 self-assessment controller."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Callable, Mapping

import numpy as np

from .decision import CallDecision, DecisionEngine
from .handoff import HandoffHandler, HandoffRequest, HandoffResponse
from .recovery import RecoveryOutcome, RecoveryReentryDecision, RecoveryReentryGate


class ControllerState(str, Enum):
    VLA_RUN = "VLA_RUN"
    CALL_PENDING = "CALL_PENDING"
    HANDOFF = "HANDOFF"
    RECOVERY_READY = "RECOVERY_READY"
    TERMINATED = "TERMINATED"


class ControllerStoppedError(RuntimeError):
    """Raised when a caller attempts another policy query after CALL/termination."""


@dataclass(frozen=True)
class QueryOutcome:
    state: ControllerState
    decision: CallDecision
    actions_to_execute: Any | None
    handoff_response: HandoffResponse | None = None
    recovery_outcome: RecoveryOutcome | None = None
    reentry_decision: RecoveryReentryDecision | None = None

    def as_api_response(self) -> dict[str, Any]:
        """Return the public actions-plus-assist response contract."""
        return {
            "actions": self.actions_to_execute,
            "assist": {
                "p_help": self.decision.p_help,
                "decision": self.decision.decision,
                "threshold": self.decision.threshold,
                "hard_threshold": self.decision.hard_threshold,
                "decision_rule": self.decision.decision_rule,
                "reason": self.decision.reason,
                "recent_scores": list(self.decision.recent_scores),
                "recovery_status": (
                    self.recovery_outcome.status
                    if self.recovery_outcome is not None
                    else None
                ),
                "reentry_allowed": (
                    self.reentry_decision.allowed
                    if self.reentry_decision is not None
                    else None
                ),
            },
        }


def _snapshot_copy(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _snapshot_copy(nested) for key, nested in value.items()}
    if isinstance(value, tuple):
        return tuple(_snapshot_copy(nested) for nested in value)
    if isinstance(value, list):
        return [_snapshot_copy(nested) for nested in value]
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return np.asarray(value.detach().cpu().numpy()).copy()
    if isinstance(value, np.ndarray):
        return value.copy()
    return value


class CallController:
    """Run one π0.5 query, score its features, and either execute or hand off.

    The policy inference result is treated as read-only. On CONTINUE the exact
    action object returned by π0.5 is passed through. On CALL_ASTRA the planned
    chunk is recorded in the handoff request but is never returned for execution.
    In shadow mode, CALL_ASTRA is recorded by the caller but the action chunk is
    returned so the episode can continue to its natural outcome. Multiple live
    handoffs are allowed after a host-enforced minimum step interval.
    """

    def __init__(
        self,
        *,
        episode_id: str,
        suite: str,
        policy_infer: Callable[[Mapping[str, Any]], Mapping[str, Any]],
        score_help: Callable[[Mapping[str, Any]], float],
        handoff_handler: HandoffHandler,
        decision_engine: DecisionEngine,
        shadow_mode: bool = False,
        reentry_gate: RecoveryReentryGate | None = None,
        min_steps_between_calls: int = 20,
    ) -> None:
        self.episode_id = str(episode_id)
        self.suite = str(suite)
        self.policy_infer = policy_infer
        self.score_help = score_help
        self.handoff_handler = handoff_handler
        self.decision_engine = decision_engine
        self.shadow_mode = bool(shadow_mode)
        self.reentry_gate = reentry_gate or RecoveryReentryGate()
        if int(min_steps_between_calls) < 0:
            raise ValueError("min_steps_between_calls must be nonnegative")
        self.min_steps_between_calls = int(min_steps_between_calls)
        self.state = ControllerState.VLA_RUN
        self._handoff_attempted = False
        self._last_handoff_step: int | None = None
        self._last_decision: CallDecision | None = None
        self._last_policy_result: Mapping[str, Any] | None = None
        self._recovery_outcome: RecoveryOutcome | None = None
        self._reentry_decision: RecoveryReentryDecision | None = None

    def process_policy_query(
        self,
        observation: Mapping[str, Any],
        *,
        environment_observation: Mapping[str, Any] | None = None,
        task_instruction: str,
        episode_id: str,
        policy_query_idx: int,
        env_step: int,
        robot_state: Any,
        simulator_state: Any,
        bddl_path: str | None = None,
        qpos: Any = None,
        qvel: Any = None,
    ) -> QueryOutcome:
        if self.state is not ControllerState.VLA_RUN:
            raise ControllerStoppedError(
                f"policy inference is disabled while controller is {self.state.value}"
            )
        if str(episode_id) != self.episode_id:
            raise ValueError(
                f"query episode {episode_id!r} does not match active episode {self.episode_id!r}"
            )

        policy_result = self.policy_infer(observation)
        if "actions" not in policy_result:
            raise KeyError("π0.5 inference result must contain actions")
        if "features" not in policy_result:
            raise KeyError("π0.5 inference result must contain frozen features")
        self._last_policy_result = policy_result

        action_chunk = policy_result["actions"]
        p_help = float(self.score_help(policy_result))
        decision = self.decision_engine.evaluate(p_help)
        self._last_decision = decision
        if decision.decision == "CONTINUE":
            return QueryOutcome(
                state=ControllerState.VLA_RUN,
                decision=decision,
                actions_to_execute=action_chunk,
            )

        if self.shadow_mode:
            return QueryOutcome(
                state=ControllerState.VLA_RUN,
                decision=decision,
                actions_to_execute=action_chunk,
            )

        if (
            self._last_handoff_step is not None
            and int(env_step) - self._last_handoff_step < self.min_steps_between_calls
        ):
            remaining = self.min_steps_between_calls - (int(env_step) - self._last_handoff_step)
            decision = replace(
                decision,
                decision="CONTINUE",
                reason="call_interval_active",
                cooldown_remaining=max(0, int(remaining)),
                cooldown_suppressed=True,
            )
            self._last_decision = decision
            return QueryOutcome(
                state=ControllerState.VLA_RUN,
                decision=decision,
                actions_to_execute=action_chunk,
            )
        self.state = ControllerState.CALL_PENDING
        self._handoff_attempted = True
        self._last_handoff_step = int(env_step)
        request = HandoffRequest(
            task_instruction=str(task_instruction),
            episode_id=self.episode_id,
            suite=self.suite,
            policy_query_idx=int(policy_query_idx),
            call_step=int(env_step),
            observation=_snapshot_copy(
                environment_observation
                if environment_observation is not None
                else observation
            ),
            robot_state=_snapshot_copy(robot_state),
            simulator_state=_snapshot_copy(simulator_state),
            pi05_action=_snapshot_copy(action_chunk),
            pi05_features=_snapshot_copy(policy_result["features"]),
            p_help=decision.p_help,
            threshold=decision.threshold,
            hard_threshold=decision.hard_threshold,
            recent_scores=list(decision.recent_scores),
            reason=decision.reason,
            decision_rule=decision.decision_rule,
            bddl_path=bddl_path,
            qpos=_snapshot_copy(qpos),
            qvel=_snapshot_copy(qvel),
        )
        try:
            response = self.handoff_handler.on_call(request)
        except Exception as error:
            # A failed handoff must not resume autonomous control or emit the
            # already planned π0.5 action chunk.
            self.state = ControllerState.HANDOFF
            response = HandoffResponse(
                status="failed",
                detail=f"handoff handler failed: {type(error).__name__}: {error}",
            )
        else:
            self.state = ControllerState.HANDOFF

        return self._finish_handoff_response(
            decision, response, call_step=request.call_step
        )

    @property
    def last_policy_result(self) -> Mapping[str, Any] | None:
        """Latest π0.5 result, used only for a hard boundary handoff."""
        return self._last_policy_result

    def _finish_handoff_response(
        self,
        decision: CallDecision,
        response: HandoffResponse,
        *,
        call_step: int,
    ) -> QueryOutcome:
        self.state = ControllerState.HANDOFF
        recovery_outcome = response.recovery
        reentry_decision = None
        if recovery_outcome is not None:
            reentry_decision = self.reentry_gate.evaluate(
                recovery_outcome, call_step=call_step
            )
            if reentry_decision.allowed:
                self.state = ControllerState.RECOVERY_READY
            self._recovery_outcome = recovery_outcome
            self._reentry_decision = reentry_decision
        return QueryOutcome(
            state=self.state,
            decision=decision,
            actions_to_execute=None,
            handoff_response=response,
            recovery_outcome=recovery_outcome,
            reentry_decision=reentry_decision,
        )

    def force_handoff(
        self,
        policy_result: Mapping[str, Any],
        *,
        environment_observation: Mapping[str, Any] | None = None,
        task_instruction: str,
        episode_id: str,
        policy_query_idx: int,
        env_step: int,
        robot_state: Any,
        simulator_state: Any,
        bddl_path: str | None = None,
        qpos: Any = None,
        qvel: Any = None,
        reason: str = "max_steps_reached",
    ) -> QueryOutcome:
        """Force CALL_ASTRA at a hard episode boundary.

        This path bypasses the learned confirmation rule but preserves the
        normal snapshot, recovery, and re-entry contracts.
        """
        if self.state is not ControllerState.VLA_RUN:
            raise ControllerStoppedError(
                f"cannot force handoff while controller is {self.state.value}"
            )
        if str(episode_id) != self.episode_id:
            raise ValueError(
                f"query episode {episode_id!r} does not match active episode {self.episode_id!r}"
            )
        if "actions" not in policy_result or "features" not in policy_result:
            raise KeyError("forced handoff requires π0.5 actions and frozen features")
        if environment_observation is None:
            raise ValueError("forced handoff requires the current environment observation")
        action_chunk = policy_result["actions"]
        p_help = float(self.score_help(policy_result))
        base_decision = self._last_decision
        if base_decision is None:
            base_decision = self.decision_engine.evaluate(p_help)
        decision = replace(
            base_decision,
            p_help=p_help,
            decision="CALL_ASTRA",
            reason=str(reason),
            cooldown_suppressed=False,
        )
        self._last_decision = decision
        if self.shadow_mode:
            return QueryOutcome(
                state=ControllerState.VLA_RUN,
                decision=decision,
                actions_to_execute=action_chunk,
            )
        self.state = ControllerState.CALL_PENDING
        self._handoff_attempted = True
        self._last_handoff_step = int(env_step)
        request = HandoffRequest(
            task_instruction=str(task_instruction),
            episode_id=self.episode_id,
            suite=self.suite,
            policy_query_idx=int(policy_query_idx),
            call_step=int(env_step),
            observation=_snapshot_copy(
                environment_observation
                if environment_observation is not None
                else {}
            ),
            robot_state=_snapshot_copy(robot_state),
            simulator_state=_snapshot_copy(simulator_state),
            pi05_action=_snapshot_copy(action_chunk),
            pi05_features=_snapshot_copy(policy_result["features"]),
            p_help=decision.p_help,
            threshold=decision.threshold,
            hard_threshold=decision.hard_threshold,
            recent_scores=list(decision.recent_scores),
            reason=decision.reason,
            decision_rule=decision.decision_rule,
            bddl_path=bddl_path,
            qpos=_snapshot_copy(qpos),
            qvel=_snapshot_copy(qvel),
        )
        try:
            response = self.handoff_handler.on_call(request)
        except Exception as error:
            response = HandoffResponse(
                status="failed",
                detail=f"handoff handler failed: {type(error).__name__}: {error}",
            )
        return self._finish_handoff_response(
            decision, response, call_step=request.call_step
        )

    def resume_after_recovery(self) -> RecoveryReentryDecision:
        """Return control to π0.5 after the gate has approved the checkpoint."""
        if self.state is not ControllerState.RECOVERY_READY:
            raise ControllerStoppedError(
                f"cannot resume π0.5 while controller is {self.state.value}"
            )
        if self._reentry_decision is None or not self._reentry_decision.allowed:
            raise RuntimeError("recovery re-entry was not approved")
        self.decision_engine.reset()
        self.state = ControllerState.VLA_RUN
        return self._reentry_decision

    def finalize_episode(self) -> None:
        if self.state is ControllerState.CALL_PENDING:
            raise RuntimeError("cannot finalize while a handoff is pending")
        if self.state not in {ControllerState.VLA_RUN, ControllerState.HANDOFF}:
            raise RuntimeError(f"cannot finalize controller from {self.state.value}")
        self.state = ControllerState.TERMINATED

    def reset_episode(self, episode_id: str, *, suite: str | None = None) -> None:
        if self.state is not ControllerState.TERMINATED:
            raise RuntimeError("finalize the current episode before resetting the controller")
        self.episode_id = str(episode_id)
        if suite is not None:
            self.suite = str(suite)
        self.decision_engine.reset()
        self._handoff_attempted = False
        self._last_handoff_step = None
        self._last_decision = None
        self._last_policy_result = None
        self._recovery_outcome = None
        self._reentry_decision = None
        self.state = ControllerState.VLA_RUN
