"""Astra recovery executor bound to one existing LIBERO environment episode."""
from __future__ import annotations

import base64
from copy import deepcopy
from dataclasses import replace
from io import BytesIO
import json
import math
import os
from pathlib import Path
import re
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Mapping, Optional
from uuid import uuid4

import numpy as np
from scipy.spatial.transform import Rotation

from ..recovery import RecoveryOutcome
from .adapter import AdapterError, LiberoEefAdapter, MotionTarget
from .codex_client import (
    DEFAULT_ASTRA_MODEL,
    DEFAULT_REASONING_EFFORT,
    CodexAppServerClient,
    CodexClientError,
    _content_items,
)
from .journal import ExecutionUncertain, Journal
from .proposal import Pi05ProposalService
from .protocol import (
    IDS,
    MIN_EEF_CHUNK_ACTIONS,
    ProtocolError,
    tool_specs,
    validate_decision,
)
from .geometry import quat_wxyz_to_matrix


def _developer_instructions() -> str:
    """Load the versioned English Developer prompt used by Astra."""
    candidates = (
        Path(__file__).resolve().parents[2] / "docs" / "prompts" / "call_llm_recovery_instructions.md",
    )
    for prompt_path in candidates:
        try:
            if prompt_path.is_file():
                prompt = prompt_path.read_text(encoding="utf-8").strip()
                if prompt:
                    return prompt
        except OSError:
            continue
    return _embedded_developer_instructions()


def _embedded_developer_instructions() -> str:
    return """You are Astra, the recovery controller for one existing single-arm Panda LIBERO episode.

Rules:
 - Use only the supplied action tools. Each tool event must be one tool call; do not return prose.
 - The host has already observed the episode and computed the current pi05 proposal. Those host operations are not model tools.
 - Use only the newest IDs and proposal in the host context. If the proposal is aligned, call libero_resume_pi05; if misaligned, call the action tool exposed by the host. The optional chunk-edit and EEF-chunk schemes are enabled only by an explicit host mode.
 - If the initial recovery message says Astra owns full takeover, never call libero_resume_pi05; keep executing legal corrections until success or a clear failure.
 - For resume, provide only the compact resume decision. For execute, provide only the compact EEF delta decision.
 - In normal take-over, the host fixes execute steps=1, advances the simulator, refreshes the state, and starts the next Astra turn. In EEF chunk mode, return 30-50 one-step corrections in one chunk; the host executes the complete chunk before refreshing observation and Pi0.5.
 - A `no_execution=true` validation error changed nothing. Read its reason and correct the same decision using the current host context; do not call hidden host operations.
 - There is no model stop action. The host owns termination and ends recovery on success, safety/environment failure, or the recovery-turn limit.
 - Make small safe corrections using the supplied coordinate, ID, and action constraints. Do not invent simulator state or claim success.
 - Do not send assessment, progress, execution-status, intent-status, target, or step-count fields; the host owns those fields.
 - The host verifies success.
  """


def _context_text(context: Mapping[str, Any]) -> str:
    visible = dict(context)
    visible.pop("_images", None)
    return json.dumps(visible, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _legacy_developer_instructions() -> str:
    return """You are Astra, the recovery controller for one existing single-arm Panda LIBERO episode.

Control protocol:
- Use only the supplied LIBERO dynamic tools. Never use shell, files, another simulator, a new environment, or an alternate controller.
- Every assistant turn in this recovery session must contain exactly one supplied dynamic tool call. Do not answer with prose, an empty message, markdown, or a claim that the task is complete; the host is the only component that can verify completion.
- For every decision tool call, fill the top-level `reason` field with a concise 1-2 sentence operational rationale. This is an audit note, not hidden chain-of-thought: never substitute private reasoning for the required tool call.
- First call libero_observe, then pi05_propose. After an accepted correction, the host only advances the simulator; Astra must call observe and then pi05_propose before making the next decision.
- Every libero_observe response replaces the prior request_id, observation_id, and proposal_id. After any new observation, discard all older identifiers and copy only the newest IDs into the next pi05_propose, libero_execute_eef, libero_resume_pi05, or libero_stop call.
- Observation images are the only evidence about objects and task progress. Robot pose/history are measured. Do not infer hidden object poses, reward, simulator truth, or task success.
- The current EEF is the Panda grip control site in world coordinates: position in meters and quaternion_wxyz. Relative rotation vectors are world-frame radians.
- The pi05 proposal is unexecuted context only. It is supplied as a dimensionful nominal control-site trajectory calculated from the live OSC_POSE scale; it is not an action to copy or execute.
- A CALL intervention means the host detected that pi05 needs help. Treat the task as unfinished unless the host has verified success. If the target is visible and the scene is safe, make a small legal eef_delta correction and re-observe; do not end the intervention merely because the first observation or proposal is incomplete.
- Astra is a temporary recovery supervisor, not the long-running task policy. When pi05 is wrong, make only the minimum correction needed to realign its next intent, then hand control back with libero_resume_pi05. Do not replace pi05 for the whole task and do not stop merely because pi05 was initially misaligned.
- Request a correction with eef_delta only when visual execution evidence or proposal-intent comparison establishes failure/misalignment. Output a world-frame position delta in meters and a world-frame rotation vector in radians; never output LIBERO normalized 7D actions or an absolute eef target.
- steps is always 1 for `libero_execute_eef`: one LIBERO env.step control call, not MuJoCo substeps. The host converts the EEF delta to one native action, checks all bounds without clipping, and rereads measured pose after the step. A requested target or controller ACK is not evidence that the target was reached.
- Every eef_delta position component must be in [-0.05, 0.05] m and the 3D position-delta norm must be at most 0.05 m. Every rotation-vector component must be in [-0.35, 0.35] rad and its norm must be at most 0.35 rad. Treat these as output constraints: choose an in-range correction before calling the tool; never rely on the host to clip or repair an out-of-range command. The host then converts that legal target into one native 7D action, strictly within LIBERO's [-1, 1] action bounds. Each execute call advances exactly one LIBERO environment step. If more motion is needed, wait for the next host-refreshed observation and submit another in-range correction.
- Gripper meanings: keep preserves the current command; open opens; closed closes. Closing is not proof of grasp. Use the images and measured history to assess whether the object followed.
- Never declare simulator task success. The host checks LIBERO's success condition privately and will stop the intervention if it is met.

Assessment and handback:
- Before handback, use the newest host-supplied observation and pi05 proposal. Call libero_resume_pi05 only when that proposal is aligned with the current visual subgoal.
- After a correction, the host owns the refresh path: it observes, obtains a fresh pi05 proposal, and returns both to Astra. Resume pi05 as soon as that host-supplied proposal is visually aligned. The host will independently verify the checkpoint and task state.
- Immediately after a fresh pi05 proposal, choose only one of two actions: call libero_resume_pi05 when the proposal is visually aligned, or call libero_execute_eef when it is misaligned. After execute, call observe and pi05_propose yourself; the host does not supply a refreshed proposal automatically.
- One Responses turn may contain multiple tool events, but only the first action tool call in that turn can be current. After any accepted execute, never emit another execute/resume/stop from the same turn, even if the host result appears to contain another action opportunity; wait for the host to start the next Astra turn with the refreshed observation and proposal. Every execute call advances exactly one simulator step.
- If the host reports `stale_action_same_responses_turn`, the previous action already executed and the duplicate call was not executed again. Call libero_observe now, then pi05_propose, and choose the next action from the new state in the same recovery cycle.
- If libero_execute_eef returns a correctable error such as target_requires_more_steps_than_requested, treat it as no execution, read the required limit, and retry with a smaller legal target; do not observe first because the environment did not change.
- If libero_resume_pi05 returns a correctable error, treat it as no handback and use the returned reason to decide whether to correct the call or obtain a fresh proposal. Never terminate while a fresh proposal is pending.
- A resume tool call is a request, not authority: the host independently validates episode/intervention/observation/proposal IDs, exact synchronized checkpoint, finite robot state, and task status.
- Call libero_stop only for an explicit safety/environment failure or when no safe recovery is possible after observing the current scene. Do not stop merely because state is incomplete, because the pi05 proposal is unexecuted, or because the first correction still needs feedback. Do not claim that a stopped or timed-out episode is a natural task failure.
- Return the required compact tool call. Do not emit prose in place of a tool call.

Concrete tool chains:
- Aligned handback: libero_observe -> pi05_propose -> libero_resume_pi05. In the resume call, use the newest IDs and a reason explaining the visual alignment.
- Misaligned recovery: libero_observe -> pi05_propose -> libero_execute_eef -> [host observe + host pi05_propose] -> libero_resume_pi05. After an accepted eef correction, use the host-supplied fresh proposal; do not hand back using the old proposal.
- Rejected correction: libero_observe -> pi05_propose -> libero_execute_eef (rejected) -> retry libero_execute_eef with a smaller legal target using the current identifiers. The rejected call changed nothing; do not observe merely because the correction was rejected.
- Rejected handback: libero_observe -> pi05_propose -> libero_resume_pi05 (rejected) -> libero_observe -> pi05_propose -> libero_resume_pi05 or libero_execute_eef. The rejected handback is not evidence of task failure; obtain a fresh proposal first.
- Never stop or return prose in the middle of any chain. Each arrow means the preceding tool call returned successfully, except where explicitly marked rejected.
"""


class RecoveryOrchestrator:
    """Host-side recovery architecture around a model and LIBERO tools.

    Astra is the selectable model name.  This class owns the recovery
    lifecycle, state machine, validation, and tool dispatch; it is not the
    model itself.
    """

    def __init__(
        self,
        *,
        env: Any,
        broker: Any,
        adapter: LiberoEefAdapter,
        history: Any,
        proposal_service: Pi05ProposalService,
        observation_encoder: Callable[[Mapping[str, Any], str], Mapping[str, Any]],
        robot_state_encoder: Callable[[Mapping[str, Any]], np.ndarray],
        audit_path: str,
        workspace: str,
        model: str = DEFAULT_ASTRA_MODEL,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
        max_total_steps: Optional[int] = None,
        episode_step_limit: Optional[int] = None,
        max_decisions: int = 25,
        max_recovery_execute_turns: int = 25,
        max_tool_calls: int = 100,
        max_invalid_calls: int = 3,
        max_wall_seconds: float = 300.0,
        max_transient_retries: int = 3,
        max_idle_retries: Optional[int] = None,
        chunk_edit_enabled: bool = False,
        chunk_mode_enabled: bool = False,
    ) -> None:
        self.env = env
        self.broker = broker
        self.adapter = adapter
        self.history = history
        self.proposal_service = proposal_service
        self.observation_encoder = observation_encoder
        self.robot_state_encoder = robot_state_encoder
        self.audit_path = str(audit_path)
        self.workspace = str(workspace)
        self.model = str(model)
        self.reasoning_effort = str(reasoning_effort)
        self.max_total_steps = (
            None if max_total_steps is None else int(max_total_steps)
        )
        self.episode_step_limit = (
            None if episode_step_limit is None else int(episode_step_limit)
        )
        self.max_decisions = int(max_decisions)
        self.max_recovery_execute_turns = int(max_recovery_execute_turns)
        self.max_tool_calls = int(max_tool_calls)
        self.max_invalid_calls = int(max_invalid_calls)
        self.max_wall_seconds = float(max_wall_seconds)
        # Keep the old keyword as an internal compatibility alias while the
        # retry budget now also covers response-stream disconnects.
        if max_idle_retries is not None:
            max_transient_retries = int(max_idle_retries)
        self.max_transient_retries = int(max_transient_retries)
        self.max_idle_retries = self.max_transient_retries
        # The established take-over controller remains the default.  Chunk
        # editing is an explicit alternative experiment, not a replacement.
        self.chunk_edit_enabled = bool(chunk_edit_enabled)
        # Astra-owned action chunks are a separate experiment from both the
        # established one-step take-over and Pi0.5 waypoint editing.
        self.chunk_mode_enabled = bool(chunk_mode_enabled)
        if self.chunk_edit_enabled and self.chunk_mode_enabled:
            raise ValueError("chunk_edit_and_eef_chunk_modes_are_mutually_exclusive")
        limits = (self.max_decisions, self.max_recovery_execute_turns,
                  self.max_tool_calls, self.max_invalid_calls)
        if (min(limits) < 1 or
                (self.max_total_steps is not None and self.max_total_steps < 1) or
                (self.episode_step_limit is not None and self.episode_step_limit < 1) or
                self.max_wall_seconds <= 0 or self.max_transient_retries < 0):
            raise ValueError("invalid_Astra_intervention_budgets")
        self.request: Any = None
        self.intervention_id = ""
        self._request_id: Optional[str] = None
        self._request_observation_id: Optional[str] = None
        self._awaiting_proposal_decision = False
        # Host-owned recovery sequencing.  After a correction, Astra must
        # inspect the new simulator state and obtain a fresh pi05 proposal
        # before it can execute again or hand control back.
        self._awaiting_observation_after_execute = False
        self._tool_calls = 0
        self._accepted_decisions = 0
        self._recovery_execute_turns = 0
        self._invalid_calls = 0
        self._starting_recovery_model_steps = 0
        self._terminal_outcome: Optional[RecoveryOutcome] = None
        self._client: Optional[Any] = None
        self._model_response_received = False
        self._codex_cli_version = "unknown"
        self._intervention_start_step = 0
        self._started_at = 0.0
        self._workspace_started = False
        self._provider_transport_summary: Optional[dict[str, Any]] = None
        self._takeover_mode = False
        self._last_host_context: Optional[Dict[str, Any]] = None
        self._last_context_env_step = 0
        self._last_context_recovery_steps = 0

    def enable_full_takeover(self) -> None:
        """Keep Astra in control after π0.5 exhausts its recovery phase."""
        self._takeover_mode = True
        try:
            self.broker.journal.append({
                "event": "astra_full_takeover_enabled",
                "reason": "pi05_steps_exhausted_after_recovery",
            })
        except Exception:
            pass

    def enable_chunk_edit_mode(self) -> None:
        """Opt into the Pi0.5-chunk editing alternative."""
        self.chunk_edit_enabled = True
        try:
            self.broker.journal.append({
                "event": "astra_chunk_edit_mode_enabled",
                "reason": "explicit_alternative_control_scheme",
            })
        except Exception:
            pass

    def enable_chunk_mode(self) -> None:
        """Opt into Astra-generated multi-step EEF chunks."""
        if self.chunk_edit_enabled:
            raise ValueError("chunk_edit_and_eef_chunk_modes_are_mutually_exclusive")
        self.chunk_mode_enabled = True
        try:
            self.broker.journal.append({
                "event": "astra_eef_chunk_mode_enabled",
                "reason": "explicit_alternative_control_scheme",
            })
        except Exception:
            pass

    def intervene(self, request: Any, snapshot_path: str | None) -> RecoveryOutcome:
        del snapshot_path  # private snapshot never enters the model workspace or prompt
        self.request = request
        # Every CALL starts a fresh model-driven graph.  No observation or
        # proposal from a previous intervention may remain pending.
        self._request_id = None
        self._request_observation_id = None
        self._awaiting_proposal_decision = False
        self._awaiting_observation_after_execute = False
        self._tool_calls = 0
        self._accepted_decisions = 0
        self._invalid_calls = 0
        self._terminal_outcome = None
        self._provider_transport_summary = None
        self._last_host_context = None
        self._last_context_env_step = int(request.call_step)
        self._last_context_recovery_steps = int(self.broker.recovery_model_total_steps)
        if not request.intervention_id or not request.observation_id:
            return self._outcome("unsafe", "handoff_identity_missing", can_resume=False)
        self.intervention_id = str(request.intervention_id)
        self._intervention_start_step = int(request.call_step)
        self._starting_recovery_model_steps = int(self.broker.recovery_model_total_steps)
        self._started_at = time.monotonic()
        try:
            checkpoint = self.broker.current_checkpoint()
            if checkpoint.episode_id != str(request.episode_id):
                return self._outcome("unsafe", "episode_identity_mismatch", can_resume=False)
            if checkpoint.env_step != int(request.call_step):
                return self._outcome("unsafe", "call_step_mismatch", can_resume=False)
            if checkpoint.observation_id != str(request.observation_id):
                return self._outcome("unsafe", "call_observation_mismatch", can_resume=False)
            if self.broker.owner != "astra" or self.broker.uncertain:
                return self._outcome("unsafe", "astra_control_owner_not_confirmed", can_resume=False)
            workspace = self.adapter.begin_intervention(checkpoint.raw)
            self._workspace_started = True
            self.broker.journal.append({
                "event": "astra_intervention_started",
                "episode_id": str(request.episode_id),
                "intervention_id": self.intervention_id,
                "call_step": int(request.call_step),
                "observation_id": str(checkpoint.observation_id),
                "workspace_guard": str(workspace.get("workspace_guard", "disabled"))
                if isinstance(workspace, Mapping) else "disabled",
            })
            self.proposal_service.install_call_proposal(checkpoint, request.pi05_action)
            if self._takeover_mode and self.chunk_mode_enabled:
                allowed_action_tools = {"libero_execute_eef_chunk"}
            elif self._takeover_mode:
                allowed_action_tools = {"libero_execute_eef"}
            elif self.chunk_mode_enabled:
                allowed_action_tools = {"libero_execute_eef_chunk", "libero_resume_pi05"}
            elif self.chunk_edit_enabled:
                allowed_action_tools = {"libero_edit_pi05_chunk", "libero_resume_pi05"}
            else:
                allowed_action_tools = {"libero_execute_eef", "libero_resume_pi05"}
            client_kwargs = {
                "workspace": self.workspace,
                "model": self.model,
                "effort": self.reasoning_effort,
                "developer_instructions": _developer_instructions(),
                "dynamic_tools": [
                    spec for spec in tool_specs()
                    if spec["name"] in allowed_action_tools
                ],
                "max_wall_seconds": self.max_wall_seconds,
            }
            # All supported Astra model variants use the same Codex app-server
            # Responses transport and the same tool/state-machine loop. The
            # model name is the only model-specific runtime choice.
            initial_context = self._refresh_host_context()
            transient_retry_count = 0
            context = initial_context
            retry_note = ""
            while True:
                self._client = CodexAppServerClient(
                    **client_kwargs,
                    use_provider_relay=True,
                )
                try:
                    self._client.start()
                    self._codex_cli_version = self._client.cli_version
                    result = self._client.run(
                        self._initial_prompt() + retry_note +
                        "\n\nCURRENT HOST CONTEXT JSON:\n" +
                        _context_text(context),
                        self._dispatch,
                        initial_input_items=_content_items(context),
                        # High-reasoning Responses turns can take longer after a large
                        # observe/proposal payload. Keep a per-response idle bound.
                        idle_timeout_seconds=min(240.0, self.max_wall_seconds),
                        max_turns_without_terminal_tool=10,
                    )
                    self._model_response_received = bool(
                        self._model_response_received or
                        result.get("model_response_received")
                    )
                    self._codex_cli_version = str(
                        result.get("cli_version", self._codex_cli_version)
                    )
                    if self._terminal_outcome is not None:
                        return self._terminal_outcome
                    return self._outcome(
                        "failed", "model_ended_without_host_terminal_decision",
                        can_resume=False,
                    )
                except CodexClientError as error:
                    model_note = getattr(error, "model_message", None)
                    if not model_note and self._client is not None:
                        model_note = getattr(self._client, "last_model_note", None)
                    if isinstance(model_note, str) and model_note.strip():
                        self._audit_tool_event(
                            "recovery_model_decision_note",
                            tool="libero_observe",
                            call_index=int(self._tool_calls),
                            note=model_note.strip()[:4000],
                            reason="model_returned_visible_text_without_a_terminal_tool_call",
                        )
                    self._model_response_received = bool(
                        self._model_response_received or
                        error.model_response_received or
                        (self._client and self._client.model_response_received)
                    )
                    if self._terminal_outcome is not None:
                        return self._terminal_outcome
                    retry_kind = _retryable_model_transport_error(error.code)
                    if (
                        retry_kind is not None and
                        transient_retry_count < self.max_transient_retries
                    ):
                        transient_retry_count += 1
                        current_checkpoint = self.broker.current_checkpoint()
                        current_env_step = int(current_checkpoint.env_step)
                        current_recovery_steps = int(
                            self.broker.recovery_model_total_steps
                        )
                        context_advanced = (
                            current_env_step > self._last_context_env_step or
                            current_recovery_steps > self._last_context_recovery_steps
                        )
                        if context_advanced:
                            # A previously accepted action advanced the simulator.
                            # Refresh exactly once through the host-owned path; never
                            # replay the old action or its IDs.
                            context = self._refresh_host_context()
                            retry_note = (
                                "\n\nThe previous Astra Responses attempt ended with "
                                f"a transient {retry_kind} after the simulator advanced. "
                                "The host refreshed the current observation and pi05 "
                                "proposal below. Do not replay the previous action; "
                                "choose one action from this current context."
                            )
                        else:
                            # No accepted action changed the simulator. Retrying the
                            # same host context is intentional; a new observation would
                            # add no information and would only rotate valid IDs.
                            context = dict(self._last_host_context or context)
                            retry_note = (
                                "\n\nThe previous Astra Responses attempt ended with "
                                f"a transient {retry_kind} before any action was accepted. "
                                "The simulator did not change, so use this same host "
                                "context and make one current action call."
                            )
                        retry_event = (
                            "model_idle_timeout_retry"
                            if retry_kind == "model_idle_timeout"
                            else "model_response_stream_retry"
                        )
                        self._audit_tool_event(
                            retry_event,
                            tool="libero_observe",
                            retry_index=int(transient_retry_count),
                            max_retries=int(self.max_transient_retries),
                            failure_kind=retry_kind,
                            context_advanced=bool(context_advanced),
                            context_reused=bool(not context_advanced),
                            env_step=current_env_step,
                            recovery_model_steps=current_recovery_steps,
                            reason=(
                                "refresh_host_context_after_accepted_action"
                                if context_advanced else
                                "reuse_unchanged_context_after_idle_timeout"
                            ),
                        )
                        self._close_client()
                        continue
                    if retry_kind is not None:
                        retries_exhausted_reason = (
                            "model_idle_timeout_retries_exhausted"
                            if retry_kind == "model_idle_timeout"
                            else "response_stream_disconnected_retries_exhausted"
                        )
                        self._audit_tool_event(
                            "model_transient_retries_exhausted",
                            tool="libero_observe",
                            failure_kind=retry_kind,
                            retry_count=int(transient_retry_count),
                            max_retries=int(self.max_transient_retries),
                            env_step=int(self.broker.env_steps),
                            recovery_model_steps=int(self.broker.recovery_model_total_steps),
                        )
                        status = "timeout" if retry_kind == "model_idle_timeout" else "failed"
                        return self._outcome(
                            status, retries_exhausted_reason, can_resume=False
                        )
                    if error.code in (
                        "model_idle_timeout", "intervention_wall_timeout",
                        "codex_rpc_timeout",
                    ):
                        return self._outcome("timeout", error.code, can_resume=False)
                    return self._outcome("failed", error.code, can_resume=False)
        except ExecutionUncertain:
            return self._outcome("unsafe", "environment_step_ack_uncertain", can_resume=False,
                                 execution_uncertain=True)
        except Exception as error:
            if self.broker.uncertain or self.broker.owner == "stopped":
                self._record_host_failure_diagnostic(error, phase="intervene")
                return self._outcome("unsafe", "host_step_or_audit_failure", can_resume=False,
                                     execution_uncertain=bool(self.broker.uncertain))
            return self._outcome("failed", "executor_error_" + type(error).__name__, can_resume=False)
        finally:
            self._close_client()
            if self._workspace_started:
                self.adapter.end_intervention()
                self._workspace_started = False

    def _initial_prompt(self) -> str:
        if getattr(self, "_takeover_mode", False) and getattr(self, "chunk_mode_enabled", False):
            action_instruction = (
                "This is Astra EEF chunk mode with full take-over. Choose exactly one current action: "
                "libero_execute_eef_chunk. Return 30-50 one-step EEF corrections. The host executes "
                "the complete chunk without an intermediate observation or Pi0.5 proposal."
            )
        elif getattr(self, "_takeover_mode", False):
            action_instruction = (
                "This is the established Astra take-over scheme. Choose exactly one current action: "
                "libero_execute_eef. Pi0.5 has exhausted its post-recovery step budget; continue legal "
                "single-step corrections until the host verifies success or a clear failure."
            )
        elif getattr(self, "chunk_mode_enabled", False):
            action_instruction = (
                "This is Astra EEF chunk mode. Choose exactly one current action: "
                "libero_execute_eef_chunk or libero_resume_pi05. The chunk must contain "
                "30-50 one-step EEF corrections. The host executes the complete chunk "
                "without an intermediate observation or Pi0.5 proposal, then starts a "
                "new turn with refreshed state."
            )
        elif getattr(self, "chunk_edit_enabled", False):
            action_instruction = (
                "This is the optional Pi0.5-chunk editing scheme. Choose exactly one current action: "
                "libero_edit_pi05_chunk or libero_resume_pi05. After a chunk edit, wait for the host "
                "to start a new turn with refreshed state."
            )
        else:
            action_instruction = (
                "Choose exactly one current action: libero_execute_eef or libero_resume_pi05. "
                "After an accepted correction, wait for the host's refreshed turn."
            )
        takeover_instruction = (
            " Do not call libero_resume_pi05 while full take-over is active."
            if getattr(self, "_takeover_mode", False) else ""
        )
        total_step_limit = getattr(self, "max_total_steps", None)
        step_budget_instruction = (
            " There is no separate fixed total Astra control-step cap; the episode-level "
            "LIBERO max-step limit and intervention decision/chunk limits still apply."
            if total_step_limit is None else
            f" Astra has at most {int(total_step_limit)} total control steps in this intervention."
        )
        decision_budget_instruction = (
            f" Intervention limits: at most {getattr(self, 'max_decisions', 25)} accepted "
            f"decisions and {getattr(self, 'max_recovery_execute_turns', 25)} executed "
            "recovery chunks."
        )
        episode_step_limit = getattr(self, "episode_step_limit", None)
        episode_budget_instruction = (
            ""
            if episode_step_limit is None else
            f" The host enforces a hard episode-wide limit of {int(episode_step_limit)} "
            "environment steps. Check episode_budget.remaining_steps in every refreshed "
            "context; never request an action or chunk that cannot finish within that budget. "
            "For an EEF chunk, every listed action consumes one environment step."
        )
        return (
            "Begin recovery for this one existing episode. The host has already performed "
            "the current observation and pi05 proposal and supplied their complete result "
            "in the read-only workspace context file referenced by this turn. Use the "
            "built-in read-only workspace/file/image tools to inspect that latest context "
            "before acting; old conversation content is not authoritative. Observation "
            "and proposal tools are host-internal and are not available to you. "
            + action_instruction + " The host owns termination and ends recovery on success, "
            "safety/environment failure, or the recovery-turn limit."
            + decision_budget_instruction + step_budget_instruction
            + episode_budget_instruction
            + takeover_instruction
        )

    def _refresh_host_context(self) -> Dict[str, Any]:
        """Run observe -> pi05_propose inside the host, not through model tools."""
        observation = self._observe({
            "episode_id": str(self.request.episode_id),
            "intervention_id": self.intervention_id,
        })
        proposal = self._propose({
            "request_id": str(observation["request_id"]),
            "observation_id": str(observation["observation_id"]),
        })
        context = dict(observation)
        context.update(proposal)
        if self.chunk_mode_enabled:
            context["eef_chunk_contract"] = {
                "enabled": True,
                "minimum_actions": 30,
                "maximum_actions": 50,
                "one_action_equals_one_environment_step": True,
                "intermediate_observation": False,
                "intermediate_pi05_proposal": False,
                "execution_semantics": (
                    "The host executes the submitted actions sequentially. It does not send "
                    "a new observation or Pi0.5 proposal to Astra until the complete chunk "
                    "has finished; then it refreshes both for the next turn."
                ),
            }
        elif self.chunk_edit_enabled and not self._takeover_mode:
            context["chunk_edit_contract"] = {
                "enabled": True,
                "max_execute_steps": int(proposal.get("control_steps", 0)),
                "waypoint_edit_semantics": (
                    "Each waypoint edit is a position/rotation offset relative to the "
                    "corresponding nominal Pi0.5 waypoint; omitted waypoints remain nominal."
                ),
                "execution_semantics": (
                    "The host executes the edited prefix with one LIBERO environment "
                    "step per waypoint, then refreshes observation and proposal."
                ),
            }
        if self._takeover_mode and self.chunk_mode_enabled:
            context["host_workflow"] = (
                "Astra owns the episode in EEF chunk mode. The host completed observe -> "
                "pi05_propose. Choose exactly one action: libero_execute_eef_chunk. "
                "The host executes 30-50 one-step corrections with no intermediate "
                "observation or Pi0.5 proposal, then refreshes state for the next turn."
            )
            context["takeover_mode"] = "astra_full_takeover_eef_chunk"
        elif self._takeover_mode:
            context["host_workflow"] = (
                "Pi0.5 has exhausted its post-recovery step budget. Astra owns the "
                "episode. The host completed observe -> pi05_propose; choose exactly "
                "one action: libero_execute_eef. Never call "
                "libero_resume_pi05 during full takeover. After execute, the host "
                "refreshes state and starts the next turn automatically."
            )
            context["takeover_mode"] = "astra_full_takeover"
        elif self.chunk_mode_enabled:
            context["host_workflow"] = (
                "Host already completed observe -> pi05_propose. Choose exactly one action: "
                "libero_execute_eef_chunk or libero_resume_pi05. A chunk contains 30-50 "
                "one-step EEF corrections. The host does not observe or run Pi0.5 between "
                "those actions; it refreshes only after the chunk finishes."
            )
        elif self.chunk_edit_enabled:
            context["host_workflow"] = (
                "Host already completed observe -> pi05_propose. Choose exactly one "
                "action: libero_edit_pi05_chunk or libero_resume_pi05. The host owns termination. After a chunk edit, the "
                "host refreshes state and starts the next turn automatically."
            )
        else:
            context["host_workflow"] = (
                "Host already completed observe -> pi05_propose. Choose exactly one "
                "action: libero_execute_eef or libero_resume_pi05. The host owns termination. "
                "After an accepted execute, the host refreshes state and starts the next turn automatically."
            )
        context["proposal_status"] = "host_provided_current_proposal_choose_action"
        images = observation.get("_images") or proposal.get("_images")
        if images:
            context["_images"] = images
        context = self._publish_workspace_context(context)
        self._last_host_context = dict(context)
        self._last_context_env_step = int(
            context.get("env_step", self.broker.current_checkpoint().env_step)
        )
        self._last_context_recovery_steps = int(self.broker.recovery_model_total_steps)
        self._awaiting_observation_after_execute = False
        self._awaiting_proposal_decision = True
        return context

    def _publish_workspace_context(self, context: Mapping[str, Any]) -> Dict[str, Any]:
        """Persist the full current state and return a compact turn packet.

        The app-server workspace is read-only from Astra's perspective.  The
        model receives only a pointer and small identity/pose fields in the
        turn; the full proposal, history, and current RGB frames remain in the
        workspace for selective read-only inspection.
        """
        workspace = Path(self.workspace)
        workspace.mkdir(parents=True, exist_ok=True)
        stored = dict(context)
        images = stored.pop("_images", []) or []
        image_refs: list[dict[str, str]] = []
        for index, image in enumerate(images):
            if not isinstance(image, Mapping):
                continue
            data_url = image.get("data_url")
            if not isinstance(data_url, str) or "," not in data_url:
                continue
            try:
                payload = data_url.split(",", 1)[1]
                image_bytes = base64.b64decode(payload, validate=True)
            except (ValueError, TypeError):
                continue
            label = str(image.get("label", "RGB observation"))
            suffix = "external_rgb" if "external" in label.lower() else "wrist_rgb"
            image_path = workspace / f"latest_{suffix}_{index}.png"
            image_path.write_bytes(image_bytes)
            image_refs.append({"label": label, "path": str(image_path)})
        stored["workspace_context_version"] = "v1"
        stored["observation_files"] = image_refs
        context_path = workspace / "latest_context.json"
        stored["workspace_context_file"] = str(context_path)
        temporary = context_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(stored, ensure_ascii=False, separators=(",", ":"), allow_nan=False),
            encoding="utf-8",
        )
        temporary.replace(context_path)

        compact = dict(stored)
        compact.pop("recent_measured_robot_history", None)
        compact.pop("nominal_trajectory", None)
        compact["workspace_read_rule"] = (
            "Read workspace_context_file and the observation_files with read-only "
            "workspace tools before choosing the action; old turn text is not authoritative."
        )
        compact["next_call"] = (
            "libero_execute_eef"
            if self._takeover_mode
            else "libero_execute_eef_or_libero_resume_pi05"
        )
        # Keep the camera frames on the private turn packet as real image
        # inputs.  They must not be placed in latest_context.json or in the
        # text context, but _content_items() needs them to build the next
        # Responses turn.  Previously _images was popped above and never
        # restored, so Astra received only observation file paths and zero
        # image items.
        if images:
            compact["_images"] = images
        return compact

    def _close_client(self) -> None:
        """Close one Responses client while preserving transport diagnostics."""
        if self._client is None:
            return
        client = self._client
        self._model_response_received = bool(
            self._model_response_received or client.model_response_received
        )
        self._codex_cli_version = client.cli_version
        transport_summary = client.provider_transport_summary
        self._provider_transport_summary = transport_summary
        if transport_summary is not None:
            try:
                self.broker.journal.append({
                    "event": "model_provider_transport",
                    "transport": "responses_via_codex_app_server",
                    "summary": transport_summary,
                })
            except Exception:
                # Cleanup must still stop both Codex and the relay.
                pass
        try:
            client.close()
        finally:
            self._client = None

    def _audit_tool_event(self, event: str, *, tool: str, **fields: Any) -> None:
        known_tools = {item["name"] for item in tool_specs()}
        safe_tool = tool if tool in known_tools else "unknown"
        record = {"event": event, "tool": safe_tool, **fields}
        try:
            self.broker.journal.append(record)
        except Exception:
            # Tool audit failure must not cause an unverified action to run;
            # the existing step journal remains the execution safety boundary.
            pass

    def _dispatch(self, name: str, args: Mapping[str, Any]) -> Mapping[str, Any]:
        self._tool_calls += 1
        self._audit_tool_event("astra_tool_call_received", tool=name,
                               call_index=int(self._tool_calls))
        if self._tool_calls > self.max_tool_calls:
            return self._terminate("budget_exhausted", "max_tool_calls_reached")
        try:
            if self._awaiting_observation_after_execute and name != "libero_observe":
                raise ProtocolError("observe_required_after_eef_correction")
            if name == "libero_observe" and self._awaiting_proposal_decision:
                raise ProtocolError("observe_not_allowed_after_fresh_pi05_proposal")
            if name == "pi05_propose" and self._awaiting_proposal_decision:
                raise ProtocolError("pi05_proposal_already_pending")
            if name == "libero_observe":
                result = self._observe(args)
            elif name == "pi05_propose":
                result = self._propose(args)
            elif name == "libero_execute_eef":
                result = self._execute(args)
            elif name == "libero_execute_eef_chunk":
                result = self._execute_eef_chunk(args)
            elif name == "libero_edit_pi05_chunk":
                result = self._edit_chunk(args)
            elif name == "libero_resume_pi05":
                result = self._resume(args)
            else:
                raise ProtocolError("unknown_tool")
            audit_fields: dict[str, Any] = {
                "call_index": int(self._tool_calls),
                "terminal": bool(result.get("_terminal", False)),
            }
            if name in ("libero_execute_eef", "libero_execute_eef_chunk",
                        "libero_edit_pi05_chunk", "libero_resume_pi05"):
                reason = args.get("reason")
                if isinstance(reason, str) and reason.strip():
                    audit_fields["reason"] = reason.strip()[:1200]
                steps = args.get("steps")
                if type(steps) is int and 0 <= steps <= 5:
                    audit_fields["steps"] = steps
                mode = args.get("mode")
                if mode in {"eef_delta", "eef_chunk", "edit_pi05_chunk", "resume_pi05"}:
                    audit_fields["mode"] = mode
            self._audit_tool_event("astra_tool_call_accepted", tool=name,
                                   **audit_fields)
            return result
        except Exception as error:
            if isinstance(error, ExecutionUncertain):
                return self._terminate("unsafe", "decision_execution_state_uncertain", uncertain=True)
            if self.broker.owner == "stopped":
                self._record_host_failure_diagnostic(
                    error,
                    phase="tool_dispatch",
                    tool=name,
                    tool_arguments=args,
                )
                return self._terminate(
                    "unsafe", "host_step_or_audit_failure",
                    uncertain=bool(self.broker.uncertain),
                )
            if self.broker.uncertain:
                return self._terminate("unsafe", "environment_step_ack_uncertain", uncertain=True)
            self._invalid_calls += 1
            code = getattr(error, "code", None) or (error.args[0] if error.args else "tool_rejected")
            if name == "libero_resume_pi05":
                # A rejected handback invalidates the pending proposal decision;
                # permit the mandated observe -> fresh proposal recovery path.
                self._awaiting_proposal_decision = False
            self._audit_tool_event(
                "astra_tool_call_rejected", tool=name,
                call_index=int(self._tool_calls), error_code=_safe_tool_error_code(code),
                rejection_index=int(self._invalid_calls),
            )
            if self._invalid_calls >= self.max_invalid_calls:
                return self._terminate("failed", "invalid_tool_budget_exhausted")
            retry_packet = {
                "error": str(code),
                "retryable": True,
                "no_execution": True,
                "instruction": (
                    "The call was rejected without changing the environment. Correct the same action using the "
                    "current host-supplied IDs and proposal. The host owns observation and proposal refresh; "
                    "do not try to call a host-internal workflow operation."
                ),
            }
            if self._request_id is not None:
                retry_packet["current_request_id"] = self._request_id
            if self._request_observation_id is not None:
                retry_packet["current_observation_id"] = self._request_observation_id
            return retry_packet

    def _observe(self, args: Mapping[str, Any]) -> Mapping[str, Any]:
        _exact_args(args, ("episode_id", "intervention_id"))
        if args["episode_id"] != str(self.request.episode_id) or args["intervention_id"] != self.intervention_id:
            raise ProtocolError("wrong_episode_or_intervention")
        checkpoint = self.broker.current_checkpoint()
        if checkpoint.episode_id != str(self.request.episode_id):
            raise ProtocolError("wrong_episode_checkpoint")
        self._request_id = "request-" + str(uuid4())
        self._request_observation_id = str(checkpoint.observation_id)
        self._awaiting_observation_after_execute = False
        self._awaiting_proposal_decision = False
        pose = self.adapter.read_pose(checkpoint.raw)
        gripper = _public_gripper(checkpoint.raw)
        packet = {
            "episode_id": str(self.request.episode_id),
            "intervention_id": self.intervention_id,
            "request_id": self._request_id,
            "observation_id": str(checkpoint.observation_id),
            "env_step": int(checkpoint.env_step),
            "task_instruction": str(self.request.task_instruction),
            "current_eef": {
                "frame": "world",
                "reference": "control_site",
                "position_m": pose.position_m.tolist(),
                "quaternion_wxyz": pose.quaternion_wxyz.tolist(),
            },
            "gripper_qpos": gripper,
            "recent_measured_robot_history": self.history.recent(),
            "episode_budget": {
                "step_limit": getattr(self, "episode_step_limit", None),
                "remaining_steps": self._remaining_episode_steps(),
            },
            "control_limits": {
                "max_target_position_norm_m": self.adapter.max_target_position_m,
                "max_target_rotation_norm_rad": self.adapter.max_target_rotation_rad,
                "max_control_step_position_norm_m": self.adapter.max_step_position_m,
                "max_control_step_rotation_norm_rad": self.adapter.max_step_rotation_rad,
                "position_tolerance_m": self.adapter.position_tolerance_m,
                "rotation_tolerance_rad": self.adapter.rotation_tolerance_rad,
                "max_control_steps_per_command": 1,
                "intervention_step_limit": self.max_total_steps,
                "remaining_intervention_steps": self._remaining_intervention_steps(),
                "remaining_decisions": max(0, self.max_decisions - self._accepted_decisions),
                "remaining_recovery_chunks": max(
                    0, self.max_recovery_execute_turns - self._recovery_execute_turns
                ),
                "chunk_action_count_min": 30 if self.chunk_mode_enabled else None,
                "chunk_action_count_max": 50 if self.chunk_mode_enabled else None,
            },
            "proposal_status": "call_pi05_propose_before_a_decision",
            "id_policy": "MANDATORY: these request_id/observation_id values replace all previous IDs; copy only these exact values into the next tool call",
            "_images": _public_images(checkpoint.raw),
        }
        return packet

    def _propose(self, args: Mapping[str, Any]) -> Mapping[str, Any]:
        _exact_args(args, ("request_id", "observation_id"))
        checkpoint = self.broker.current_checkpoint()
        self._require_current_request(args, checkpoint)
        proposal = self.proposal_service.propose(checkpoint, str(self.request.task_instruction))
        if proposal.observation_id != checkpoint.observation_id or proposal.env_step != checkpoint.env_step:
            raise ProtocolError("stale_pi05_proposal")
        self._awaiting_proposal_decision = True
        return {
            "request_id": self._request_id,
            **proposal.public_packet(),
            "id_policy": "MANDATORY: use only this request_id, observation_id, and proposal_id; all older IDs are invalid",
            "_images": _public_images(checkpoint.raw),
        }

    def _execute(self, raw_decision: Mapping[str, Any]) -> Mapping[str, Any]:
        decision, checkpoint = self._validate_current_decision(raw_decision, "eef_delta")
        cached = self.broker.journal.completed_duplicate(decision)
        if cached is not None:
            return self._with_current_images(cached)
        if self._recovery_execute_turns >= self.max_recovery_execute_turns:
            return self._terminate("failed", "recovery_execute_turn_limit_exceeded")
        if self._accepted_decisions >= self.max_decisions:
            return self._terminate("budget_exhausted", "max_decisions_reached")
        remaining = self._remaining_intervention_steps()
        if remaining is not None and int(decision["steps"]) > remaining:
            raise ProtocolError("requested_steps_exceed_remaining_intervention_budget")
        current_pose = self.adapter.read_pose(checkpoint.raw)
        target = self.adapter.resolve_target(decision, current_pose)
        required = self.adapter.minimum_required_steps(target, current_pose)
        remaining_episode_steps = self._remaining_episode_steps()
        if (remaining_episode_steps is not None and
                remaining_episode_steps <= 0):
            return self._terminate("budget_exhausted", "episode_step_limit_reached")
        if (remaining_episode_steps is not None and
                int(decision["steps"]) > remaining_episode_steps):
            raise ProtocolError("requested_steps_exceed_remaining_episode_step_budget")
        if required == 0 and target.gripper == "keep":
            # A zero-motion execute cannot improve the episode.  Treat it as a
            # terminal host-side no-progress condition instead of consuming a
            # recovery turn and asking Astra to repeat the same no-op.
            return self._terminate("failed", "zero_motion_no_progress")
        if required > int(decision["steps"]):
            raise ProtocolError("target_requires_more_steps_than_requested")
        self.broker.journal.accept_and_consume(decision)
        self._accepted_decisions += 1
        self._recovery_execute_turns += 1
        action_count_before = self._recovery_model_steps()
        for _ in range(int(decision["steps"])):
            current = self.broker.current_checkpoint()
            pose = self.adapter.read_pose(current.raw)
            position_error, rotation_error = self.adapter.target_errors(target, pose)
            if (target.gripper == "keep" and
                    position_error <= self.adapter.position_tolerance_m and
                    rotation_error <= self.adapter.rotation_tolerance_rad):
                break
            action = self.adapter.action_toward_target(target, pose)
            step_result = self.broker.step(
                action, source="astra", decision_id=str(decision["decision_id"])
            )
            if step_result.task_succeeded:
                self.broker.stop("task_success_verified_by_host")
                return self._finish_decision_and_terminate(
                    decision, self._terminate("completed", "host_verified_task_success", task_succeeded=True)
                )
            if step_result.environment_ended:
                self.broker.stop("environment_ended_during_recovery")
                return self._finish_decision_and_terminate(
                    decision, self._terminate("failed", "environment_ended_during_recovery")
                )
            if (self.episode_step_limit is not None and
                    int(step_result.checkpoint.env_step) >= self.episode_step_limit):
                self.broker.stop("episode_step_limit_reached")
                return self._finish_decision_and_terminate(
                    decision,
                    self._terminate("budget_exhausted", "episode_step_limit_reached"),
                )
        packet = self._public_execution_result(decision, target, action_count_before)
        try:
            self.broker.journal.finish(decision, _without_images(packet))
        except Exception:
            self.broker.stop("decision_journal_finish_failed_after_action")
            return self._terminate("unsafe", "decision_journal_finish_failed_after_action")
        if self._recovery_execute_turns >= self.max_recovery_execute_turns:
            return self._terminate("failed", "recovery_execute_turn_limit_exceeded")
        context = self._refresh_host_context()
        packet["host_transition"] = "execute -> host_observe -> host_pi05_propose -> new_turn"
        packet["next_required_tool"] = "libero_execute_eef_or_libero_resume_pi05"
        packet["next_required_reason"] = (
            "The host already refreshed the simulator and supplied a new observation and "
            "pi05 proposal. Choose execute or resume in the next turn."
        )
        packet.pop("_images", None)
        packet["_next_turn_input"] = {
            "text": (
                "The host executed the previous correction and automatically completed "
                "observe -> pi05_propose. Read the refreshed workspace context file and "
                "make exactly one action call: execute or resume.\n\n"
                "CURRENT HOST CONTEXT JSON:\n" + _context_text(context)
            ),
            "input_items": _content_items(context),
        }
        return packet

    def _execute_eef_chunk(self, raw_decision: Mapping[str, Any]) -> Mapping[str, Any]:
        """Execute one Astra-generated chunk without mid-chunk model feedback."""
        if not self.chunk_mode_enabled:
            raise ProtocolError("eef_chunk_mode_not_enabled")
        if self.chunk_edit_enabled:
            raise ProtocolError("eef_chunk_mode_not_enabled")
        decision, checkpoint = self._validate_current_decision(raw_decision, "eef_chunk")
        cached = self.broker.journal.completed_duplicate(decision)
        if cached is not None:
            return self._with_current_images(cached)
        if self._recovery_execute_turns >= self.max_recovery_execute_turns:
            return self._terminate("failed", "recovery_execute_turn_limit_exceeded")
        if self._accepted_decisions >= self.max_decisions:
            return self._terminate("budget_exhausted", "max_decisions_reached")

        actions = list(decision["chunk"]["actions"])
        remaining = self._remaining_intervention_steps()
        if remaining is not None and len(actions) > remaining:
            raise ProtocolError("eef_chunk_steps_exceed_remaining_intervention_budget")
        remaining_episode_steps = self._remaining_episode_steps()
        if (remaining_episode_steps is not None and
                remaining_episode_steps < MIN_EEF_CHUNK_ACTIONS and
                self._takeover_mode):
            return self._terminate(
                "budget_exhausted", "episode_step_budget_below_minimum_chunk"
            )
        if (remaining_episode_steps is not None and
                len(actions) > remaining_episode_steps):
            raise ProtocolError("eef_chunk_exceeds_remaining_episode_step_budget")

        # Validate the complete sequence before consuming it. This prevents a
        # bad later item from causing a partially executed chunk.
        for action in actions:
            if (math.sqrt(sum(float(x) * float(x) for x in action["delta_position"])) >
                    float(self.adapter.max_step_position_m) + 1e-9):
                raise ProtocolError("eef_chunk_action_exceeds_one_step_position_limit")
            if (math.sqrt(sum(float(x) * float(x) for x in action["delta_rotation_vector"])) >
                    float(self.adapter.max_step_rotation_rad) + 1e-9):
                raise ProtocolError("eef_chunk_action_exceeds_one_step_rotation_limit")

        self.broker.journal.accept_and_consume(decision)
        self._accepted_decisions += 1
        self._recovery_execute_turns += 1
        action_count_before = self._recovery_model_steps()
        executed = 0
        for action_delta in actions:
            current = self.broker.current_checkpoint()
            pose = self.adapter.read_pose(current.raw)
            target = self.adapter.resolve_target(
                {"mode": "eef_delta", "delta": action_delta}, pose
            )
            step_result = self.broker.step(
                self.adapter.action_toward_target(target, pose),
                source="astra",
                decision_id=str(decision["decision_id"]),
            )
            executed += 1
            if step_result.task_succeeded:
                self.broker.stop("task_success_verified_by_host")
                return self._finish_decision_and_terminate(
                    decision,
                    self._terminate("completed", "host_verified_task_success",
                                    task_succeeded=True),
                )
            if step_result.environment_ended:
                self.broker.stop("environment_ended_during_eef_chunk")
                return self._finish_decision_and_terminate(
                    decision,
                    self._terminate("failed", "environment_ended_during_eef_chunk"),
                )
            if (self.episode_step_limit is not None and
                    int(step_result.checkpoint.env_step) >= self.episode_step_limit):
                self.broker.stop("episode_step_limit_reached")
                return self._finish_decision_and_terminate(
                    decision,
                    self._terminate("budget_exhausted", "episode_step_limit_reached"),
                )

        checkpoint = self.broker.current_checkpoint()
        pose = self.adapter.read_pose(checkpoint.raw)
        packet = {
            "result": "eef_chunk_executed",
            "decision_id": str(decision["decision_id"]),
            "requested_steps": len(actions),
            "executed_control_steps": self._recovery_model_steps() - action_count_before,
            "executed_actions": executed,
            "env_step": int(checkpoint.env_step),
            "observation_id": str(checkpoint.observation_id),
            "intermediate_observation": False,
            "intermediate_pi05_proposal": False,
            "host_refresh": "after_complete_chunk",
            "landing_report": {
                "current_eef": {
                    "frame": "world",
                    "reference": "control_site",
                    "position_m": pose.position_m.tolist(),
                    "quaternion_wxyz": pose.quaternion_wxyz.tolist(),
                },
                "task_success_verified_by_host": False,
            },
        }
        try:
            self.broker.journal.finish(decision, _without_images(packet))
        except Exception:
            self.broker.stop("decision_audit_failed_after_eef_chunk")
            return self._terminate("unsafe", "decision_audit_failed_after_eef_chunk")
        if self._recovery_execute_turns >= self.max_recovery_execute_turns:
            return self._terminate("failed", "recovery_execute_turn_limit_exceeded")

        context = self._refresh_host_context()
        packet["host_transition"] = (
            "eef_chunk -> host_observe -> host_pi05_propose -> new_turn"
        )
        packet["next_required_tool"] = "libero_execute_eef_chunk_or_libero_resume_pi05"
        packet["next_required_reason"] = (
            "The host executed the complete chunk. It now supplied a fresh observation "
            "and Pi0.5 proposal for the next decision."
        )
        packet.pop("_images", None)
        packet["_next_turn_input"] = {
            "text": (
                "The host executed the complete Astra EEF chunk and automatically "
                "completed observe -> pi05_propose only after the final chunk action. "
                "Read the refreshed context and make exactly one action call: "
                "libero_execute_eef_chunk or libero_resume_pi05.\n\n"
                "CURRENT HOST CONTEXT JSON:\n" + _context_text(context)
            ),
            "input_items": _content_items(context),
        }
        return packet

    def _edit_chunk(self, raw_decision: Mapping[str, Any]) -> Mapping[str, Any]:
        """Apply sparse EEF edits to the current pi05 chunk and execute its prefix."""
        if self._takeover_mode or not self.chunk_edit_enabled:
            raise ProtocolError("chunk_edit_mode_not_enabled")
        decision, checkpoint = self._validate_current_decision(raw_decision, "edit_pi05_chunk")
        cached = self.broker.journal.completed_duplicate(decision)
        if cached is not None:
            return self._with_current_images(cached)
        if self._recovery_execute_turns >= self.max_recovery_execute_turns:
            return self._terminate("failed", "recovery_execute_turn_limit_exceeded")
        if self._accepted_decisions >= self.max_decisions:
            return self._terminate("budget_exhausted", "max_decisions_reached")
        proposal = self.proposal_service.current
        if proposal is None or proposal.observation_id != checkpoint.observation_id:
            raise ProtocolError("fresh_pi05_proposal_required")
        chunk = decision["chunk_edits"]
        execute_steps = int(chunk["execute_steps"])
        trajectory = list(proposal.nominal_trajectory)
        if execute_steps > len(trajectory):
            raise ProtocolError("chunk_edit_steps_exceed_proposal")
        remaining = self._remaining_intervention_steps()
        if remaining is not None and execute_steps > remaining:
            raise ProtocolError("chunk_edit_steps_exceed_remaining_intervention_budget")
        edits = {}
        for edit in chunk["waypoint_edits"]:
            proposal_step = int(edit["proposal_step"])
            if proposal_step >= execute_steps:
                raise ProtocolError("chunk_edit_step_outside_execution_prefix")
            edits[proposal_step] = edit

        targets: list[MotionTarget] = []
        for index in range(execute_steps):
            row = trajectory[index]
            position = np.asarray(row["predicted_control_site_position_m"], dtype=np.float64)
            rotation = quat_wxyz_to_matrix(row["predicted_control_site_quaternion_wxyz"])
            gripper = str(row["gripper"])
            edit = edits.get(index)
            if edit is not None:
                position = position + np.asarray(edit["delta_position_m"], dtype=np.float64)
                rotation = (
                    Rotation.from_rotvec(np.asarray(edit["delta_rotation_vector_rad"], dtype=np.float64)).as_matrix()
                    @ rotation
                )
                gripper = str(edit["gripper"])
            if not np.isfinite(position).all() or not np.isfinite(rotation).all():
                raise ProtocolError("chunk_edit_target_nonfinite")
            targets.append(MotionTarget(position.copy(), rotation.copy(), gripper))

        self.broker.journal.accept_and_consume(decision)
        self._accepted_decisions += 1
        self._recovery_execute_turns += 1
        action_count_before = self._recovery_model_steps()
        executed_waypoints: list[int] = []
        for index, target in enumerate(targets):
            current = self.broker.current_checkpoint()
            pose = self.adapter.read_pose(current.raw)
            action = self.adapter.action_toward_target(target, pose)
            step_result = self.broker.step(
                action, source="astra", decision_id=str(decision["decision_id"])
            )
            executed_waypoints.append(index)
            if step_result.task_succeeded:
                self.broker.stop("task_success_verified_by_host")
                return self._finish_decision_and_terminate(
                    decision, self._terminate("completed", "host_verified_task_success", task_succeeded=True)
                )
            if step_result.environment_ended:
                self.broker.stop("environment_ended_during_chunk_edit")
                return self._finish_decision_and_terminate(
                    decision, self._terminate("failed", "environment_ended_during_chunk_edit")
                )

        checkpoint = self.broker.current_checkpoint()
        packet = {
            "result": "edited_pi05_chunk_executed_with_feedback",
            "proposal_id": proposal.proposal_id,
            "execute_steps": execute_steps,
            "edited_waypoints": sorted(edits),
            "executed_waypoints": executed_waypoints,
            "executed_control_steps": self._recovery_model_steps() - action_count_before,
            "env_step": int(checkpoint.env_step),
            "observation_id": str(checkpoint.observation_id),
            "landing_report": {
                "current_eef": {
                    "frame": "world",
                    "reference": "control_site",
                    "position_m": self.adapter.read_pose(checkpoint.raw).position_m.tolist(),
                    "quaternion_wxyz": self.adapter.read_pose(checkpoint.raw).quaternion_wxyz.tolist(),
                },
                "task_success_verified_by_host": False,
            },
        }
        try:
            self.broker.journal.finish(decision, _without_images(packet))
        except Exception:
            self.broker.stop("decision_journal_finish_failed_after_chunk_edit")
            return self._terminate("unsafe", "decision_journal_finish_failed_after_chunk_edit")
        if self._recovery_execute_turns >= self.max_recovery_execute_turns:
            return self._terminate("failed", "recovery_execute_turn_limit_exceeded")
        context = self._refresh_host_context()
        packet["host_transition"] = "edit_chunk -> host_observe -> host_pi05_propose -> new_turn"
        packet["next_required_tool"] = "libero_edit_pi05_chunk_or_libero_resume_pi05"
        packet["next_required_reason"] = (
            "The host executed the edited chunk and supplied a fresh observation and pi05 proposal. "
            "Choose another bounded edit or resume from the refreshed landing report."
        )
        packet.pop("_images", None)
        packet["_next_turn_input"] = {
            "text": (
                "The host executed the edited pi05 chunk and automatically completed "
                "observe -> pi05_propose. Read the refreshed workspace context file and "
                "make exactly one action call: edit_pi05_chunk or resume.\n\n"
                "CURRENT HOST CONTEXT JSON:\n" + _context_text(context)
            ),
            "input_items": _content_items(context),
        }
        return packet

    def _resume(self, raw_decision: Mapping[str, Any]) -> Mapping[str, Any]:
        decision, checkpoint = self._validate_current_decision(raw_decision, "resume_pi05")
        if self._takeover_mode:
            raise ProtocolError("astra_full_takeover_active_resume_forbidden")
        cached = self.broker.journal.completed_duplicate(decision)
        if cached is not None:
            return cached
        if self._accepted_decisions >= self.max_decisions:
            return self._terminate("budget_exhausted", "max_decisions_reached")
        if self.broker.task_succeeded:
            return self._terminate("completed", "host_verified_task_success", task_succeeded=True)
        if self.broker.uncertain or self.broker.environment_ended or self.broker.owner != "astra":
            raise ProtocolError("host_state_not_recoverable")
        proposal = self.proposal_service.current
        if (proposal is None or proposal.observation_id != checkpoint.observation_id or
                proposal.proposal_id != decision["proposal_id"]):
            raise ProtocolError("resume_requires_fresh_pi05_proposal_for_current_checkpoint")
        self.broker.journal.accept_and_consume(decision)
        self._accepted_decisions += 1
        packet = {
            "result": "resume_requested",
            "host_gate": "pending_runner_checkpoint_verification",
            "observation_id": checkpoint.observation_id,
            "env_step": int(checkpoint.env_step),
            "proposal_id": proposal.proposal_id,
        }
        self.broker.journal.finish(decision, packet)
        outcome = self._outcome("completed", "astra_explicit_resume_and_fresh_proposal",
                                can_resume=True, checkpoint=checkpoint)
        self._terminal_outcome = outcome
        return {**packet, "_terminal": True}

    def _stop(self, raw_decision: Mapping[str, Any]) -> Mapping[str, Any]:
        decision, checkpoint = self._validate_current_decision(raw_decision, "stop")
        cached = self.broker.journal.completed_duplicate(decision)
        if cached is not None:
            return cached
        self.broker.journal.accept_and_consume(decision)
        self.broker.journal.finish(decision, {"result": "stopped", "env_step": int(checkpoint.env_step)})
        if self.broker.task_succeeded:
            outcome = self._outcome("completed", "host_verified_task_success",
                                    can_resume=False, task_succeeded=True, checkpoint=checkpoint)
            packet = {"result": "host_verified_terminal_state", "_terminal": True}
        else:
            outcome = self._outcome("stopped", "astra_requested_stop", can_resume=False, checkpoint=checkpoint)
            packet = {"result": "intervention_stopped_without_resume", "_terminal": True}
        self.broker.stop("astra_requested_stop")
        self._terminal_outcome = outcome
        return packet

    def _validate_current_decision(self, raw: Mapping[str, Any], *modes: str) -> tuple[dict[str, Any], Any]:
        checkpoint = self.broker.current_checkpoint()
        if self.broker.owner != "astra" or checkpoint.env_step < self._intervention_start_step:
            raise ProtocolError("astra_owner_or_step_mismatch")
        proposal = self.proposal_service.current
        if proposal is None or proposal.observation_id != checkpoint.observation_id:
            raise ProtocolError("fresh_pi05_proposal_required")
        if self._request_id is None or self._request_observation_id != checkpoint.observation_id:
            raise ProtocolError("fresh_libero_observe_required")
        expected_ids = {
            "episode_id": str(self.request.episode_id),
            "intervention_id": self.intervention_id,
            "request_id": self._request_id,
            "observation_id": str(checkpoint.observation_id),
            "proposal_id": str(proposal.proposal_id),
        }
        decision = validate_decision(raw, expected_ids, expected_modes=set(modes))
        return decision, checkpoint

    def _require_current_request(self, args: Mapping[str, Any], checkpoint: Any) -> None:
        if self._request_id is None or self._request_observation_id != checkpoint.observation_id:
            raise ProtocolError("fresh_libero_observe_required")
        if args["request_id"] != self._request_id or args["observation_id"] != checkpoint.observation_id:
            raise ProtocolError("stale_or_wrong_request_observation")

    def _public_execution_result(self, decision: Mapping[str, Any], target: MotionTarget,
                                  action_count_before: int) -> Dict[str, Any]:
        checkpoint = self.broker.current_checkpoint()
        pose = self.adapter.read_pose(checkpoint.raw)
        position_error, rotation_error = self.adapter.target_errors(target, pose)
        return {
            "result": "correction_executed_with_feedback",
            "next_required_tool": "libero_observe",
            "next_required_reason": (
                "The simulator advanced after the correction. Read the new state before "
                "accepting another pi05 proposal or handing control back."
            ),
            "decision_id": str(decision["decision_id"]),
            "executed_control_steps": self._recovery_model_steps() - action_count_before,
            "env_step": int(checkpoint.env_step),
            "observation_id": str(checkpoint.observation_id),
            "measured_current_eef": {
                "frame": "world", "reference": "control_site",
                "position_m": pose.position_m.tolist(),
                "quaternion_wxyz": pose.quaternion_wxyz.tolist(),
            },
            "remaining_target_error_m": position_error,
            "remaining_target_rotation_error_rad": rotation_error,
            "gripper_qpos": _public_gripper(checkpoint.raw),
            "_images": _public_images(checkpoint.raw),
        }

    def _with_current_images(self, packet: Mapping[str, Any]) -> Dict[str, Any]:
        result = dict(packet)
        # Current observation images live in latest_context.json/workspace and
        # must not be copied into every duplicate-tool result.
        result.pop("_images", None)
        result["workspace_context_file"] = str(
            Path(self.workspace) / "latest_context.json"
        )
        return result

    def _finish_decision_and_terminate(self, decision: Mapping[str, Any], result: Mapping[str, Any]) -> Mapping[str, Any]:
        try:
            self.broker.journal.finish(decision, dict(result))
        except Exception as error:
            self._record_host_failure_diagnostic(
                error,
                phase="decision_audit_finish",
                tool=(
                    "libero_execute_eef_chunk"
                    if decision.get("mode") == "eef_chunk"
                    else "libero_execute_eef"
                ),
                decision_id=str(decision.get("decision_id", "")),
            )
            self.broker.stop("decision_audit_failed_after_environment_step")
            self._terminal_outcome = self._outcome("unsafe", "decision_audit_failed_after_step",
                                                   can_resume=False)
            return {"result": "host_stopped_after_audit_failure", "_terminal": True}
        return result

    def _terminate(self, status: str, reason: str, *, uncertain: bool = False,
                   task_succeeded: Optional[bool] = None) -> Mapping[str, Any]:
        succeeded = self.broker.task_succeeded if task_succeeded is None else bool(task_succeeded)
        checkpoint = None
        if not self.broker.uncertain:
            try:
                checkpoint = self.broker.current_checkpoint()
            except Exception:
                checkpoint = None
        self._terminal_outcome = self._outcome(
            status, reason, can_resume=False, task_succeeded=succeeded,
            execution_uncertain=bool(uncertain or self.broker.uncertain), checkpoint=checkpoint,
        )
        if not succeeded and self.broker.owner != "stopped":
            self.broker.stop(reason)
        return {
            "result": "host_terminated_intervention",
            "terminal_reason": str(reason),
            "last_confirmed_env_step": int(self.broker.env_steps),
            "_terminal": True,
        }

    def _record_host_failure_diagnostic(
        self,
        error: Exception,
        *,
        phase: str,
        tool: Optional[str] = None,
        tool_arguments: Optional[Mapping[str, Any]] = None,
        decision_id: Optional[str] = None,
    ) -> Optional[str]:
        """Write private exception detail when fail-closed host handling fires."""
        if not self.audit_path:
            return None

        def redact(value: Any, limit: int) -> str:
            text = str(value)
            text = re.sub(r"(?i)\bBearer\s+[^\s,;]+", "Bearer [REDACTED]", text)
            text = re.sub(
                r"(?i)((?:api[_-]?key|access[_-]?token|password|secret)\s*[:=]\s*)[^\s,;]+",
                r"\1[REDACTED]",
                text,
            )
            text = re.sub(r"\bsk-[A-Za-z0-9_-]{12,}", "[REDACTED_KEY]", text)
            return text[:limit]

        args = tool_arguments if isinstance(tool_arguments, Mapping) else {}
        chunk = args.get("chunk") if isinstance(args.get("chunk"), Mapping) else {}
        actions = chunk.get("actions") if isinstance(chunk, Mapping) else None
        context = self._last_host_context if isinstance(self._last_host_context, Mapping) else {}
        try:
            env_step = int(getattr(self.broker, "env_steps"))
        except Exception:
            env_step = None
        try:
            recovery_steps = int(self._recovery_model_steps())
        except Exception:
            recovery_steps = None
        record = {
            "event": "host_step_or_audit_failure_diagnostic",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "phase": str(phase),
            "episode_id": str(getattr(self.request, "episode_id", "")),
            "intervention_id": str(self.intervention_id),
            "tool": str(tool) if tool else None,
            "tool_call_index": int(self._tool_calls),
            "decision_id": str(decision_id or args.get("decision_id") or "") or None,
            "tool_call_summary": {
                "mode": args.get("mode"),
                "steps": args.get("steps"),
                "chunk_action_count": len(actions) if isinstance(actions, list) else None,
            },
            "env_step": env_step,
            "astra_control_steps": recovery_steps,
            "broker_owner": str(getattr(self.broker, "owner", "unknown")),
            "broker_uncertain": bool(getattr(self.broker, "uncertain", False)),
            "latest_context_ids": {
                key: context.get(key)
                for key in ("env_step", "request_id", "observation_id", "proposal_id")
                if context.get(key) is not None
            },
            "exception_type": f"{type(error).__module__}.{type(error).__name__}",
            "exception_message": redact(error, 4000),
            "traceback": redact(
                "".join(traceback.format_exception(type(error), error, error.__traceback__)),
                30000,
            ),
        }
        target = Path(self.audit_path).with_name(
            "host_step_or_audit_failure_diagnostics.jsonl"
        )
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            return str(target)
        except Exception:
            return None

    def _outcome(self, status: str, detail: str, *, can_resume: bool,
                 task_succeeded: Optional[bool] = None, execution_uncertain: bool = False,
                 checkpoint: Any = None) -> RecoveryOutcome:
        task_succeeded = self.broker.task_succeeded if task_succeeded is None else bool(task_succeeded)
        if checkpoint is None and not (execution_uncertain or self.broker.uncertain):
            try:
                checkpoint = self.broker.current_checkpoint()
            except Exception:
                checkpoint = None
        end_step = int(self.broker.env_steps)
        if checkpoint is not None:
            end_step = int(checkpoint.env_step)
        can_resume = bool(can_resume and status == "completed" and not task_succeeded and
                          not execution_uncertain and not self.broker.uncertain and
                          checkpoint is not None and self.broker.owner == "astra" and
                          not self.broker.environment_ended)
        resume_observation = deepcopy(checkpoint.raw) if can_resume else None
        try:
            robot_state = self.robot_state_encoder(checkpoint.raw) if can_resume else None
        except Exception:
            can_resume = False
            resume_observation = None
            robot_state = None
            status = "unsafe"
            detail = "resume_robot_state_encoding_failed"
        if can_resume:
            try:
                robot_state = np.asarray(robot_state, dtype=np.float32)
                if robot_state.shape != (8,) or not np.isfinite(robot_state).all():
                    raise ValueError("invalid_robot_state")
            except Exception:
                can_resume = False
                resume_observation = None
                robot_state = None
                status = "unsafe"
                detail = "resume_robot_state_invalid"
        start = int(self._intervention_start_step)
        return RecoveryOutcome(
            status=str(status),
            detail=str(detail),
            intervention_start_step=start,
            intervention_end_step=end_step,
            resume_step=end_step if can_resume else None,
            resume_observation=resume_observation,
            resume_robot_state=robot_state,
            can_resume=can_resume,
            task_succeeded=bool(task_succeeded),
            checkpoint_id=str(checkpoint.checkpoint_id) if checkpoint is not None else None,
            astra_called=bool(self._model_response_received or
                              (self._client is not None and self._client.model_response_received)),
            recovery_executed=self._recovery_model_steps() > self._starting_recovery_model_steps,
            episode_id=str(self.request.episode_id) if self.request is not None else None,
            intervention_id=self.intervention_id or None,
            resume_observation_id=str(checkpoint.observation_id) if can_resume and checkpoint is not None else None,
            execution_uncertain=bool(execution_uncertain or self.broker.uncertain),
            model_response_received=bool(self._model_response_received or
                                         (self._client is not None and self._client.model_response_received)),
            resume_requested=bool(can_resume and detail == "astra_explicit_resume_and_fresh_proposal"),
            audit_path=self.audit_path,
            end_reason=str(detail),
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            codex_cli_version=self._codex_cli_version,
        )

    def _recovery_model_steps(self) -> int:
        return int(self.broker.recovery_model_total_steps) - int(self._starting_recovery_model_steps)

    def _remaining_intervention_steps(self) -> Optional[int]:
        if self.max_total_steps is None:
            return None
        return max(0, self.max_total_steps - self._recovery_model_steps())

    def _remaining_episode_steps(self) -> Optional[int]:
        limit = getattr(self, "episode_step_limit", None)
        if limit is None:
            return None
        return max(0, int(limit) - int(self.broker.env_steps))


# Backward-compatible import name for existing runners and saved experiments.
AstraRecoveryExecutor = RecoveryOrchestrator


def _exact_args(args: Mapping[str, Any], expected: tuple[str, ...]) -> None:
    if not isinstance(args, Mapping) or set(args) != set(expected):
        raise ProtocolError("tool_arguments_wrong_keys")


def _safe_tool_error_code(value: Any) -> str:
    if not isinstance(value, str):
        return "tool_rejected"
    token = value.strip()
    if (not 1 <= len(token) <= 96 or
            not all(char.isalnum() or char in "._:-" for char in token) or
            any(word in token.lower() for word in ("secret", "password", "token", "key="))):
        return "tool_rejected"
    return token


def _retryable_model_transport_error(code: Any) -> Optional[str]:
    """Classify only transient Responses failures safe for bounded replay."""
    if not isinstance(code, str):
        return None
    normalized = re.sub(r"[^a-z0-9]", "", code.lower())
    if normalized == "modelidletimeout":
        return "model_idle_timeout"
    if "responsestreamdisconnected" in normalized:
        return "response_stream_disconnected"
    return None


def _public_gripper(raw: Mapping[str, Any]) -> list[float]:
    value = np.asarray(raw.get("robot0_gripper_qpos"), dtype=np.float64)
    if value.shape != (2,) or not np.isfinite(value).all():
        raise AdapterError("public_gripper_state_invalid")
    return value.tolist()


def _public_images(raw: Mapping[str, Any]) -> list[dict[str, str]]:
    import imageio.v2 as imageio

    required = (("agentview_image", "external RGB"),
                ("robot0_eye_in_hand_image", "wrist RGB"))
    images = []
    for key, label in required:
        if key not in raw:
            raise AdapterError("public_RGB_view_missing")
        array = np.asarray(raw[key])
        if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
            raise AdapterError("public_RGB_view_must_be_HxWx3_uint8")
        upright = np.ascontiguousarray(array[::-1, ::-1])
        buffer = BytesIO()
        imageio.imwrite(buffer, upright, format="png")
        data_url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
        images.append({"label": label, "data_url": data_url})
    return images


def _without_images(packet: Mapping[str, Any]) -> Dict[str, Any]:
    result = dict(packet)
    result.pop("_images", None)
    return result
