#!/usr/bin/env python3
"""Run a task-disjoint π0.5 CALL_ASTRA closed loop on LIBERO-10.

The default logging handoff saves a complete CALL snapshot and stops
autonomous π0.5 control.  The runner also accepts a resumable handoff handler:
when an injected Astra/recovery executor returns a validated checkpoint, the
re-entry gate can hand control back to π0.5 at the recorded step.  Heavy
JAX/OpenPI imports are deliberately deferred until after the selected GPU
passes the idle guard.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
EXPERIMENT_ROOT = PROJECT_ROOT
DEFAULT_CHECKPOINT = Path(os.environ["PI05_CHECKPOINT"]) if os.environ.get("PI05_CHECKPOINT") else None
DEFAULT_OPENPI_ROOT = Path(os.environ["OPENPI_ROOT"]) if os.environ.get("OPENPI_ROOT") else None

for _path in (PROJECT_ROOT, EXPERIMENT_ROOT, SCRIPT_DIR, PROJECT_ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from call_llm.action_statistics import action_statistics  # noqa: E402
from call_llm.call_controller import CallController, ControllerState  # noqa: E402
from call_llm.decision import (  # noqa: E402
    BudgetDecisionEngine,
    CallDecisionEngine,
    DecisionEngine,
    OracleCompetenceDecisionEngine,
)
from call_llm.handoff import HandoffHandler, LoggingHandoffHandler  # noqa: E402


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(nested) for nested in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(_json_safe(value), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class _EpisodeVideoRecorder:
    """Write the live agent and wrist observations with step/source overlays."""

    def __init__(self, path: Path, *, fps: int = 10) -> None:
        import cv2
        self._cv2 = cv2
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = None
        self._fps = int(fps)

    def write(self, observation: Mapping[str, Any], env_step: int, source: str) -> None:
        import cv2
        frames = []
        for key in ("agentview_image", "robot0_eye_in_hand_image"):
            image = observation.get(key)
            if image is None:
                continue
            frame = np.asarray(image)
            if frame.ndim != 3 or frame.shape[2] != 3:
                continue
            frame = np.ascontiguousarray(frame[:, :, ::-1])
            frames.append(frame)
        if not frames:
            return
        height = max(frame.shape[0] for frame in frames)
        normalized = []
        for frame in frames:
            if frame.shape[0] != height:
                width = max(1, int(frame.shape[1] * height / frame.shape[0]))
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            normalized.append(frame)
        canvas = np.concatenate(normalized, axis=1)
        cv2.putText(canvas, f"step={int(env_step)} source={source}", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        if self._writer is None:
            h, w = canvas.shape[:2]
            self._writer = cv2.VideoWriter(
                str(self.path), cv2.VideoWriter_fourcc(*"mp4v"),
                self._fps, (int(w), int(h)), True,
            )
            if not self._writer.isOpened():
                raise RuntimeError("video_writer_open_failed")
        self._writer.write(canvas)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None


def parse_gpu_status(output: str, gpu_index: int) -> tuple[int, int]:
    """Parse nvidia-smi CSV without importing CUDA libraries."""
    for line in output.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) < 3:
            continue
        try:
            index = int(fields[0])
            memory_mib = int(fields[1])
            utilization = int(fields[2])
        except ValueError:
            continue
        if index == int(gpu_index):
            return memory_mib, utilization
    raise ValueError(f"nvidia-smi did not report physical GPU {gpu_index}")


def assert_gpu_idle(
    gpu_index: int,
    *,
    max_memory_mib: int = 2048,
    max_utilization_percent: int = 5,
    query: Callable[[], str] | None = None,
) -> tuple[int, int]:
    if query is None:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        output = completed.stdout
    else:
        output = query()
    memory_mib, utilization = parse_gpu_status(output, gpu_index)
    if memory_mib > max_memory_mib or utilization > max_utilization_percent:
        raise RuntimeError(
            f"refusing model inference on GPU {gpu_index}: {memory_mib} MiB used, "
            f"{utilization}% utilization; idle limits are {max_memory_mib} MiB and "
            f"{max_utilization_percent}%. No model was loaded."
        )
    return memory_mib, utilization


def _quat_to_axisangle(quat: Any) -> np.ndarray:
    quaternion = np.asarray(quat, dtype=np.float64).copy()
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError("robot end-effector quaternion must be a finite 4-vector")
    quaternion[3] = np.clip(quaternion[3], -1.0, 1.0)
    denominator = math.sqrt(max(0.0, 1.0 - float(quaternion[3]) ** 2))
    if denominator < 1e-8:
        return np.zeros(3, dtype=np.float32)
    return (
        quaternion[:3] * (2.0 * math.acos(float(quaternion[3]))) / denominator
    ).astype(np.float32)


def robot_state_from_observation(observation: Mapping[str, Any]) -> np.ndarray:
    return np.concatenate(
        (
            np.asarray(observation["robot0_eef_pos"], dtype=np.float32),
            _quat_to_axisangle(observation["robot0_eef_quat"]),
            np.asarray(observation["robot0_gripper_qpos"], dtype=np.float32),
        )
    ).astype(np.float32)


def policy_observation(observation: Mapping[str, Any], task_instruction: str) -> dict[str, Any]:
    from openpi_client import image_tools

    image = np.ascontiguousarray(observation["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(
        observation["robot0_eye_in_hand_image"][::-1, ::-1]
    )
    image = image_tools.convert_to_uint8(image_tools.resize_with_pad(image, 224, 224))
    wrist = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(wrist, 224, 224)
    )
    return {
        "observation/image": image,
        "observation/wrist_image": wrist,
        "observation/state": robot_state_from_observation(observation),
        "prompt": str(task_instruction),
    }


def state_statistics_from_policy_result(policy_result: Mapping[str, Any]) -> np.ndarray:
    state = np.asarray(policy_result["state"], dtype=np.float32)
    actions = np.asarray(policy_result["actions"], dtype=np.float32)
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError(f"π0.5 raw robot state must be finite with shape (8,), got {state.shape}")
    if actions.shape != (10, 7) or not np.isfinite(actions).all():
        raise ValueError(f"π0.5 planned action must be finite with shape (10, 7), got {actions.shape}")
    result = np.concatenate((state, action_statistics(actions))).astype(np.float32)
    if result.shape != (44,) or not np.isfinite(result).all():
        raise ValueError("derived state/action statistics are malformed")
    return result


def feature_vector_from_policy_result(
    policy_result: Mapping[str, Any], feature_set: str
) -> np.ndarray:
    from call_llm.call_assist_head import FEATURE_SET_DIMS

    if "features" not in policy_result:
        raise KeyError("π0.5 feature path did not return hidden features")
    features = policy_result["features"]
    semantic = np.asarray(features["h_sem"], dtype=np.float32).reshape(-1)
    action = np.asarray(features["h_act"], dtype=np.float32).reshape(-1)
    if semantic.shape != (2048,) or action.shape != (1024,):
        raise ValueError(
            f"unexpected frozen π0.5 feature shapes: h_sem={semantic.shape}, h_act={action.shape}"
        )
    state_stats = state_statistics_from_policy_result(policy_result)
    if feature_set == "full":
        vector = np.concatenate((semantic, action, state_stats))
    elif feature_set == "state_only":
        vector = state_stats
    elif feature_set == "semantic_only":
        vector = semantic
    elif feature_set == "action_only":
        vector = action
    else:
        raise ValueError(f"unsupported feature set {feature_set!r}")
    if vector.shape != (FEATURE_SET_DIMS[feature_set],) or not np.isfinite(vector).all():
        raise ValueError(f"invalid {feature_set} inference vector: {vector.shape}")
    return vector.astype(np.float32, copy=False)


def _copy_simulator_snapshot(env: Any) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
    sim = getattr(env, "sim", None)
    if sim is None or getattr(sim, "data", None) is None:
        raise ValueError("CALL requires simulator qpos/qvel state")
    qpos = np.asarray(sim.data.qpos, dtype=np.float64).copy()
    qvel = np.asarray(sim.data.qvel, dtype=np.float64).copy()
    simulator_state = None
    if callable(getattr(sim, "get_state", None)):
        state = sim.get_state()
        flatten = getattr(state, "flatten", None)
        if callable(flatten):
            simulator_state = np.asarray(flatten(), dtype=np.float64).copy()
    return simulator_state, qpos, qvel


def _flattened_state_for_audit(
    env: Any,
    simulator_state: np.ndarray | None,
    qpos: np.ndarray,
    qvel: np.ndarray,
) -> np.ndarray | None:
    if simulator_state is not None:
        state = np.asarray(simulator_state, dtype=np.float64)
    else:
        sim = getattr(env, "sim", None)
        data = getattr(sim, "data", None)
        simulation_time = getattr(data, "time", None)
        if simulation_time is None:
            return None
        state = np.concatenate(
            (
                np.asarray([simulation_time], dtype=np.float64),
                np.asarray(qpos, dtype=np.float64).reshape(-1),
                np.asarray(qvel, dtype=np.float64).reshape(-1),
            )
        )
    if state.ndim != 1 or not state.size or not np.isfinite(state).all():
        return None
    return state.copy()


def _atomic_numpy_save(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite audit state: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".npy.tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, np.asarray(value, dtype=np.float64), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _task_succeeded(env: Any, reward: Any, done: bool, info: Mapping[str, Any]) -> bool:
    check_success = getattr(env, "check_success", None)
    if callable(check_success):
        return bool(check_success())
    for key in ("success", "is_success", "task_success"):
        if key in info:
            return bool(info[key])
    if reward is not None and float(reward) > 0.0:
        return True
    # Match the existing LIBERO counterfactual protocol if no explicit success
    # accessor is exposed by the environment wrapper.
    return bool(done)


def _progress_snapshot(
    observation: Mapping[str, Any], env: Any
) -> tuple[np.ndarray | None, str]:
    """Return a compact, non-image progress telemetry vector.

    LIBERO observations sometimes expose an object-state vector. When it is
    unavailable, the simulator object-joint tail is used as a benchmark-only
    fallback. The source is always recorded so downstream analyses can avoid
    treating a privileged fallback as deployable visual progress.
    """
    for key in ("object-state", "object_state", "object_state_vector"):
        value = observation.get(key) if isinstance(observation, Mapping) else None
        if value is None:
            continue
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
        if vector.size and np.isfinite(vector).all():
            return vector.copy(), f"observation:{key}"

    sim = getattr(env, "sim", None)
    data = getattr(sim, "data", None)
    qpos = getattr(data, "qpos", None)
    if qpos is not None:
        vector = np.asarray(qpos, dtype=np.float32).reshape(-1)
        # The first 9 coordinates are the Panda arm/gripper coordinates in
        # LIBERO. The remaining coordinates provide a generic object-joint
        # motion proxy for the offline benchmark.
        if vector.size > 9 and np.isfinite(vector[9:]).all():
            return vector[9:].copy(), "sim_qpos_object_tail"
        if vector.size and np.isfinite(vector).all():
            return vector.copy(), "sim_qpos"

    try:
        vector = robot_state_from_observation(observation)
    except (KeyError, TypeError, ValueError):
        return None, "unavailable"
    return vector.astype(np.float32, copy=True), "robot_state_fallback"


def _normalized_progress_delta(
    current: np.ndarray | None, previous: np.ndarray | None
) -> float | None:
    if current is None or previous is None or current.shape != previous.shape:
        return None
    delta = float(np.linalg.norm(current.astype(np.float64) - previous.astype(np.float64)))
    return delta / max(1.0, math.sqrt(float(current.size)))


def _action_effort(action_chunk: np.ndarray) -> float:
    values = np.asarray(action_chunk, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 7 or not np.isfinite(values).all():
        raise ValueError(f"cannot compute action effort from shape {values.shape}")
    return float(np.linalg.norm(values, axis=1).mean())


def _build_astra_runtime(
    *,
    env: Any,
    initial_observation: Mapping[str, Any],
    episode_id: str,
    task_instruction: str,
    policy: Any,
    policy_infer: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    observation_encoder: Callable[[Mapping[str, Any], str], Mapping[str, Any]],
    action_queue: deque[np.ndarray],
    runtime_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Create one isolated recovery runtime around this already-reset LIBERO env."""
    from call_llm.runtime.adapter import LiberoEefAdapter
    from call_llm.runtime.codex_client import DEFAULT_ASTRA_MODEL, DEFAULT_REASONING_EFFORT
    from call_llm.runtime.executor import RecoveryOrchestrator
    from call_llm.runtime.handoff import ExclusiveAstraHandoffHandler
    from call_llm.runtime.history import PublicEefHistory
    from call_llm.runtime.journal import Journal
    from call_llm.runtime.proposal import Pi05ProposalService
    from call_llm.runtime.step_broker import StepBroker

    root = Path(runtime_config["runtime_root"]).expanduser().resolve()
    if root.exists():
        raise FileExistsError("refusing_to_reuse_Astra_episode_runtime_directory")
    root.mkdir(parents=True, exist_ok=False)
    private_dir = root / "host_private"
    workspace = root / "codex_workspace"
    private_dir.mkdir()
    workspace.mkdir()
    for protected in (root, private_dir, workspace):
        try:
            os.chmod(protected, 0o700)
        except OSError:
            pass

    journal = None
    history = None
    try:
        adapter = LiberoEefAdapter(
            env,
        )
        controller_audit = adapter.runtime_audit(initial_observation)
        audit_path = private_dir / "controller_runtime_audit.json"
        _atomic_json(audit_path, controller_audit)
        journal = Journal(str(private_dir / "step_journal.jsonl"))
        history = PublicEefHistory(
            str(private_dir / "measured_eef_history.jsonl"), adapter,
            max_records=256, model_window=20,
        )
        broker = StepBroker(
            env,
            dict(initial_observation),
            episode_id=str(episode_id),
            journal=journal,
            history=history,
            pi05_action_validator=adapter.pi05_action_validator,
            astra_action_validator=adapter.action_validator,
        )
        broker.frame_sink = runtime_config.get("frame_sink")
        history.register_initial(broker.current_checkpoint())
        journal.append({
            "event": "episode_runtime_started",
            "episode_id": str(episode_id),
            "suite_task_instruction": str(task_instruction),
            "initial_observation_id": broker.current_checkpoint().observation_id,
            "controller_audit": controller_audit,
            "model_name": str(runtime_config.get("model", DEFAULT_ASTRA_MODEL)),
            # Legacy key retained because existing reports use it.
            "astra_model": str(runtime_config.get("model", DEFAULT_ASTRA_MODEL)),
            "reasoning_effort": str(runtime_config.get("reasoning_effort", DEFAULT_REASONING_EFFORT)),
        })
        proposal_service = Pi05ProposalService(
            policy, policy_infer, observation_encoder, adapter
        )
        executor = RecoveryOrchestrator(
            env=env,
            broker=broker,
            adapter=adapter,
            history=history,
            proposal_service=proposal_service,
            observation_encoder=observation_encoder,
            robot_state_encoder=robot_state_from_observation,
            audit_path=str(private_dir / "step_journal.jsonl"),
            workspace=str(workspace),
            model=str(runtime_config.get("model", DEFAULT_ASTRA_MODEL)),
            reasoning_effort=str(runtime_config.get("reasoning_effort", DEFAULT_REASONING_EFFORT)),
            max_total_steps=runtime_config.get("max_total_steps"),
            episode_step_limit=runtime_config.get("episode_step_limit"),
            max_decisions=int(runtime_config.get("max_decisions", 25)),
            max_recovery_execute_turns=int(runtime_config.get("max_execution_chunks", 25)),
            max_tool_calls=int(runtime_config.get("max_tool_calls", 100)),
            max_invalid_calls=int(runtime_config.get("max_invalid_calls", 3)),
            max_wall_seconds=float(runtime_config.get("max_wall_seconds", 300.0)),
            chunk_edit_enabled=bool(runtime_config.get("chunk_edit_enabled", False)),
            chunk_mode_enabled=bool(runtime_config.get("chunk_mode_enabled", False)),
        )
        handler = ExclusiveAstraHandoffHandler(
            snapshot_root=str(runtime_config["snapshot_root"]),
            executor=executor,
            broker=broker,
            action_queue=action_queue,
        )
        return {
            "root": root,
            "private_dir": private_dir,
            "workspace": workspace,
            "adapter": adapter,
            "controller_audit": controller_audit,
            "journal": journal,
            "history": history,
            "broker": broker,
            "proposal_service": proposal_service,
            "executor": executor,
            "handler": handler,
            "model": executor.model,
            "reasoning_effort": executor.reasoning_effort,
        }
    except BaseException:
        if history is not None:
            history.close()
        if journal is not None:
            journal.close()
        raise


def _resume_pi05_from_astra(
    *, controller: CallController, recovery: Any, broker: Any,
    call_step: int, episode_id: str,
) -> tuple[Any, Any]:
    """Validate the exact live host checkpoint before transferring ownership."""
    def reject(reason: str) -> None:
        raise RuntimeError("Astra_resume_gate:" + reason)

    if recovery is None or recovery.status != "completed":
        reject("status_not_completed")
    if not recovery.can_resume or not recovery.resume_requested:
        reject("explicit_resume_not_authorized")
    if recovery.task_succeeded or recovery.execution_uncertain:
        reject("terminal_or_uncertain_episode")
    if not recovery.model_response_received:
        reject("no_model_response")
    if recovery.end_reason != "astra_explicit_resume_and_fresh_proposal":
        reject("wrong_resume_reason")
    if recovery.episode_id != str(episode_id) or not recovery.intervention_id:
        reject("identity_missing_or_mismatched")
    if int(recovery.intervention_start_step) != int(call_step):
        reject("call_step_mismatch")
    if broker.owner != "astra" or broker.uncertain or broker.environment_ended:
        reject("host_owner_or_environment_not_resumable")
    if broker.task_succeeded:
        reject("host_verified_task_success")

    checkpoint = broker.current_checkpoint()
    if checkpoint.episode_id != str(episode_id):
        reject("live_episode_mismatch")
    if recovery.checkpoint_id != checkpoint.checkpoint_id:
        reject("checkpoint_id_mismatch")
    if recovery.resume_observation_id != checkpoint.observation_id:
        reject("observation_id_mismatch")
    if recovery.intervention_end_step is None or int(recovery.intervention_end_step) != checkpoint.env_step:
        reject("intervention_end_step_mismatch")
    if recovery.resume_step is None or int(recovery.resume_step) != checkpoint.env_step:
        reject("resume_step_mismatch")
    if not isinstance(recovery.resume_observation, Mapping):
        reject("resume_observation_missing")
    for key in (
        "agentview_image", "robot0_eye_in_hand_image", "robot0_eef_pos",
        "robot0_eef_quat", "robot0_gripper_qpos",
    ):
        if key not in recovery.resume_observation or key not in checkpoint.raw:
            reject("resume_public_observation_missing_" + key)
        proposed = np.asarray(recovery.resume_observation[key])
        current = np.asarray(checkpoint.raw[key])
        if proposed.shape != current.shape or proposed.dtype != current.dtype or not np.array_equal(proposed, current):
            reject("resume_observation_not_live_checkpoint_" + key)
    expected_robot_state = robot_state_from_observation(checkpoint.raw)
    returned_robot_state = np.asarray(recovery.resume_robot_state, dtype=np.float32)
    if returned_robot_state.shape != (8,) or not np.array_equal(returned_robot_state, expected_robot_state):
        reject("resume_robot_state_mismatch")

    gate_decision = controller.resume_after_recovery()
    if not gate_decision.allowed or gate_decision.resume_step != checkpoint.env_step:
        reject("controller_reentry_gate_disagrees")
    broker.begin_pi05_phase()
    broker.transfer("astra", "pi05")
    return gate_decision, checkpoint


def run_closed_loop_episode(
    env: Any,
    initial_observation: Mapping[str, Any],
    *,
    suite: str,
    task_id: int,
    task_name: str,
    task_instruction: str,
    episode_id: str,
    bddl_path: str | Path,
    policy_infer: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    score_help: Callable[[Mapping[str, Any]], float],
    decision_engine: DecisionEngine,
    handoff_handler: HandoffHandler,
    max_steps: int = 520,
    episode_step_limit: int = 1000,
    replan_steps: int = 5,
    environment_seed: int | None = None,
    policy_sampling_seed: int | None = None,
    reset_retries: int = 0,
    source: str = "libero_10_reset",
    initial_state_sample_id: str | None = None,
    initial_q_pi: float | None = None,
    score_source: str = "assist_head",
    audit_state_dir: str | Path | None = None,
    shadow_mode: bool = False,
    cooldown_queries: int = 0,
    observation_encoder: Callable[[Mapping[str, Any], str], Mapping[str, Any]] = policy_observation,
    policy: Any = None,
    astra_runtime_config: Mapping[str, Any] | None = None,
    smoke_force_call_after_query: int | None = None,
    video_path: str | Path | None = None,
    min_steps_between_calls: int = 20,
) -> dict[str, Any]:
    if max_steps <= 0 or episode_step_limit <= 0 or replan_steps <= 0:
        raise ValueError("max_steps, episode_step_limit, and replan_steps must be positive")
    if cooldown_queries < 0:
        raise ValueError("cooldown_queries must be nonnegative")
    if int(min_steps_between_calls) < 0:
        raise ValueError("min_steps_between_calls must be nonnegative")
    if astra_runtime_config is not None and shadow_mode:
        raise ValueError("Astra_live_control_cannot_run_in_shadow_mode")
    if astra_runtime_config is not None and policy is None:
        raise ValueError("live_Astra_runtime_requires_pi05_policy_object")
    if smoke_force_call_after_query is not None and astra_runtime_config is None:
        raise ValueError("diagnostic_forced_CALL_requires_live_Astra_runtime")
    action_queue: deque[np.ndarray] = deque()
    env_steps = 0
    phase_env_steps = 0
    policy_queries = 0
    query_log: list[dict[str, Any]] = []
    audit_state_history: deque[dict[str, Any]] = deque(maxlen=5)
    call_decision = None
    handoff_response = None
    terminal_category: str | None = None
    terminal_reason: str | None = None
    success: bool | None = None
    call_step: int | None = None
    call_query_idx: int | None = None
    p_help_at_call: float | None = None
    shadow_would_call = False
    shadow_call_events: list[dict[str, Any]] = []
    recovery_events: list[dict[str, Any]] = []
    astra_takeover_started = False
    max_step_call: dict[str, Any] | None = None
    previous_progress: np.ndarray | None = None
    previous_progress_source = "unavailable"
    previous_action_effort: float | None = None
    astra_components = None
    smoke_call_forced = False
    handoff_detail = None
    pi05_inference_seconds = 0.0
    astra_inference_seconds = 0.0
    video_recorder = None
    if video_path is not None:
        video_recorder = _EpisodeVideoRecorder(Path(video_path))
        video_recorder.write(initial_observation, 0, "reset")
        if isinstance(astra_runtime_config, dict):
            astra_runtime_config["frame_sink"] = video_recorder.write

    if astra_runtime_config is not None:
        astra_components = _build_astra_runtime(
            env=env,
            initial_observation=initial_observation,
            episode_id=episode_id,
            task_instruction=task_instruction,
            policy=policy,
            policy_infer=policy_infer,
            observation_encoder=observation_encoder,
            action_queue=action_queue,
            runtime_config=astra_runtime_config,
        )
        handoff_handler = astra_components["handler"]

    controller = CallController(
        episode_id=episode_id,
        suite=suite,
        policy_infer=policy_infer,
        score_help=score_help,
        handoff_handler=handoff_handler,
        decision_engine=decision_engine,
        shadow_mode=shadow_mode,
        min_steps_between_calls=int(min_steps_between_calls),
    )
    observation = initial_observation
    started = time.monotonic()
    try:
        while True:
            if env_steps >= episode_step_limit:
                terminal_category = (
                    "FAILURE_AFTER_RECOVERY" if recovery_events
                    else "FAILURE_WITHOUT_CALL"
                )
                terminal_reason = "episode_step_limit_reached"
                success = False
                break
            if phase_env_steps >= max_steps:
                # A phase boundary is a hard safety path.  If no handoff has
                # happened yet, force CALL_ASTRA even when the learned rule
                # never reached its confirmation count.
                # In shadow mode a learned CALL_ASTRA is still a simulated
                # prior handoff. Do not add a second max-step event for the
                # same episode merely because shadow mode keeps π0.5 running.
                post_recovery_takeover_pending = bool(recovery_events) and not astra_takeover_started
                if (
                    (call_step is not None and not post_recovery_takeover_pending)
                    or shadow_would_call
                    or terminal_category is not None
                ):
                    break
                last_policy_result = controller.last_policy_result
                if last_policy_result is None:
                    raise RuntimeError(
                        "max-step handoff has no preceding π0.5 policy result"
                    )
                simulator_state, qpos, qvel = _copy_simulator_snapshot(env)
                boundary_query_idx = max(0, policy_queries - 1)
                boundary_env_step = int(env_steps)
                boundary_phase_step = int(phase_env_steps)
                boundary_robot_state = robot_state_from_observation(observation)
                if post_recovery_takeover_pending and astra_components is not None:
                    astra_takeover_started = True
                    astra_components["executor"].enable_full_takeover()
                astra_started = time.monotonic()
                forced_outcome = controller.force_handoff(
                    last_policy_result,
                    environment_observation=observation,
                    task_instruction=task_instruction,
                    episode_id=episode_id,
                    policy_query_idx=boundary_query_idx,
                    env_step=env_steps,
                    robot_state=boundary_robot_state,
                    simulator_state=simulator_state,
                    bddl_path=str(bddl_path),
                    qpos=qpos,
                    qvel=qvel,
                    reason=(
                        "pi05_steps_exhausted_after_recovery"
                        if post_recovery_takeover_pending
                        else "max_steps_reached"
                    ),
                )
                astra_inference_seconds += time.monotonic() - astra_started
                if astra_components is not None:
                    env_steps = int(astra_components["broker"].env_steps)
                    phase_env_steps = int(astra_components["broker"].phase_pi05_steps)
                max_step_call = {
                    "env_step": boundary_env_step,
                    "phase_env_step": boundary_phase_step,
                    "policy_query_idx": int(boundary_query_idx),
                    "p_help": float(forced_outcome.decision.p_help),
                    "threshold": forced_outcome.decision.threshold,
                    "hard_threshold": forced_outcome.decision.hard_threshold,
                    "reason": forced_outcome.decision.reason,
                    "decision_rule": forced_outcome.decision.decision_rule,
                    "shadow_mode": bool(shadow_mode),
                    "takeover": bool(post_recovery_takeover_pending),
                }
                if shadow_mode:
                    shadow_call_events.append(
                        {
                            **max_step_call,
                            "max_step_trigger": True,
                        }
                    )
                    shadow_would_call = True
                    terminal_category = "SHADOW_WOULD_CALL_FAILURE"
                    terminal_reason = "max_steps_reached_before_success"
                    success = False
                    break

                call_decision = forced_outcome.decision
                handoff_response = forced_outcome.handoff_response
                handoff_detail = (
                    str(handoff_response.detail)
                    if handoff_response is not None and getattr(handoff_response, "detail", None)
                    else None
                )
                call_step = boundary_env_step
                call_query_idx = int(boundary_query_idx)
                p_help_at_call = float(forced_outcome.decision.p_help)
                if forced_outcome.state is ControllerState.RECOVERY_READY:
                    recovery = forced_outcome.recovery_outcome
                    if recovery is None or recovery.resume_observation is None:
                        raise RuntimeError(
                            "controller approved max-step recovery without a resume observation"
                        )
                    if astra_components is not None:
                        resume_decision, checkpoint = _resume_pi05_from_astra(
                            controller=controller,
                            recovery=recovery,
                            broker=astra_components["broker"],
                            call_step=boundary_env_step,
                            episode_id=episode_id,
                        )
                        observation = checkpoint.raw
                        env_steps = int(checkpoint.env_step)
                        phase_env_steps = int(astra_components["broker"].phase_pi05_steps)
                    else:
                        resume_decision = controller.resume_after_recovery()
                        observation = recovery.resume_observation
                        env_steps = int(resume_decision.resume_step)
                        phase_env_steps = int(resume_decision.resume_step)
                    action_queue.clear()
                    recovery_events.append(
                        {
                            "status": recovery.status,
                            "detail": recovery.detail,
                            "intervention_start_step": int(
                                recovery.intervention_start_step
                            ),
                            "intervention_end_step": int(
                                recovery.intervention_end_step
                            ),
                            "resume_step": int(recovery.resume_step),
                            "intervention_steps": int(
                                resume_decision.intervention_steps or 0
                            ),
                            "checkpoint_id": recovery.checkpoint_id,
                            "reentry_reason": resume_decision.reason,
                            "trigger": "max_steps_reached",
                        }
                    )
                    previous_progress = None
                    previous_progress_source = "recovery_resume"
                    previous_action_effort = None
                    continue
                if (
                    forced_outcome.recovery_outcome is not None
                    and forced_outcome.recovery_outcome.task_succeeded
                ):
                    terminal_category = "SUCCESS_DURING_RECOVERY"
                    terminal_reason = "task_succeeded_during_recovery"
                    success = True
                else:
                    terminal_category = "CALL_ASTRA"
                    terminal_reason = "max_steps_reached"
                    success = None
                break
            if not action_queue:
                query_env_step = int(env_steps)
                query_phase_env_step = int(phase_env_steps)
                policy_obs = observation_encoder(observation, task_instruction)
                current_progress, progress_source = _progress_snapshot(observation, env)
                progress_delta = _normalized_progress_delta(
                    current_progress, previous_progress
                )
                if isinstance(decision_engine, BudgetDecisionEngine):
                    decision_engine.set_context(
                        env_step=phase_env_steps, policy_query_idx=policy_queries
                    )
                elif isinstance(decision_engine, OracleCompetenceDecisionEngine):
                    if initial_q_pi is None:
                        raise ValueError("oracle baseline requires measured q_pi for this initial state")
                    decision_engine.set_context(q_pi=initial_q_pi)
                simulator_state, qpos, qvel = _copy_simulator_snapshot(env)
                if audit_state_dir is not None:
                    flattened_state = _flattened_state_for_audit(
                        env, simulator_state, qpos, qvel
                    )
                    if flattened_state is None:
                        raise ValueError(
                            "no-call failure auditing requires an exact flattened simulator state"
                        )
                    audit_state_history.append(
                        {
                            "policy_query_idx": int(policy_queries),
                            "env_step": int(env_steps),
                            "sim_state": flattened_state,
                        }
                    )
                if "observation/state" in policy_obs:
                    robot_state = np.asarray(
                        policy_obs["observation/state"], dtype=np.float32
                    )
                else:
                    robot_state = robot_state_from_observation(observation)
                pi05_started = time.monotonic()
                outcome = controller.process_policy_query(
                    policy_obs,
                    environment_observation=observation,
                    task_instruction=task_instruction,
                    episode_id=episode_id,
                    policy_query_idx=policy_queries,
                    env_step=env_steps,
                    robot_state=robot_state,
                    simulator_state=simulator_state,
                    bddl_path=str(bddl_path),
                    qpos=qpos,
                    qvel=qvel,
                )
                pi05_duration = time.monotonic() - pi05_started
                pi05_inference_seconds += pi05_duration
                if outcome.state in {ControllerState.HANDOFF, ControllerState.RECOVERY_READY}:
                    astra_inference_seconds += pi05_duration
                smoke_override = False
                normal_decision = outcome.decision
                if (
                    astra_components is not None
                    and smoke_force_call_after_query is not None
                    and not smoke_call_forced
                    and policy_queries == int(smoke_force_call_after_query)
                    and outcome.state is ControllerState.VLA_RUN
                ):
                    astra_started = time.monotonic()
                    outcome = controller.force_handoff(
                        controller.last_policy_result,
                        environment_observation=observation,
                        task_instruction=task_instruction,
                        episode_id=episode_id,
                        policy_query_idx=policy_queries,
                        env_step=env_steps,
                        robot_state=robot_state,
                        simulator_state=simulator_state,
                        bddl_path=str(bddl_path),
                        qpos=qpos,
                        qvel=qvel,
                        reason="astra_diagnostic_forced_smoke_call",
                    )
                    astra_inference_seconds += time.monotonic() - astra_started
                    smoke_override = True
                    smoke_call_forced = True
                if astra_components is not None:
                    env_steps = int(astra_components["broker"].env_steps)
                    phase_env_steps = int(astra_components["broker"].phase_pi05_steps)
                decision = outcome.decision
                query_log.append(
                    {
                        "policy_query_idx": policy_queries,
                        "env_step": query_env_step,
                        "phase_env_step": query_phase_env_step,
                        "p_help": float(decision.p_help),
                        "decision": decision.decision,
                        "reason": decision.reason,
                        "threshold": decision.threshold,
                        "hard_threshold": decision.hard_threshold,
                        "recent_scores": list(decision.recent_scores),
                        "decision_rule": decision.decision_rule,
                        "diagnostic_forced_call": smoke_override,
                        "decision_before_diagnostic_override": (
                            normal_decision.decision if smoke_override else None
                        ),
                        "score_source": score_source,
                        "cooldown_queries": int(cooldown_queries),
                        "cooldown_remaining": int(
                            getattr(decision, "cooldown_remaining", 0)
                        ),
                        "cooldown_suppressed": bool(
                            getattr(decision, "cooldown_suppressed", False)
                        ),
                        "progress_source": progress_source,
                        "progress_delta": progress_delta,
                        "previous_progress_source": previous_progress_source,
                        "previous_action_effort": previous_action_effort,
                        "recovery_status": (
                            outcome.recovery_outcome.status
                            if outcome.recovery_outcome is not None
                            else None
                        ),
                        "reentry_allowed": (
                            outcome.reentry_decision.allowed
                            if outcome.reentry_decision is not None
                            else None
                        ),
                    }
                )
                previous_progress = current_progress
                previous_progress_source = progress_source
                policy_queries += 1
                if shadow_mode and decision.decision == "CALL_ASTRA":
                    event = {
                        "policy_query_idx": int(policy_queries - 1),
                        "env_step": int(env_steps),
                        "p_help": float(decision.p_help),
                        "reason": decision.reason,
                        "threshold": decision.threshold,
                        "hard_threshold": decision.hard_threshold,
                        "recent_scores": list(decision.recent_scores),
                        "decision_rule": decision.decision_rule,
                        "cooldown_queries": int(cooldown_queries),
                    }
                    shadow_call_events.append(event)
                    shadow_would_call = True
                if outcome.state in {
                    ControllerState.HANDOFF,
                    ControllerState.RECOVERY_READY,
                }:
                    call_decision = decision
                    handoff_response = outcome.handoff_response
                    handoff_detail = (
                        str(handoff_response.detail)
                        if handoff_response is not None and getattr(handoff_response, "detail", None)
                        else None
                    )
                    call_step = query_env_step
                    call_query_idx = int(policy_queries - 1)
                    p_help_at_call = float(decision.p_help)
                    if outcome.state is ControllerState.RECOVERY_READY:
                        recovery = outcome.recovery_outcome
                        if recovery is None or recovery.resume_observation is None:
                            raise RuntimeError(
                                "controller approved recovery re-entry without a resume observation"
                            )
                        if astra_components is not None:
                            resume_decision, checkpoint = _resume_pi05_from_astra(
                                controller=controller,
                                recovery=recovery,
                                broker=astra_components["broker"],
                                call_step=int(call_step),
                                episode_id=episode_id,
                            )
                            observation = checkpoint.raw
                            env_steps = int(checkpoint.env_step)
                            phase_env_steps = int(astra_components["broker"].phase_pi05_steps)
                        else:
                            resume_decision = controller.resume_after_recovery()
                            observation = recovery.resume_observation
                            env_steps = int(resume_decision.resume_step)
                            phase_env_steps = int(resume_decision.resume_step)
                        action_queue.clear()
                        recovery_events.append(
                            {
                                "status": recovery.status,
                                "detail": recovery.detail,
                                "intervention_start_step": int(
                                    recovery.intervention_start_step
                                ),
                                "intervention_end_step": int(
                                    recovery.intervention_end_step
                                ),
                                "resume_step": int(recovery.resume_step),
                                "intervention_steps": int(
                                    resume_decision.intervention_steps or 0
                                ),
                                "checkpoint_id": recovery.checkpoint_id,
                                "reentry_reason": resume_decision.reason,
                            }
                        )
                        query_log[-1]["resume_step"] = int(recovery.resume_step)
                        query_log[-1]["intervention_steps"] = int(
                            resume_decision.intervention_steps or 0
                        )
                        previous_progress = None
                        previous_progress_source = "recovery_resume"
                        previous_action_effort = None
                        continue
                    if (
                        outcome.recovery_outcome is not None
                        and outcome.recovery_outcome.task_succeeded
                    ):
                        terminal_category = "SUCCESS_DURING_RECOVERY"
                        terminal_reason = "task_succeeded_during_recovery"
                        success = True
                    else:
                        terminal_category = "CALL_ASTRA"
                        terminal_reason = (
                            "astra_handoff_failed:" + handoff_detail
                            if handoff_response is not None
                            and handoff_response.status != "completed"
                            and handoff_detail
                            else decision.reason
                        )
                        recovery = outcome.recovery_outcome
                        known_astra_stop = bool(
                            recovery is not None
                            and not recovery.task_succeeded
                            and not recovery.execution_uncertain
                            and (
                                handoff_detail == "astra_requested_stop"
                                or recovery.detail == "astra_requested_stop"
                                or recovery.end_reason == "astra_requested_stop"
                                or recovery.status == "budget_exhausted"
                            )
                        )
                        known_episode_budget_stop = bool(
                            handoff_response is not None
                            and handoff_response.status == "recovery_budget_exhausted"
                            and handoff_detail in {
                                "episode_step_limit_reached",
                                "episode_step_budget_below_minimum_chunk",
                            }
                            and (
                                recovery is None
                                or (
                                    not recovery.task_succeeded
                                    and not recovery.execution_uncertain
                                )
                            )
                        )
                        if known_episode_budget_stop:
                            terminal_reason = "astra_budget_exhausted:" + str(
                                handoff_detail
                            )
                        # An explicit Astra stop or a known host-enforced step
                        # budget is an observed non-success. Reserve success=null
                        # for outcomes whose task result is genuinely unavailable
                        # (for example a transport or execution-ack uncertainty).
                        success = (
                            False
                            if known_astra_stop or known_episode_budget_stop
                            else None
                        )
                    break

                action_chunk = np.asarray(outcome.actions_to_execute, dtype=np.float32)
                if action_chunk.ndim != 2 or action_chunk.shape[1] != 7:
                    raise ValueError(f"unexpected π0.5 action chunk shape {action_chunk.shape}")
                if not np.isfinite(action_chunk).all():
                    raise ValueError("π0.5 action chunk contains non-finite values")
                if astra_components is not None:
                    action_chunk = astra_components["adapter"].validate_pi05_chunk(action_chunk).astype(np.float32)
                if replan_steps > len(action_chunk):
                    raise ValueError(
                        f"replan_steps={replan_steps} exceeds action chunk length {len(action_chunk)}"
                    )
                action_queue.extend(action_chunk[:replan_steps])
                if not action_queue:
                    raise ValueError("π0.5 returned an empty executable action prefix")
                effort = _action_effort(action_chunk[:replan_steps])
                query_log[-1]["planned_action_effort"] = effort
                previous_action_effort = effort

            action = action_queue.popleft()
            if astra_components is not None:
                step_result = astra_components["broker"].step(
                    action, source="pi05"
                )
                observation = step_result.checkpoint.raw
                env_steps = int(step_result.checkpoint.env_step)
                phase_env_steps = int(astra_components["broker"].phase_pi05_steps)
                pi05_step_success = bool(step_result.task_succeeded)
                done = bool(step_result.environment_ended)
            else:
                observation, reward, done, info = env.step(np.asarray(action).tolist())
                env_steps += 1
                phase_env_steps += 1
                info = info if isinstance(info, Mapping) else {}
                pi05_step_success = _task_succeeded(env, reward, bool(done), info)
            if video_recorder is not None:
                video_recorder.write(observation, env_steps, "pi05")
            if pi05_step_success:
                terminal_category = "SUCCESS_WITHOUT_CALL"
                terminal_reason = "task_success"
                success = True
                break
            if bool(done):
                terminal_category = "FAILURE_WITHOUT_CALL"
                terminal_reason = "environment_terminated_without_success"
                success = False
                break
        if terminal_category is None:
            if recovery_events:
                terminal_category = "FAILURE_AFTER_RECOVERY"
                terminal_reason = "post_recovery_step_budget_exhausted"
            else:
                terminal_category = "FAILURE_WITHOUT_CALL"
                terminal_reason = "step_budget_exhausted"
            success = False
        if shadow_mode and shadow_would_call:
            if success:
                terminal_category = "SHADOW_WOULD_CALL_SUCCESS"
                terminal_reason = "task_success_after_shadow_call"
            else:
                terminal_category = "SHADOW_WOULD_CALL_FAILURE"
                terminal_reason = "task_failure_after_shadow_call"
        if recovery_events:
            if terminal_category == "SUCCESS_WITHOUT_CALL":
                terminal_category = "SUCCESS_AFTER_RECOVERY"
                terminal_reason = "task_success_after_recovery"
            elif terminal_category == "FAILURE_WITHOUT_CALL":
                terminal_category = "FAILURE_AFTER_RECOVERY"
                terminal_reason = "task_failure_after_recovery"
        controller.finalize_episode()
    finally:
        if astra_components is not None:
            broker = astra_components["broker"]
            if broker.owner != "stopped" and not broker.task_succeeded:
                broker.stop(terminal_reason or terminal_category or "episode_closed")
            try:
                astra_components["journal"].append({
                    "event": "episode_runtime_finished",
                    "episode_id": str(episode_id),
                    "env_steps": int(broker.env_steps),
                    "pi05_total_steps": int(broker.pi05_total_steps),
                    "recovery_model_total_steps": int(broker.recovery_model_total_steps),
                    # Legacy report key retained for compatibility.
                    "astra_total_steps": int(broker.recovery_model_total_steps),
                    "pi05_phase_steps": int(broker.phase_pi05_steps),
                    "owner": str(broker.owner),
                    "task_succeeded": bool(broker.task_succeeded),
                    "terminal_category": terminal_category,
                    "terminal_reason": terminal_reason,
                    "execution_uncertain": bool(broker.uncertain),
                })
            except Exception:
                broker.stop("episode_finish_audit_failed")
            astra_components["history"].close()
            astra_components["journal"].close()
        try:
            env.close()
        except Exception:
            pass
        if video_recorder is not None:
            video_recorder.close()

    no_call_audit_states: list[dict[str, Any]] = []
    if terminal_category in {"FAILURE_WITHOUT_CALL", "SHADOW_WOULD_CALL_FAILURE"} and audit_state_dir is not None:
        if not audit_state_history:
            raise RuntimeError("failed episode has no captured policy-query simulator states")
        no_call_audit_states = _persist_no_call_audit_states(
            Path(audit_state_dir), episode_id, list(audit_state_history), max_steps=max_steps
        )

    return {
        "status": "complete",
        "episode_id": str(episode_id),
        "suite": str(suite),
        "task_id": int(task_id),
        "task_name": str(task_name),
        "task_instruction": str(task_instruction),
        "source": str(source),
        "initial_state_sample_id": initial_state_sample_id,
        "initial_q_pi": initial_q_pi,
        "initial_competence_stratum": (
            "low"
            if initial_q_pi is not None and initial_q_pi <= 0.2
            else "high"
            if initial_q_pi is not None and initial_q_pi >= 0.8
            else "mid"
            if initial_q_pi is not None
            else None
        ),
        "environment_seed": environment_seed,
        "policy_sampling_seed": policy_sampling_seed,
        "reset_retries": int(reset_retries),
        "success": success,
        "num_env_steps": int(env_steps),
        "episode_step_limit": int(episode_step_limit),
        "pi05_phase_steps": int(phase_env_steps),
        "num_policy_queries": int(policy_queries),
        "call_triggered": call_step is not None,
        "call_step": call_step,
        "call_query_idx": call_query_idx,
        "p_help_at_call": p_help_at_call,
        "recent_scores_at_call": (
            list(call_decision.recent_scores) if call_decision is not None else None
        ),
        "call_reason": call_decision.reason if call_decision is not None else None,
        "call_threshold": call_decision.threshold if call_decision is not None else None,
        "call_hard_threshold": (
            call_decision.hard_threshold if call_decision is not None else None
        ),
        "decision_rule": call_decision.decision_rule if call_decision is not None else None,
        "shadow_mode": bool(shadow_mode),
        "shadow_would_call": bool(shadow_would_call),
        "shadow_call_count": len(shadow_call_events),
        "shadow_first_call": shadow_call_events[0] if shadow_call_events else None,
        "shadow_call_events": shadow_call_events,
        "max_step_call": max_step_call,
        "recovery_events": recovery_events,
        "pi05_resumed_after_recovery": bool(recovery_events),
        "astra_runtime": (
            {
                "enabled": True,
                "runtime_root": str(astra_components["root"]),
                "host_private_dir": str(astra_components["private_dir"]),
                "codex_workspace": str(astra_components["workspace"]),
                "model": astra_components["model"],
                "reasoning_effort": astra_components["reasoning_effort"],
                "controller_audit": astra_components["controller_audit"],
                "pi05_total_steps": int(astra_components["broker"].pi05_total_steps),
                "recovery_model_total_steps": int(astra_components["broker"].recovery_model_total_steps),
                # Legacy report key retained for compatibility.
                "astra_total_steps": int(astra_components["broker"].recovery_model_total_steps),
                "final_owner": str(astra_components["broker"].owner),
                "execution_uncertain": bool(astra_components["broker"].uncertain),
                "provider_transport_summary": getattr(
                    astra_components["executor"], "_provider_transport_summary", None
                ),
                "smoke_forced_call": bool(smoke_call_forced),
            }
            if astra_components is not None else {"enabled": False}
        ),
        "cooldown_queries": int(cooldown_queries),
        "progress_telemetry": {
            "enabled": True,
            "sources": sorted(
                {
                    str(item.get("progress_source"))
                    for item in query_log
                    if item.get("progress_source") is not None
                }
            ),
            "query_fields": [
                "progress_delta",
                "previous_action_effort",
                "planned_action_effort",
            ],
        },
        "terminal_category": terminal_category,
        "terminal_reason": terminal_reason,
        "remaining_steps_if_call": None,
        "handoff_status": handoff_response.status if handoff_response is not None else None,
        "handoff_detail": handoff_detail,
        "call_snapshot_path": (
            handoff_response.snapshot_path if handoff_response is not None else None
        ),
        "recovery_model_called": bool(
            handoff_response is not None
            and handoff_response.recovery is not None
            and handoff_response.recovery.astra_called
        ) or bool(recovery_events),
        "recovery_executed": bool(
            handoff_response is not None
            and handoff_response.recovery is not None
            and handoff_response.recovery.recovery_executed
        ) or any(
            str(event.get("status")) == "completed"
            for event in recovery_events
            if isinstance(event, Mapping)
        ),
        # Legacy report key: Astra is the selected recovery model name.
        "astra_called": bool(
            handoff_response is not None
            and handoff_response.recovery is not None
            and handoff_response.recovery.astra_called
        ) or bool(recovery_events),
        "max_steps": int(max_steps),
        "replan_steps": int(replan_steps),
        "elapsed_seconds": time.monotonic() - started,
        "pi05_inference_seconds": float(pi05_inference_seconds),
        "astra_inference_seconds": float(astra_inference_seconds),
        "video_path": str(video_path) if video_path is not None else None,
        "query_log": query_log,
        "no_call_audit_states": no_call_audit_states,
    }


def _make_env_with_reset(
    bddl_path: Path,
    seed: int,
    *,
    camera_size: int,
    episode_horizon: int = 1000,
    max_reset_retries: int,
) -> tuple[Any, Mapping[str, Any], int]:
    if int(episode_horizon) <= 0:
        raise ValueError("episode_horizon must be positive")
    from libero.libero.envs import OffScreenRenderEnv

    last_error: Exception | None = None
    for retry in range(max_reset_retries + 1):
        env = None
        try:
            env = OffScreenRenderEnv(
                bddl_file_name=str(bddl_path),
                camera_heights=camera_size,
                camera_widths=camera_size,
                horizon=int(episode_horizon),
            )
            env.seed(int(seed) + retry)
            return env, env.reset(), retry
        except Exception as exc:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
            if type(exc).__name__ != "RandomizationError":
                raise
            last_error = exc
    raise RuntimeError(
        f"LIBERO placement reset failed after {max_reset_retries + 1} attempts: {last_error}"
    ) from last_error


def _prepare_libero(openpi_root: Path) -> tuple[Any, dict[int, tuple[str, str, Path]]]:
    libero_root = openpi_root / "third_party" / "libero"
    for path in (openpi_root / "src", libero_root, EXPERIMENT_ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from scripts.libero_compat import patch_torch_load_for_libero

    patch_torch_load_for_libero()
    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark_dict

    return get_libero_path, get_benchmark_dict()


class LoadedAssistHead:
    def __init__(self, checkpoint: Mapping[str, Any], model: Any, torch_module: Any) -> None:
        self.checkpoint = dict(checkpoint)
        self.model = model.eval()
        self.torch = torch_module
        self.feature_set = str(checkpoint["feature_set"])
        self.mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
        self.scale = np.asarray(checkpoint["feature_scale"], dtype=np.float32)
        self.temperature = float(checkpoint["temperature"])
        calibration = checkpoint["operating_points"]
        self.operating_points = calibration
        self.hard_point = checkpoint.get("hard_call_operating_point") or {}
        if self.mean.ndim != 1 or self.scale.shape != self.mean.shape:
            raise ValueError("head checkpoint has malformed feature standardization arrays")
        if not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all():
            raise ValueError("head checkpoint has non-finite feature standardization values")
        if np.any(self.scale <= 0.0) or self.temperature <= 0.0:
            raise ValueError("head checkpoint has invalid scale or calibration temperature")
        if not math.isfinite(self.temperature):
            raise ValueError("head checkpoint temperature is non-finite")

    def threshold(self, threshold_point: str) -> tuple[float, float | None]:
        point = self.operating_points.get(threshold_point)
        if not isinstance(point, Mapping) or point.get("status") != "selected":
            raise ValueError(
                f"validation did not select the {threshold_point!r} threshold "
                f"for held-out task {self.checkpoint.get('test_task')}"
            )
        threshold = point.get("threshold")
        if threshold is None:
            raise ValueError(f"selected {threshold_point} point has no threshold")
        hard_threshold = (
            self.hard_point.get("threshold")
            if self.hard_point.get("status") == "selected"
            else None
        )
        if hard_threshold is not None and float(hard_threshold) < float(threshold):
            raise ValueError(
                "calibrated hard threshold is below the selected normal threshold; "
                "refusing an ambiguous CALL configuration"
            )
        return float(threshold), None if hard_threshold is None else float(hard_threshold)

    def predict(self, policy_result: Mapping[str, Any]) -> float:
        from calibrate_call_head import apply_temperature

        vector = feature_vector_from_policy_result(policy_result, self.feature_set)
        if vector.shape != self.mean.shape:
            raise ValueError(
                f"head expects {self.mean.size} input features but got {vector.size}"
            )
        normalized = np.clip((vector - self.mean) / self.scale, -20.0, 20.0)
        tensor = self.torch.from_numpy(normalized[None, :].astype(np.float32))
        with self.torch.no_grad():
            logit = self.model(tensor)
            raw_probability = self.torch.sigmoid(logit).cpu().numpy().reshape(-1)
        calibrated = apply_temperature(raw_probability, self.temperature)
        score = float(calibrated[0])
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("AssistMLP emitted an invalid calibrated p_help")
        return score


def load_task_heads(
    model_dir: Path, *, threshold_point: str, require_threshold: bool = True
) -> dict[str, LoadedAssistHead]:
    import torch

    from call_llm.call_assist_head import build_call_assist_mlp

    from call_llm.checkpoint_provenance import (
        OFFICIAL_PI05_LIBERO_CONFIG,
        OFFICIAL_PI05_LIBERO_SOURCE_URI,
    )

    folder = model_dir / "fold_models"
    paths = sorted(folder.glob("fold_*.pt")) if folder.is_dir() else []
    if folder.is_dir():
        paths.extend(sorted(folder.glob("fallback_*.pt")))
    if not paths:
        raise FileNotFoundError(f"no fold checkpoints found under {folder}")
    result: dict[str, LoadedAssistHead] = {}
    for path in paths:
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # Older supported PyTorch builds do not expose weights_only.
            checkpoint = torch.load(path, map_location="cpu")
        provenance = checkpoint.get("checkpoint_provenance", {})
        if (
            provenance.get("checkpoint_id") != "openpi_pi05_libero"
            or provenance.get("checkpoint_source_uri") != OFFICIAL_PI05_LIBERO_SOURCE_URI
            or provenance.get("policy_config") != OFFICIAL_PI05_LIBERO_CONFIG
            or provenance.get("weight_role") != "official_pi0.5_LIBERO_30k_finetuned_checkpoint"
        ):
            raise ValueError(f"head checkpoint lacks verified official pi05_libero provenance: {path}")
        test_task = str(checkpoint.get("test_task", ""))
        if not test_task or test_task in result:
            raise ValueError(f"missing or duplicate held-out task in head checkpoint {path}")
        feature_set = str(checkpoint.get("feature_set", ""))
        model_config = checkpoint.get("model_config", {})
        model = build_call_assist_mlp(
            hidden_size=int(model_config["hidden_size"]), feature_set=feature_set
        )
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        head = LoadedAssistHead(checkpoint, model, torch)
        # Validate thresholds before any environment is created when the learned
        # detector is the decision source. Fixed baselines only use the score as
        # a diagnostic and do not depend on an operating point.
        if require_threshold:
            head.threshold(threshold_point)
        result[test_task] = head
    return result


def _task_specs(
    suite_name: str,
    task_ids: Sequence[int] | None,
    *,
    get_libero_path: Callable[[str], str],
    benchmark_dict: Mapping[str, Any],
) -> dict[int, tuple[str, str, Path]]:
    suite_factory = benchmark_dict.get(suite_name)
    if suite_factory is None:
        raise ValueError(f"LIBERO benchmark suite {suite_name!r} is not installed")
    suite = suite_factory()
    selected = list(range(int(suite.n_tasks))) if task_ids is None else [int(i) for i in task_ids]
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("task ids must be a nonempty list without duplicates")
    bddl_root = Path(get_libero_path("bddl_files")).resolve()
    result: dict[int, tuple[str, str, Path]] = {}
    for task_id in selected:
        if task_id < 0 or task_id >= int(suite.n_tasks):
            raise ValueError(f"task id {task_id} is outside suite {suite_name}")
        task = suite.get_task(task_id)
        bddl_path = (bddl_root / task.problem_folder / task.bddl_file).resolve()
        if not bddl_path.is_file():
            raise FileNotFoundError(f"official task BDDL is missing: {bddl_path}")
        result[task_id] = (
            str(getattr(task, "name", task.bddl_file)),
            str(task.language),
            bddl_path,
        )
    return result


def normalize_state_rows(
    rows: Sequence[Mapping[str, Any]], table_path: Path
) -> list[dict[str, Any]]:
    """Validate state-table rows and resolve their files relative to the table."""
    table_path = table_path.expanduser().resolve()
    base_dir = table_path.parent
    if not rows:
        raise ValueError(f"state table is empty: {table_path}")
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for source_row in rows:
        row = dict(source_row)
        sample_id = str(row.get("sample_id", ""))
        if not sample_id or sample_id in seen:
            raise ValueError("state table needs unique, nonempty sample_id values")
        seen.add(sample_id)
        if row.get("suite") is None or row.get("task_id") is None:
            raise ValueError(f"state {sample_id} lacks suite/task_id")

        def resolve_file(value: Any, field: str) -> str:
            if value is None or not str(value).strip():
                raise ValueError(f"state {sample_id} lacks {field}")
            candidate = Path(str(value)).expanduser()
            if not candidate.is_absolute():
                candidate = base_dir / candidate
            candidate = candidate.resolve()
            if not candidate.is_file():
                raise FileNotFoundError(
                    f"state {sample_id} references missing {field}: {candidate}"
                )
            return str(candidate)

        bddl_value = (
            row.get("restore_bddl_path")
            or row.get("bddl_path")
            or row.get("official_bddl_path")
        )
        if not bddl_value:
            raise FileNotFoundError(f"state {sample_id} has no readable BDDL path")
        row["resolved_bddl_path"] = resolve_file(bddl_value, "BDDL path")

        simulator_state_value = row.get("sim_state_path") or row.get("state_path")
        has_inline_state = row.get("sim_state") is not None
        has_qpos_state = row.get("qpos") is not None or row.get("qpos_path") is not None
        if simulator_state_value:
            row["resolved_sim_state_path"] = resolve_file(
                simulator_state_value, "simulator state path"
            )
        elif not has_inline_state and not has_qpos_state:
            raise ValueError(f"state {sample_id} has no restorable simulator state")

        for field in ("qpos_path", "qvel_path"):
            if row.get(field) is not None:
                row[f"resolved_{field}"] = resolve_file(row[field], field)
        if row.get("resolved_qpos_path") and not (
            row.get("qvel") is not None or row.get("resolved_qvel_path")
        ):
            raise ValueError(f"state {sample_id} has qpos but no qvel")
        if row.get("resolved_qvel_path") and not (
            row.get("qpos") is not None or row.get("resolved_qpos_path")
        ):
            raise ValueError(f"state {sample_id} has qvel but no qpos")
        if not row.get("resolved_sim_state_path") and not has_inline_state and has_qpos_state:
            if row.get("env_step") is None or row.get("control_freq") is None:
                raise ValueError(
                    f"state {sample_id} qpos/qvel restore requires env_step and control_freq"
                )
            if float(row["control_freq"]) <= 0:
                raise ValueError(f"state {sample_id} has invalid control_freq")

        if row.get("q_pi") is not None:
            q_pi = float(row["q_pi"])
            if not math.isfinite(q_pi) or not 0.0 <= q_pi <= 1.0:
                raise ValueError(f"state {sample_id} has invalid q_pi")
            row["q_pi"] = q_pi
        row["sample_id"] = sample_id
        row["suite"] = str(row["suite"])
        row["task_id"] = int(row["task_id"])
        if row["task_id"] < 0:
            raise ValueError(f"state {sample_id} has negative task_id")
        normalized.append(row)
    return sorted(normalized, key=lambda row: row["sample_id"])


def read_state_rows(path: Path) -> list[dict[str, Any]]:
    """Read exact-start-state cases, such as the 207 validated Recovery Assets."""
    import pyarrow.parquet as pq

    path = path.expanduser().resolve()
    rows = [dict(row) for row in pq.read_table(path).to_pylist()]
    return normalize_state_rows(rows, path)


def select_state_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    suite: str,
    task_ids: Sequence[int] | None,
    require_oracle_labels: bool = False,
) -> list[dict[str, Any]]:
    selected = [dict(row) for row in rows]
    suites = {str(row.get("suite")) for row in selected}
    if suites != {str(suite)}:
        raise ValueError(
            f"state table suite(s) {sorted(suites)} do not match requested LIBERO suite {suite!r}"
        )
    if task_ids is not None:
        requested = {int(task_id) for task_id in task_ids}
        available = {int(row["task_id"]) for row in selected}
        missing = requested - available
        if missing:
            raise ValueError(f"requested task ids have no state rows: {sorted(missing)}")
        selected = [row for row in selected if int(row["task_id"]) in requested]
    if not selected:
        raise ValueError("no state rows remain after suite/task selection")
    if require_oracle_labels:
        invalid_source = sorted(
            {
                str(row.get("source"))
                for row in selected
                if row.get("source") != "recovery_asset"
            }
        )
        if invalid_source:
            raise ValueError(
                "Oracle competence is restricted to Recovery Assets; "
                f"found source value(s): {invalid_source}"
            )
        missing_q = [row["sample_id"] for row in selected if row.get("q_pi") is None]
        if missing_q:
            raise ValueError(
                f"Oracle baseline requires measured q_pi for every state; missing {len(missing_q)}"
            )
        for row in selected:
            if row.get("num_trials") is not None and int(row["num_trials"]) <= 0:
                raise ValueError(f"state {row['sample_id']} has no competence trials")
    return sorted(selected, key=lambda row: row["sample_id"])


def state_row_episode_cases(
    rows: Sequence[Mapping[str, Any]],
    tasks: Mapping[int, tuple[str, str, Path]],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    per_task_index: dict[int, int] = {}
    for row_index, row in enumerate(rows):
        task_id = int(row["task_id"])
        if task_id not in tasks:
            raise ValueError(f"state {row['sample_id']} has no metadata for task {task_id}")
        default_name, default_instruction, _ = tasks[task_id]
        task_index = per_task_index.get(task_id, 0)
        per_task_index[task_id] = task_index + 1
        source = str(row.get("source") or "static_state")
        sample_id = str(row["sample_id"])
        cases.append(
            {
                "episode_id": f"{_safe_segment(source)}_state_{row_index:04d}_{sample_id}",
                "suite": str(row["suite"]),
                "task_id": task_id,
                "task_name": str(row.get("task_name") or default_name),
                "task_instruction": str(row.get("task_instruction") or default_instruction),
                "bddl_path": Path(str(row["resolved_bddl_path"])),
                "source": source,
                "state_row": dict(row),
                "episode_index": task_index,
                "environment_seed": int(seed + task_id * 100_003 + task_index),
                "sampling_seed": int(seed + 50_000_000 + task_id * 100_003 + task_index),
            }
        )
    return cases


def load_initial_simulator_state(row: Mapping[str, Any]) -> np.ndarray:
    state_value = row.get("sim_state")
    state_path = (
        row.get("resolved_sim_state_path")
        or row.get("sim_state_path")
        or row.get("state_path")
    )
    if state_path:
        path = Path(str(state_path)).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"simulator state file is missing: {path}")
        state = np.load(path, allow_pickle=False)
    elif state_value is not None:
        state = np.asarray(state_value, dtype=np.float64)
    else:
        qpos_value = row.get("qpos")
        qvel_value = row.get("qvel")
        qpos_path = row.get("resolved_qpos_path") or row.get("qpos_path")
        qvel_path = row.get("resolved_qvel_path") or row.get("qvel_path")
        if qpos_value is None and qpos_path:
            qpos_value = np.load(Path(str(qpos_path)), allow_pickle=False)
        if qvel_value is None and qvel_path:
            qvel_value = np.load(Path(str(qvel_path)), allow_pickle=False)
        if qpos_value is None or qvel_value is None:
            raise ValueError(f"state {row.get('sample_id')} needs both qpos and qvel")
        control_freq = float(row.get("control_freq") or 0.0)
        if control_freq <= 0.0 or row.get("env_step") is None:
            raise ValueError(
                "qpos/qvel state restoration requires env_step and positive control_freq"
            )
        simulation_time = float(row["env_step"]) / control_freq
        state = np.concatenate(
            (
                np.asarray([simulation_time], dtype=np.float64),
                np.asarray(qpos_value, dtype=np.float64).reshape(-1),
                np.asarray(qvel_value, dtype=np.float64).reshape(-1),
            )
        )
    state = np.asarray(state, dtype=np.float64)
    if state.ndim != 1 or not state.size or not np.isfinite(state).all():
        raise ValueError(f"state {row.get('sample_id')} is not a finite flattened simulator state")
    return state


def restore_observation_from_state(env: Any, row: Mapping[str, Any]) -> Mapping[str, Any]:
    restore = getattr(env, "regenerate_obs_from_state", None)
    if not callable(restore):
        raise TypeError("LIBERO environment does not expose regenerate_obs_from_state")
    return restore(load_initial_simulator_state(row))


def _safe_segment(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "unknown"


def _persist_no_call_audit_states(
    audit_state_root: Path,
    episode_id: str,
    history: Sequence[Mapping[str, Any]],
    *,
    max_steps: int,
) -> list[dict[str, Any]]:
    if not history:
        return []
    parent = audit_state_root / _safe_segment(episode_id)
    parent.mkdir(parents=True, exist_ok=True)
    existing_attempts = [
        int(match.group(1))
        for child in parent.glob("attempt_*")
        if child.is_dir()
        for match in [re.fullmatch(r"attempt_(\d+)", child.name)]
        if match
    ]
    attempt_dir = parent / f"attempt_{max(existing_attempts, default=0) + 1:02d}"
    attempt_dir.mkdir(parents=False, exist_ok=False)

    records: list[dict[str, Any]] = []
    available = len(history)
    for index, item in enumerate(history):
        distance_from_latest = available - index - 1
        windows = []
        if distance_from_latest == 0:
            windows.append("last_1")
        if distance_from_latest < 3:
            windows.append("last_3")
        if distance_from_latest < 5:
            windows.append("last_5")
        query_idx = int(item["policy_query_idx"])
        env_step = int(item["env_step"])
        state_path = attempt_dir / f"query_{query_idx:05d}.npy"
        _atomic_numpy_save(state_path, np.asarray(item["sim_state"], dtype=np.float64))
        records.append(
            {
                "state_id": f"failure_query_{query_idx:05d}",
                "sample_id": f"{_safe_segment(episode_id)}__failure_query_{query_idx:05d}",
                "policy_query_idx": query_idx,
                "env_step": env_step,
                "state_step": env_step,
                "episode_progress": min(1.0, max(0.0, env_step / max_steps)),
                "sim_state_path": str(state_path.resolve()),
                "audit_windows": windows,
                "audit_trial_target": 5,
            }
        )
    return records


def _episode_path(output_dir: Path, episode_id: str) -> Path:
    return output_dir / "episodes" / f"{_safe_segment(episode_id)}.json"


def _next_snapshot_root(output_dir: Path, episode_id: str) -> Path:
    parent = output_dir / "call_snapshots" / _safe_segment(episode_id)
    parent.mkdir(parents=True, exist_ok=True)
    existing = []
    for path in parent.glob("attempt_*"):
        match = re.fullmatch(r"attempt_(\d+)", path.name)
        if path.is_dir() and match:
            existing.append(int(match.group(1)))
    attempt = max(existing, default=0) + 1
    return parent / f"attempt_{attempt:02d}"


def _next_astra_runtime_root(output_dir: Path, episode_id: str) -> Path:
    parent = output_dir / "astra_runtime" / _safe_segment(episode_id)
    parent.mkdir(parents=True, exist_ok=True)
    existing = []
    for path in parent.glob("attempt_*"):
        match = re.fullmatch(r"attempt_(\d+)", path.name)
        if path.is_dir() and match:
            existing.append(int(match.group(1)))
    return parent / f"attempt_{max(existing, default=0) + 1:02d}"


def _write_episode_parquet(output_dir: Path) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    paths = sorted((output_dir / "episodes").glob("*.json"))
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if not rows:
        raise ValueError("no completed episode records to aggregate")
    table = pa.Table.from_pylist(rows)
    target = output_dir / "episodes.parquet"
    temporary = target.with_suffix(".parquet.tmp")
    pq.write_table(table, temporary)
    os.replace(temporary, target)
    return len(rows)


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--openpi-root", type=Path, default=DEFAULT_OPENPI_ROOT)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Trained model variant directory; required only for --baseline learned",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--task-ids", type=int, nargs="*", default=None)
    parser.add_argument(
        "--states-parquet",
        type=Path,
        default=None,
        help="Optional exact-start-state table; each row becomes one episode instead of random reset episodes",
    )
    parser.add_argument("--episodes-per-task", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=520)
    parser.add_argument(
        "--episode-horizon",
        type=int,
        default=1000,
        help="hard LIBERO environment-step limit for the entire episode (independent of --max-steps)",
    )
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--camera-size", type=int, default=256)
    parser.add_argument("--reset-retries", type=int, default=12)
    parser.add_argument("--gpu-index", type=int, default=1)
    parser.add_argument("--gpu-max-memory-mib", type=int, default=2048)
    parser.add_argument("--gpu-max-utilization", type=int, default=5)
    parser.add_argument("--threshold-point", choices=("conservative", "balanced", "aggressive"), default="balanced")
    parser.add_argument(
        "--decision-rule",
        choices=("instant", "2_of_3", "3_of_5", "4_of_5", "hysteresis"),
        default="4_of_5",
    )
    parser.add_argument(
        "--shadow-mode",
        action="store_true",
        help="record would-call decisions but continue π0.5 to its natural outcome",
    )
    parser.add_argument(
        "--cooldown-queries",
        type=int,
        default=6,
        help="suppress repeated learned CALL decisions for this many policy queries after a trigger",
    )
    parser.add_argument(
        "--astra-min-call-steps",
        type=int,
        default=20,
        help="minimum π0.5 environment steps between real Astra CALL handoffs",
    )
    parser.add_argument(
        "--baseline",
        choices=("learned", "never_call", "fixed_step", "fixed_query", "oracle"),
        default="learned",
    )
    parser.add_argument("--budget", type=int, default=None, help="Steps or policy queries for the selected fixed-budget baseline")
    parser.add_argument("--enable-astra", action="store_true", help="enable the real Codex CLI Astra controller for CALL handoffs")
    parser.add_argument(
        "--astra-chunk-edit",
        action="store_true",
        help="optional alternative: let Astra edit the current Pi0.5 waypoint chunk instead of using take-over",
    )
    parser.add_argument(
        "--astra-chunk",
        action="store_true",
        help="optional alternative: let Astra return and execute a 30-50 action EEF chunk",
    )
    parser.add_argument("--astra-smoke-force-call", action="store_true", help="diagnostic only: force one CALL after π0.5 has executed its first action segment; requires exactly one task and one episode")
    parser.add_argument("--astra-model", default=os.environ.get("ASTRA_MODEL", "gpt-6-luna"))
    parser.add_argument(
        "--astra-reasoning-effort",
        choices=("low", "medium", "high", "xhigh", "max"),
        default=os.environ.get("ASTRA_REASONING_EFFORT", "medium"),
    )
    parser.add_argument("--astra-max-decisions", type=int, default=25)
    parser.add_argument("--astra-max-execution-chunks", type=int, default=25)
    parser.add_argument("--astra-max-tool-calls", type=int, default=100)
    parser.add_argument("--astra-max-wall-seconds", type=float, default=300.0)
    parser.add_argument("--resume", action="store_true", help="Resume only when the saved run configuration matches exactly")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _make_parser().parse_args(argv)
    if args.checkpoint is None:
        raise SystemExit("set PI05_CHECKPOINT or pass --checkpoint")
    if args.openpi_root is None:
        raise SystemExit("set OPENPI_ROOT or pass --openpi-root")
    if (args.episodes_per_task <= 0 or args.max_steps <= 0 or
            args.episode_horizon <= 0 or args.replan_steps <= 0):
        raise SystemExit(
            "episodes-per-task, max-steps, episode-horizon, and replan-steps must be positive"
        )
    if args.reset_retries < 0 or args.num_steps <= 0 or args.camera_size <= 0:
        raise SystemExit("reset-retries must be nonnegative; num-steps and camera-size must be positive")
    if args.cooldown_queries < 0:
        raise SystemExit("cooldown-queries must be nonnegative")
    if args.astra_min_call_steps < 0:
        raise SystemExit("astra-min-call-steps must be nonnegative")
    if args.astra_chunk_edit and args.astra_chunk:
        raise SystemExit("astra-chunk-edit and astra-chunk are mutually exclusive")
    if args.baseline in {"fixed_step", "fixed_query"} and (args.budget is None or args.budget <= 0):
        raise SystemExit("fixed-step and fixed-query baselines require a positive --budget")
    if args.baseline in {"learned", "never_call", "oracle"} and args.budget is not None:
        raise SystemExit("--budget is only valid for fixed-step or fixed-query baselines")
    if args.baseline == "oracle" and args.states_parquet is None:
        raise SystemExit("the oracle baseline is only available with --states-parquet")
    if args.baseline == "learned" and args.model_dir is None:
        raise SystemExit("--model-dir is required for --baseline learned")
    if args.astra_smoke_force_call and not args.enable_astra:
        raise SystemExit("--astra-smoke-force-call requires --enable-astra")
    if args.enable_astra:
        if args.baseline != "learned" or args.model_dir is None:
            raise SystemExit("live Astra requires the trained learned CALL head")
        if args.astra_model not in {"gpt-6-luna", "gpt-6-astra"}:
            raise SystemExit("live Astra requires an explicitly supported GPT-6 model; provider/model fallback is disabled")
        if args.shadow_mode:
            raise SystemExit("live Astra cannot be combined with --shadow-mode")
        if args.decision_rule != "4_of_5":
            raise SystemExit("live Astra requires the configured 4_of_5 CALL rule")
        if (args.astra_max_decisions < 1 or args.astra_max_execution_chunks < 1 or
                args.astra_max_tool_calls < 1 or args.astra_max_wall_seconds <= 0):
            raise SystemExit("Astra intervention budgets must be positive")
        if args.astra_smoke_force_call:
            if args.states_parquet is not None or args.episodes_per_task != 1 or args.task_ids is None or len(args.task_ids) != 1:
                raise SystemExit("diagnostic Astra smoke requires one explicit task id, one episode, and a normal LIBERO reset")
    state_rows: list[dict[str, Any]] | None = None
    state_table_info: dict[str, Any] | None = None
    if args.states_parquet is not None:
        state_table_path = args.states_parquet.expanduser().resolve()
        try:
            all_state_rows = read_state_rows(state_table_path)
            state_rows = select_state_rows(
                all_state_rows,
                suite=args.suite,
                task_ids=args.task_ids,
                require_oracle_labels=args.baseline == "oracle",
            )
        except (ImportError, OSError, ValueError, KeyError, TypeError) as exc:
            raise SystemExit(f"state-table preflight failed: {exc}") from exc
        state_table_stat = state_table_path.stat()
        state_table_info = {
            "path": str(state_table_path),
            "size_bytes": int(state_table_stat.st_size),
            "mtime_ns": int(state_table_stat.st_mtime_ns),
            "selected_rows": len(state_rows),
            "source_counts": dict(Counter(str(row.get("source") or "unknown") for row in state_rows)),
        }
    if not 0 <= args.gpu_index:
        raise SystemExit("gpu-index must be nonnegative")
    if args.replan_steps > 10:
        raise SystemExit("replan-steps cannot exceed the π0.5 ten-action chunk")
    try:
        gpu_memory, gpu_utilization = assert_gpu_idle(
            args.gpu_index,
            max_memory_mib=args.gpu_max_memory_mib,
            max_utilization_percent=args.gpu_max_utilization,
        )
    except (OSError, subprocess.SubprocessError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"GPU preflight refused inference: {exc}") from exc

    from call_llm.checkpoint_provenance import (
        official_pi05_libero_metadata,
        validate_official_pi05_libero_checkpoint,
    )

    try:
        checkpoint = validate_official_pi05_libero_checkpoint(args.checkpoint)
        heads = (
            load_task_heads(
                args.model_dir.expanduser().resolve(),
                threshold_point=args.threshold_point,
                require_threshold=args.baseline == "learned",
            )
            if args.model_dir is not None
            else {}
        )
    except (ValueError, FileNotFoundError, KeyError, RuntimeError) as exc:
        raise SystemExit(f"model preflight failed: {exc}") from exc
    model_variant = args.model_dir.name if args.model_dir is not None else "none"

    selected_task_ids = (
        sorted({int(row["task_id"]) for row in state_rows})
        if state_rows is not None
        else args.task_ids
    )
    run_config = {
        "suite": args.suite,
        "task_ids": selected_task_ids,
        "episodes_per_task": None if state_rows is not None else args.episodes_per_task,
        "state_table": state_table_info,
        "seed": args.seed,
        "num_steps": args.num_steps,
        "max_steps": args.max_steps,
        "episode_horizon": args.episode_horizon,
        "replan_steps": args.replan_steps,
        "camera_size": args.camera_size,
        "reset_retries": args.reset_retries,
        "gpu_index": args.gpu_index,
        "gpu_max_memory_mib": args.gpu_max_memory_mib,
        "gpu_max_utilization": args.gpu_max_utilization,
        "openpi_root": str(args.openpi_root.expanduser().resolve()),
        "threshold_point": args.threshold_point,
        "decision_rule": args.decision_rule,
        "shadow_mode": bool(args.shadow_mode),
        "cooldown_queries": int(args.cooldown_queries),
        "astra_min_call_steps": int(args.astra_min_call_steps),
        "astra_runtime": {
            "enabled": bool(args.enable_astra),
            "chunk_edit_enabled": bool(args.astra_chunk_edit) if args.enable_astra else False,
            "chunk_mode_enabled": bool(args.astra_chunk) if args.enable_astra else False,
            "model": args.astra_model if args.enable_astra else None,
            "reasoning_effort": args.astra_reasoning_effort if args.enable_astra else None,
            "provider_fallback": False,
            "max_control_steps": None,
            "max_decisions": int(args.astra_max_decisions) if args.enable_astra else None,
            "max_execution_chunks": int(args.astra_max_execution_chunks) if args.enable_astra else None,
            "max_tool_calls": int(args.astra_max_tool_calls) if args.enable_astra else None,
            "max_wall_seconds": float(args.astra_max_wall_seconds) if args.enable_astra else None,
            "smoke_force_call_after_query": 1 if args.astra_smoke_force_call else None,
            "diagnostic_not_formal_evaluation": bool(args.astra_smoke_force_call),
        },
        "baseline": args.baseline,
        "budget": args.budget,
        "no_call_failure_audit": {
            "trigger": "FAILURE_WITHOUT_CALL",
            "saved_policy_queries": "last_up_to_5",
            "evaluation_windows": [1, 3, 5],
            "state_format": "flattened_libero_simulator_state",
        },
        "model_dir": (
            str(args.model_dir.expanduser().resolve())
            if args.model_dir is not None
            else None
        ),
        "checkpoint": str(checkpoint),
        "checkpoint_provenance": official_pi05_libero_metadata(checkpoint),
    }
    output_dir = args.output_dir.expanduser().resolve()
    config_path = output_dir / "run_config.json"
    if output_dir.exists():
        if not args.resume:
            raise SystemExit(f"refusing to overwrite existing run directory: {output_dir}")
        if not config_path.is_file():
            raise SystemExit("--resume requires a prior run_config.json")
        if json.loads(config_path.read_text(encoding="utf-8")) != run_config:
            raise SystemExit("saved run configuration differs; refusing to mix experiment settings")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
        _atomic_json(config_path, run_config)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    get_libero_path, benchmark_dict = _prepare_libero(args.openpi_root.resolve())
    from scripts.libero_compat import patch_torch_load_for_libero

    patch_torch_load_for_libero()
    from call_llm.feature_policy import Pi05FeaturePolicy
    import jax

    tasks = _task_specs(
        args.suite,
        selected_task_ids,
        get_libero_path=get_libero_path,
        benchmark_dict=benchmark_dict,
    )
    if args.astra_smoke_force_call:
        if args.suite != "libero_10" or len(tasks) != 1:
            raise SystemExit("diagnostic Astra smoke is pinned to one LIBERO-10 task")
        only_task_key = f"{args.suite}:{next(iter(tasks))}"
        if only_task_key not in heads:
            raise SystemExit("diagnostic Astra smoke requires that task's trained CALL head")
    expected_head_tasks = {f"{args.suite}:{task_id}" for task_id in tasks}
    fallback_head_key = f"{args.suite}:__fallback__"
    missing_heads = expected_head_tasks - set(heads)
    if args.baseline == "learned" and missing_heads and fallback_head_key not in heads:
        raise SystemExit(f"no held-out AssistMLP checkpoint for task(s): {sorted(missing_heads)}")
    policy = Pi05FeaturePolicy(str(checkpoint), seed=args.seed, num_steps=args.num_steps)

    _atomic_json(
        output_dir / "runtime_provenance.json",
        {
            "policy": official_pi05_libero_metadata(checkpoint),
            "head_dir": (
                str(args.model_dir.expanduser().resolve())
                if args.model_dir is not None
                else None
            ),
            "head_tasks": sorted(heads),
            "gpu_preflight": {
                "physical_gpu": args.gpu_index,
                "memory_used_mib": gpu_memory,
                "utilization_percent": gpu_utilization,
            },
            "recovery_model_called": False,
            "astra_called": False,
            "recovery_executed": False,
            "shadow_mode": bool(args.shadow_mode),
            "astra_enabled": bool(args.enable_astra),
            "episode_horizon": int(args.episode_horizon),
        },
    )

    if state_rows is not None:
        cases = state_row_episode_cases(state_rows, tasks, seed=args.seed)
    else:
        cases = []
        for task_id, (task_name, task_instruction, bddl_path) in sorted(tasks.items()):
            for episode_index in range(args.episodes_per_task):
                cases.append(
                    {
                        "episode_id": f"{args.suite}_task{task_id:02d}_episode{episode_index:04d}",
                        "suite": args.suite,
                        "task_id": task_id,
                        "task_name": task_name,
                        "task_instruction": task_instruction,
                        "bddl_path": bddl_path,
                        "source": f"{args.suite}_reset",
                        "state_row": None,
                        "episode_index": episode_index,
                        "environment_seed": int(args.seed + task_id * 100_003 + episode_index),
                        "sampling_seed": int(
                            args.seed + 50_000_000 + task_id * 100_003 + episode_index
                        ),
                    }
                )
    expected_count = len(cases)
    completed = 0
    errors = 0
    astra_called_count = 0
    recovery_executed_count = 0
    pi05_resumed_count = 0
    started = time.monotonic()
    for case in cases:
        task_id = int(case["task_id"])
        task_name = str(case["task_name"])
        task_instruction = str(case["task_instruction"])
        suite_name = str(case["suite"])
        bddl_path = Path(case["bddl_path"])
        state_row = case["state_row"]
        model_task_key = f"{args.suite}:{task_id}"
        head = heads.get(model_task_key)
        head_source = "task_head"
        if head is None and args.baseline == "learned":
            head = heads.get(fallback_head_key)
            head_source = "suite_fallback"
        if args.baseline == "learned" and head is None:
            raise SystemExit(f"no held-out AssistMLP checkpoint for task {model_task_key}")
        if args.baseline == "learned":
            task_threshold, hard_threshold = head.threshold(args.threshold_point)
        else:
            task_threshold, hard_threshold = None, None
        episode_id = str(case["episode_id"])
        episode_path = _episode_path(output_dir, episode_id)
        if episode_path.is_file():
            existing = json.loads(episode_path.read_text(encoding="utf-8"))
            if existing.get("status") == "complete":
                completed += 1
                continue
        environment_seed = int(case["environment_seed"])
        sampling_seed = int(case["sampling_seed"])
        env = None
        try:
            env, initial_observation, reset_retries = _make_env_with_reset(
                bddl_path,
                environment_seed,
                camera_size=args.camera_size,
                episode_horizon=args.episode_horizon,
                max_reset_retries=args.reset_retries,
            )
            if state_row is not None:
                initial_observation = restore_observation_from_state(env, state_row)
            policy._rng = jax.random.key(sampling_seed)
            if args.baseline == "learned":
                engine: DecisionEngine = CallDecisionEngine(
                    threshold=task_threshold,
                    hard_threshold=hard_threshold,
                    rule=args.decision_rule,
                    cooldown_queries=args.cooldown_queries,
                )
            elif args.baseline == "oracle":
                engine = OracleCompetenceDecisionEngine(low_competence_threshold=0.2)
            else:
                engine = BudgetDecisionEngine(
                    args.baseline,
                    budget=args.budget if args.baseline != "never_call" else None,
                )
            snapshot_root = _next_snapshot_root(output_dir, episode_id)
            handler = LoggingHandoffHandler(snapshot_root)
            astra_runtime_config = None
            if args.enable_astra:
                astra_runtime_config = {
                    "runtime_root": str(_next_astra_runtime_root(output_dir, episode_id)),
                    "snapshot_root": str(snapshot_root),
                    "model": args.astra_model,
                    "reasoning_effort": args.astra_reasoning_effort,
                    "max_total_steps": None,
                    "episode_step_limit": args.episode_horizon,
                    "max_decisions": args.astra_max_decisions,
                    "max_execution_chunks": args.astra_max_execution_chunks,
                    "max_tool_calls": args.astra_max_tool_calls,
                    "max_wall_seconds": args.astra_max_wall_seconds,
                    "chunk_edit_enabled": bool(args.astra_chunk_edit),
                    "chunk_mode_enabled": bool(args.astra_chunk),
                }
            initial_q_pi = (
                None
                if state_row is None or state_row.get("q_pi") is None
                else float(state_row["q_pi"])
            )
            if args.baseline == "oracle":
                score_help = lambda _policy_result, q=initial_q_pi: 1.0 - float(q)
                score_source = "oracle_one_minus_measured_q_pi"
            elif head is not None:
                score_help = head.predict
                score_source = (
                    "assist_head_decision"
                    if args.baseline == "learned"
                    else "assist_head_diagnostic_only"
                )
            else:
                score_help = lambda _policy_result: 0.0
                score_source = "baseline_without_assist_head"
            episode = run_closed_loop_episode(
                env,
                initial_observation,
                suite=suite_name,
                task_id=task_id,
                task_name=task_name,
                task_instruction=task_instruction,
                episode_id=episode_id,
                bddl_path=bddl_path,
                policy_infer=policy.infer_with_features,
                score_help=score_help,
                decision_engine=engine,
                handoff_handler=handler,
                max_steps=args.max_steps,
                episode_step_limit=args.episode_horizon,
                replan_steps=args.replan_steps,
                environment_seed=environment_seed + reset_retries,
                policy_sampling_seed=sampling_seed,
                reset_retries=reset_retries,
                source=str(case["source"]),
                initial_state_sample_id=(
                    str(state_row["sample_id"]) if state_row is not None else None
                ),
                initial_q_pi=initial_q_pi,
                score_source=score_source,
                audit_state_dir=output_dir / "audit_states",
                shadow_mode=args.shadow_mode,
                cooldown_queries=args.cooldown_queries,
                policy=policy,
                astra_runtime_config=astra_runtime_config,
                smoke_force_call_after_query=(1 if args.astra_smoke_force_call else None),
                video_path=output_dir / "videos" / f"{_safe_segment(episode_id)}.mp4",
                min_steps_between_calls=args.astra_min_call_steps,
            )
            env = None  # run_closed_loop_episode owns and closes the environment.
            episode.update(
                {
                    "method": args.baseline,
                    "model_variant": model_variant,
                    "threshold_point": args.threshold_point,
                    "held_out_task": model_task_key,
                    "head_source": head_source if args.baseline == "learned" else None,
                    "threshold_for_task": task_threshold if args.baseline == "learned" else None,
                    "hard_threshold_for_task": hard_threshold if args.baseline == "learned" else None,
                    "budget": args.budget,
                    "bddl_path": str(bddl_path),
                    "initial_sim_state_path": (
                        state_row.get("resolved_sim_state_path") if state_row is not None else None
                    ),
                    "recovery_model_called": bool(
                        episode.get("recovery_model_called", episode.get("astra_called", False))
                    ),
                    "astra_called": bool(episode.get("astra_called", False)),
                    "recovery_executed": bool(episode.get("recovery_executed", False)),
                }
            )
            astra_called_count += int(bool(episode.get("astra_called")))
            recovery_executed_count += int(bool(episode.get("recovery_executed")))
            pi05_resumed_count += int(bool(episode.get("pi05_resumed_after_recovery")))
            _atomic_json(episode_path, episode)
            completed += 1
        except Exception as exc:
            errors += 1
            error_root = output_dir / "episode_errors"
            error_id = f"{_safe_segment(episode_id)}_attempt_{errors:05d}.json"
            _atomic_json(
                error_root / error_id,
                {
                    "status": "error",
                    "episode_id": episode_id,
                    "suite": suite_name,
                    "task_id": task_id,
                    "task_name": task_name,
                    "environment_seed": environment_seed,
                    "policy_sampling_seed": sampling_seed,
                    "initial_state_sample_id": (
                        str(state_row["sample_id"]) if state_row is not None else None
                    ),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "recovery_model_called": False,
                    "astra_called": False,
                    "recovery_executed": False,
                },
            )
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
            print(f"episode error {episode_id}: {type(exc).__name__}: {exc}", flush=True)

    try:
        parquet_rows = _write_episode_parquet(output_dir)
    except (ImportError, ValueError):
        parquet_rows = 0
    episode_records = []
    for episode_file in sorted((output_dir / "episodes").glob("*.json")):
        try:
            episode_records.append(json.loads(episode_file.read_text(encoding="utf-8")))
        except Exception:
            continue
    total_env_steps = sum(int(item.get("num_env_steps", 0) or 0) for item in episode_records)
    total_pi05_steps = sum(int((item.get("astra_runtime") or {}).get("pi05_total_steps", item.get("pi05_phase_steps", 0)) or 0) for item in episode_records)
    total_recovery_model_steps = sum(int((item.get("astra_runtime") or {}).get(
        "recovery_model_total_steps",
        (item.get("astra_runtime") or {}).get("astra_total_steps", 0),
    ) or 0) for item in episode_records)
    total_pi05_inference_seconds = sum(float(item.get("pi05_inference_seconds", 0.0) or 0.0) for item in episode_records)
    total_astra_inference_seconds = sum(float(item.get("astra_inference_seconds", 0.0) or 0.0) for item in episode_records)
    call_steps = [item.get("call_step") for item in episode_records if item.get("call_step") is not None]
    resume_steps = [event.get("resume_step") for item in episode_records for event in item.get("recovery_events", []) if event.get("resume_step") is not None]
    video_files = [item.get("video_path") for item in episode_records if item.get("video_path")]
    total_usage = Counter()
    for item in episode_records:
        runtime = item.get("astra_runtime") or {}
        transport = runtime.get("provider_transport_summary") or {}
        usage = transport.get("usage") or {}
        if isinstance(usage, Mapping):
            for key, value in usage.items():
                if isinstance(value, (int, float)):
                    total_usage[key] += value
    run_report = {
        "status": "complete" if completed == expected_count and errors == 0 else "partial",
        "output_dir": str(output_dir),
        "expected_episodes": expected_count,
        "completed_episodes": completed,
        "episode_errors": errors,
        "episodes_parquet_rows": parquet_rows,
        "suite": args.suite,
        "episode_horizon": int(args.episode_horizon),
        "task_count": len(tasks),
        "task_head_count": len(expected_head_tasks & set(heads)),
        "suite_fallback_head_used": bool(
            args.baseline == "learned" and bool(missing_heads)
        ),
        "suite_fallback_task_count": len(missing_heads),
        "episodes_per_task": None if state_rows is not None else args.episodes_per_task,
        "state_count": len(state_rows) if state_rows is not None else 0,
        "method": args.baseline,
        "model_variant": model_variant,
        "threshold_point": args.threshold_point,
        "decision_rule": args.decision_rule,
        "shadow_mode": bool(args.shadow_mode),
        "astra_enabled": bool(args.enable_astra),
        "astra_model": args.astra_model if args.enable_astra else None,
        "astra_reasoning_effort": args.astra_reasoning_effort if args.enable_astra else None,
        "diagnostic_smoke_only": bool(args.astra_smoke_force_call),
        "checkpoint_provenance": official_pi05_libero_metadata(checkpoint),
        "reentry_module_enabled": True,
        "recovery_model_called": astra_called_count > 0,
        # Legacy report key retained for existing analysis scripts.
        "astra_called": astra_called_count > 0,
        "recovery_executed": recovery_executed_count > 0,
        "astra_called_episodes": astra_called_count,
        "recovery_executed_episodes": recovery_executed_count,
        "pi05_resumed_after_recovery_episodes": pi05_resumed_count,
        "elapsed_seconds": time.monotonic() - started,
        "metrics": {
            "total_env_steps": total_env_steps,
            "total_pi05_steps": total_pi05_steps,
            "total_recovery_model_steps": total_recovery_model_steps,
            "total_astra_steps": total_recovery_model_steps,
            "pi05_step_share": (total_pi05_steps / total_env_steps) if total_env_steps else None,
            "recovery_model_step_share": (
                total_recovery_model_steps / total_env_steps
            ) if total_env_steps else None,
            "astra_step_share": (
                total_recovery_model_steps / total_env_steps
            ) if total_env_steps else None,
            "call_steps": call_steps,
            "resume_steps": resume_steps,
            "total_pi05_inference_seconds": total_pi05_inference_seconds,
            "total_astra_inference_seconds": total_astra_inference_seconds,
            "pi05_inference_frequency_per_env_step": (len(episode_records) and sum(int(item.get("num_policy_queries", 0) or 0) for item in episode_records) / total_env_steps) if total_env_steps else None,
            "astra_token_usage": dict(total_usage),
            "astra_cost_estimate_usd": None,
            "astra_cost_note": "Gateway pricing was not exposed; exact token usage is recorded per episode.",
            "video_files": video_files,
            "video_count": len(video_files),
        },
    }
    _atomic_json(output_dir / "run_report.json", run_report)
    print(json.dumps(run_report, ensure_ascii=False, indent=2), flush=True)
    return 0 if run_report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
