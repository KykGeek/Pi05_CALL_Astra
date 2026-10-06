"""Public robot-only EEF history; simulator and object state never enter it."""
from __future__ import annotations

import json
import os
from pathlib import Path
import threading
from typing import Any, Dict, List, Mapping

import numpy as np
from scipy.spatial.transform import Rotation


class PublicEefHistory:
    """Persist measured robot transitions and expose a bounded recent window."""

    def __init__(self, path: str, adapter: Any, *, max_records: int = 256,
                 model_window: int = 20) -> None:
        if max_records < 1 or model_window < 1 or model_window > max_records:
            raise ValueError("invalid_EEF_history_limits")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.adapter = adapter
        self.max_records = int(max_records)
        self.model_window = int(model_window)
        self._lock = threading.RLock()
        self._records: List[Dict[str, Any]] = []
        self._poses: Dict[str, Any] = {}
        self._stream = self.path.open("x", encoding="utf-8")
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            self._stream.close()
            raise

    def register_initial(self, checkpoint: Any) -> None:
        with self._lock:
            pose = self.adapter.read_pose(checkpoint.raw)
            self._poses[str(checkpoint.observation_id)] = pose
            gripper = _gripper(checkpoint.raw)
            record = {
                "env_step": int(checkpoint.env_step),
                "observation_id": str(checkpoint.observation_id),
                "source": "initial_observation",
                "current_eef": _pose_json(pose),
                "gripper_qpos": gripper,
            }
            self._append(record)

    def record_transition(self, before: Any, action: np.ndarray, after: Any,
                          source: str, decision_id: str | None) -> None:
        with self._lock:
            before_id = str(before.observation_id)
            after_id = str(after.observation_id)
            old_pose = self._poses.get(before_id)
            if old_pose is None:
                raise RuntimeError("missing_public_pose_for_previous_checkpoint")
            new_pose = self.adapter.read_pose(after.raw)
            dp = new_pose.position_m - old_pose.position_m
            dr = Rotation.from_matrix(new_pose.rotation_world @ old_pose.rotation_world.T).as_rotvec()
            record = {
                "env_step": int(after.env_step),
                "observation_id": after_id,
                "previous_observation_id": before_id,
                "source": str(source),
                "decision_id": decision_id,
                "current_eef": _pose_json(new_pose),
                "measured_delta_position_m": dp.tolist(),
                "measured_delta_rotation_vector_rad": dr.tolist(),
                "gripper_qpos": _gripper(after.raw),
            }
            self._append(record)
            self._poses[after_id] = new_pose
            # Keep only an ID window needed by the recent history. The broker
            # retains the authoritative checkpoint registry separately.
            keep = {str(row["observation_id"]) for row in self._records[-self.model_window - 1:]}
            for key in list(self._poses):
                if key not in keep:
                    del self._poses[key]

    def recent(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(row) for row in self._records[-self.model_window:]]

    def pose_for(self, observation_id: str) -> Any:
        with self._lock:
            return self._poses.get(str(observation_id))

    def _append(self, row: Mapping[str, Any]) -> None:
        value = dict(row)
        line = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        self._stream.write(line + "\n")
        self._stream.flush()
        os.fsync(self._stream.fileno())
        self._records.append(value)
        if len(self._records) > self.max_records:
            del self._records[:-self.max_records]

    def close(self) -> None:
        with self._lock:
            if not self._stream.closed:
                self._stream.close()


def _gripper(raw: Mapping[str, Any]) -> List[float]:
    value = np.asarray(raw.get("robot0_gripper_qpos"), dtype=np.float64)
    if value.shape != (2,) or not np.isfinite(value).all():
        raise ValueError("invalid_public_gripper_state")
    return value.tolist()


def _pose_json(pose: Any) -> Dict[str, Any]:
    return {
        "frame": "world",
        "reference": "control_site",
        "position_m": np.asarray(pose.position_m, dtype=np.float64).tolist(),
        "quaternion_wxyz": np.asarray(pose.quaternion_wxyz, dtype=np.float64).tolist(),
    }
