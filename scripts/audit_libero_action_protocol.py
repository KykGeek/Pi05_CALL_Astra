#!/usr/bin/env python3
"""Audit the production EEF adapter against an externally installed LIBERO env.

Without ``--motion`` this checks the live controller/action protocol only.
With ``--motion`` it also executes small, signed EEF movements in a fresh
simulation episode and verifies their observed direction. It never reads
object-state fields.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.astra.adapter import LiberoEefAdapter  # noqa: E402


def _public_delta(before, after):
    translation = after.position_m - before.position_m
    rotation = Rotation.from_matrix(
        after.rotation_world @ before.rotation_world.T
    ).as_rotvec()
    return translation, rotation


def _run_motion_probe(env, adapter, observation):
    rows = []
    for axis in range(6):
        for sign in (1.0, -1.0):
            before = adapter.read_pose(observation)
            delta_position = np.zeros(3, dtype=np.float64)
            delta_rotation = np.zeros(3, dtype=np.float64)
            if axis < 3:
                delta_position[axis] = sign * 0.003
            else:
                delta_rotation[axis - 3] = sign * 0.02
            target = adapter.resolve_target(
                {
                    "mode": "eef_delta",
                    "delta": {
                        "delta_position": delta_position.tolist(),
                        "delta_rotation_vector": delta_rotation.tolist(),
                        "gripper": "keep",
                    },
                },
                before,
            )
            current = before
            steps = 0
            for _ in range(8):
                action = adapter.action_toward_target(target, current)
                observation, _reward, done, _info = env.step(action.tolist())
                steps += 1
                if done:
                    raise RuntimeError("diagnostic_environment_terminated")
                current = adapter.read_pose(observation)
                pos_error, rot_error = adapter.target_errors(target, current)
                if (pos_error <= adapter.position_tolerance_m and
                        rot_error <= adapter.rotation_tolerance_rad):
                    break
            measured_position, measured_rotation = _public_delta(before, current)
            measured_axis = measured_position[axis] if axis < 3 else measured_rotation[axis - 3]
            direction_ok = bool(sign * float(measured_axis) > 0.0)
            rows.append(
                {
                    "axis": axis,
                    "sign": int(sign),
                    "steps": steps,
                    "measured_translation_m": measured_position.tolist(),
                    "measured_rotation_vector_rad": measured_rotation.tolist(),
                    "direction_verified": direction_ok,
                }
            )
    return observation, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--openpi-root", type=Path,
                        default=Path(os.environ["OPENPI_ROOT"]).expanduser()
                        if os.environ.get("OPENPI_ROOT") else None)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--motion", action="store_true",
                        help="execute a small signed-motion probe in the simulator")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Diagnostic output already exists")
    if args.openpi_root is None:
        parser.error("set --openpi-root or OPENPI_ROOT to the external OpenPI checkout")
    if args.gpu_index < 0:
        parser.error("--gpu-index must be nonnegative")

    openpi_root = args.openpi_root.expanduser().resolve()
    libero_root = openpi_root / "third_party" / "libero"
    if not openpi_root.is_dir() or not libero_root.is_dir():
        parser.error("OpenPI checkout or its third_party/libero directory was not found")
    sys.path.insert(0, str(openpi_root))
    sys.path.insert(0, str(libero_root))
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark_dict
    from libero.libero.envs import OffScreenRenderEnv

    suite = get_benchmark_dict()["libero_10"]()
    task = suite.get_task(args.task_id)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl), camera_heights=128, camera_widths=128,
        render_gpu_device_id=args.gpu_index,
    )
    try:
        env.seed(0)
        observation = env.reset()
        adapter = LiberoEefAdapter(env)
        report = adapter.runtime_audit(observation)
        report["task_id"] = int(args.task_id)
        report["motion_requested"] = bool(args.motion)
        report["motion_steps"] = []
        report["motion_verified"] = None
        if args.motion:
            observation, rows = _run_motion_probe(env, adapter, observation)
            report["motion_steps"] = rows
            report["motion_verified"] = bool(rows) and all(
                row["direction_verified"] for row in rows
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({
            "output": str(args.output),
            "motion_verified": report["motion_verified"],
            "controller": report["controller_class"],
            "action_dim": report["env_action_dim"],
        }))
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
