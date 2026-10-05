"""Strict LIBERO Panda OSC_POSE adapter for dimensionful EEF commands.

The Astra-facing pose is the robot's measured grip control-site pose in the
world frame. LIBERO's public position and the live OSC controller both use that
site; its public quaternion is the EEF-body quaternion. The fixed body-to-site
transform is measured from the live robot model and used in both directions.
No object or task-state fields are read here.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import inspect
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import inverse_scale, matrix_to_quat_wxyz, quat_wxyz_to_matrix


class AdapterError(ValueError):
    """A command or live controller state cannot be safely adapted."""


@dataclass(frozen=True)
class EefPose:
    position_m: np.ndarray
    rotation_world: np.ndarray
    quaternion_wxyz: np.ndarray
    body_position_m: np.ndarray
    body_rotation_world: np.ndarray
    body_to_site_position_m: np.ndarray
    body_to_site_rotation: np.ndarray


@dataclass(frozen=True)
class MotionTarget:
    position_m: np.ndarray
    rotation_world: np.ndarray
    gripper: str


def _finite_vector(value: Any, shape: Tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise AdapterError(name + ":invalid_shape_or_number")
    return result.copy()


def _find_robot(env: Any) -> Any:
    current = env
    seen = set()
    for _ in range(12):
        if current is None or id(current) in seen:
            break
        seen.add(id(current))
        robots = getattr(current, "robots", None)
        if robots is not None:
            if len(robots) != 1:
                raise AdapterError("expected_one_robot")
            return robots[0]
        current = getattr(current, "env", None)
    raise AdapterError("robot_not_found_in_environment_wrappers")


def _read_action_spec(env: Any) -> Tuple[np.ndarray, np.ndarray, Any]:
    current = env
    seen = set()
    for _ in range(12):
        if current is None or id(current) in seen:
            break
        seen.add(id(current))
        try:
            spec = getattr(current, "action_spec")
        except Exception:
            spec = None
        if spec is not None:
            try:
                low, high = spec
            except Exception as error:
                raise AdapterError("libero_action_spec_invalid") from error
            low = np.asarray(low, dtype=np.float64)
            high = np.asarray(high, dtype=np.float64)
            if low.shape != (7,) or high.shape != (7,):
                raise AdapterError("libero_action_spec_must_be_7d")
            if not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(low >= high):
                raise AdapterError("libero_action_spec_invalid")
            return low, high, current
        current = getattr(current, "env", None)
    raise AdapterError("libero_action_spec_unavailable")


class LiberoEefAdapter:
    """Convert safe world-frame EEF targets to native 7D OSC_POSE actions."""

    def __init__(
        self,
        env: Any,
        *,
        max_target_position_m: float = 0.05,
        max_target_rotation_rad: float = 0.35,
        max_step_position_m: float = 0.01,
        max_step_rotation_rad: float = 0.05,
        position_tolerance_m: float = 0.002,
        rotation_tolerance_rad: float = 0.02,
        observation_position_tolerance_m: float = 2e-4,
    ) -> None:
        self.env = env
        self.robot = _find_robot(env)
        self.controller = getattr(self.robot, "controller", None)
        if self.controller is None:
            raise AdapterError("robot_controller_missing")
        self.sim = getattr(self.robot, "sim", None) or getattr(self.controller, "sim", None)
        if self.sim is None:
            raise AdapterError("robot_sim_missing")
        self.action_low, self.action_high, action_spec_owner = _read_action_spec(env)
        self.action_spec_source = type(action_spec_owner).__name__
        action_dim = getattr(action_spec_owner, "action_dim", getattr(env, "action_dim", 7))
        self.action_dim = int(action_dim)
        if self.action_dim != 7:
            raise AdapterError("libero_action_dim_must_be_7")

        if "OperationalSpaceController" not in type(self.controller).__name__:
            raise AdapterError("unsupported_controller_type")
        if not bool(getattr(self.controller, "use_delta", False)):
            raise AdapterError("OSC_POSE_must_use_delta_control")
        if not bool(getattr(self.controller, "use_ori", False)):
            raise AdapterError("OSC_POSE_must_control_orientation")
        if getattr(self.controller, "impedance_mode", "fixed") != "fixed":
            raise AdapterError("variable_impedance_not_supported")
        if int(getattr(self.controller, "control_dim", 0)) != 6:
            raise AdapterError("OSC_POSE_control_dim_must_be_6")

        self.input_min = _finite_vector(getattr(self.controller, "input_min", None), (6,), "input_min")
        self.input_max = _finite_vector(getattr(self.controller, "input_max", None), (6,), "input_max")
        self.output_min = _finite_vector(getattr(self.controller, "output_min", None), (6,), "output_min")
        self.output_max = _finite_vector(getattr(self.controller, "output_max", None), (6,), "output_max")
        if np.any(self.input_min >= self.input_max) or np.any(self.output_min >= self.output_max):
            raise AdapterError("OSC_POSE_scale_ranges_invalid")
        if np.any(self.action_low[:6] > self.input_min + 1e-8) or np.any(
            self.action_high[:6] < self.input_max - 1e-8
        ):
            raise AdapterError("LIBERO_action_spec_does_not_cover_OSC_POSE_input")
        if self.action_low[6] > -1.0 + 1e-8 or self.action_high[6] < 1.0 - 1e-8:
            raise AdapterError("Panda_gripper_action_range_must_cover_minus1_plus1")

        gripper = getattr(self.robot, "gripper", None)
        self.gripper_class = type(gripper).__name__
        if self.gripper_class != "PandaGripper":
            raise AdapterError("expected_PandaGripper")
        self.control_freq = float(
            getattr(self.controller, "control_freq", getattr(env, "control_freq", 0.0))
        )
        if not math.isfinite(self.control_freq) or self.control_freq <= 0:
            raise AdapterError("control_frequency_unavailable")

        self.max_target_position_m = float(max_target_position_m)
        self.max_target_rotation_rad = float(max_target_rotation_rad)
        self.max_step_position_m = float(max_step_position_m)
        self.max_step_rotation_rad = float(max_step_rotation_rad)
        self.position_tolerance_m = float(position_tolerance_m)
        self.rotation_tolerance_rad = float(rotation_tolerance_rad)
        self.observation_position_tolerance_m = float(observation_position_tolerance_m)
        configured = (self.max_target_position_m, self.max_target_rotation_rad,
                      self.max_step_position_m, self.max_step_rotation_rad,
                      self.position_tolerance_m, self.rotation_tolerance_rad,
                      self.observation_position_tolerance_m)
        if not all(math.isfinite(x) and x > 0 for x in configured):
            raise AdapterError("adapter_limits_invalid")
        self._body_to_site = None
        self._body_name = str(self.robot.robot_model.eef_name)
        self._site_id = int(self.robot.eef_site_id)
        self._site_rotation_delta_convention = "world_left_multiply"
        self._rotation_delta_source_verified = self._verify_rotation_delta_source()
        orientation_limits = getattr(self.controller, "orientation_limits", None)
        if orientation_limits is not None and np.asarray(orientation_limits).size and np.asarray(orientation_limits).any():
            raise AdapterError("OSC_orientation_limits_not_supported_without_nonclipping_adapter")
        position_limits = getattr(self.controller, "position_limits", None)
        if position_limits is not None:
            limits = np.asarray(position_limits, dtype=np.float64)
            if limits.shape != (2, 3) or not np.isfinite(limits).all() or np.any(limits[0] >= limits[1]):
                raise AdapterError("OSC_position_limits_invalid")

    def _pose_from_runtime(self, raw: Mapping[str, Any]) -> EefPose:
        required = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")
        if any(key not in raw for key in required):
            raise AdapterError("public_robot_observation_missing_fields")
        obs_site_pos = _finite_vector(raw["robot0_eef_pos"], (3,), "observation_eef_position")
        obs_body_quat_xyzw = _finite_vector(raw["robot0_eef_quat"], (4,), "observation_eef_quaternion")
        obs_body_rot = quat_wxyz_to_matrix(obs_body_quat_xyzw[[3, 0, 1, 2]])
        gripper_qpos = _finite_vector(raw["robot0_gripper_qpos"], (2,), "gripper_qpos")

        # LIBERO returns the robot0_eef_* sensors from the same simulator
        # observation checkpoint. On the first frame, audit those public
        # sensors against the live robot/controller and calibrate the fixed
        # body-to-control-site transform. After a physics step, do not call
        # controller.update(force=True): Robosuite's update calls sim.forward,
        # which recomputes derived site positions at a newer timestamp than
        # the observation just returned by env.step. Continue from the exact
        # public checkpoint and the audited fixed transform instead.
        if self._body_to_site is None:
            try:
                self.controller.update(force=True)
                live_body_pos = np.asarray(
                    self.sim.data.get_body_xpos(self._body_name), dtype=np.float64
                ).reshape(3)
                live_body_quat_wxyz = np.asarray(
                    self.sim.data.get_body_xquat(self._body_name), dtype=np.float64
                ).reshape(4)
                live_site_pos = np.asarray(
                    self.sim.data.site_xpos[self._site_id], dtype=np.float64
                ).reshape(3)
                live_site_rot = np.asarray(
                    self.sim.data.site_xmat[self._site_id], dtype=np.float64
                ).reshape(3, 3)
            except Exception as error:
                raise AdapterError("robot_body_or_control_site_pose_unavailable") from error
            live_body_rot = quat_wxyz_to_matrix(live_body_quat_wxyz)
            if not np.isfinite(live_body_pos).all() or not np.isfinite(live_site_pos).all() or not np.isfinite(live_site_rot).all():
                raise AdapterError("robot_pose_nonfinite")
            if np.linalg.norm(live_site_pos - obs_site_pos) > self.observation_position_tolerance_m:
                error_mm = ",".join(
                    f"{value * 1000.0:.3f}" for value in live_site_pos - obs_site_pos
                )
                raise AdapterError(
                    "EEF_site_position_does_not_match_LIBERO_observation:delta_mm=" + error_mm
                )
            if not np.allclose(live_body_rot, obs_body_rot, atol=2e-4, rtol=0):
                raise AdapterError("EEF_body_quaternion_order_or_sync_mismatch")
            controller_pos = np.asarray(self.controller.ee_pos, dtype=np.float64).reshape(3)
            controller_rot = np.asarray(self.controller.ee_ori_mat, dtype=np.float64).reshape(3, 3)
            if not np.allclose(controller_pos, live_site_pos, atol=2e-4, rtol=0) or not np.allclose(
                controller_rot, live_site_rot, atol=2e-4, rtol=0
            ):
                raise AdapterError("OSC_controller_pose_does_not_match_live_EEF_control_site")

            body_to_site_rot = live_body_rot.T @ live_site_rot
            body_to_site_pos = live_body_rot.T @ (live_site_pos - live_body_pos)
            self._body_to_site = (body_to_site_pos.copy(), body_to_site_rot.copy())
            body_pos = live_body_pos
            body_rot = live_body_rot
            site_rot = live_site_rot
        else:
            body_to_site_pos, body_to_site_rot = self._body_to_site
            body_rot = obs_body_rot
            site_rot = body_rot @ body_to_site_rot
            body_pos = obs_site_pos - body_rot @ body_to_site_pos

        site_pos = obs_site_pos
        if not np.isfinite(body_pos).all() or not np.isfinite(site_pos).all() or not np.isfinite(site_rot).all():
            raise AdapterError("robot_pose_nonfinite")
        return EefPose(
            position_m=site_pos.copy(),
            rotation_world=site_rot.copy(),
            quaternion_wxyz=matrix_to_quat_wxyz(site_rot),
            body_position_m=body_pos.copy(),
            body_rotation_world=body_rot.copy(),
            body_to_site_position_m=body_to_site_pos.copy(),
            body_to_site_rotation=body_to_site_rot.copy(),
        )

    def read_pose(self, raw: Mapping[str, Any]) -> EefPose:
        return self._pose_from_runtime(raw)

    def begin_intervention(self, raw: Mapping[str, Any]) -> Dict[str, Any]:
        self._pose_from_runtime(raw)
        return {"active": False, "workspace_guard": "disabled"}

    def end_intervention(self) -> None:
        return None

    def runtime_audit(self, raw: Mapping[str, Any]) -> Dict[str, Any]:
        pose = self.read_pose(raw)
        return {
            "robosuite_version": self._robosuite_version(),
            "controller_class": type(self.controller).__name__,
            "controller_control_dim": int(self.controller.control_dim),
            "controller_use_delta": bool(self.controller.use_delta),
            "controller_use_ori": bool(self.controller.use_ori),
            "controller_impedance_mode": str(self.controller.impedance_mode),
            "controller_pose_reference": "world_control_site",
            "controller_input_min": self.input_min.tolist(),
            "controller_input_max": self.input_max.tolist(),
            "controller_output_min": self.output_min.tolist(),
            "controller_output_max": self.output_max.tolist(),
            "controller_position_limits": _json_value(getattr(self.controller, "position_limits", None)),
            "controller_orientation_limits": _json_value(getattr(self.controller, "orientation_limits", None)),
            "env_action_dim": self.action_dim,
            "env_action_spec_source": self.action_spec_source,
            "env_action_low": self.action_low.tolist(),
            "env_action_high": self.action_high.tolist(),
            "control_freq_hz": self.control_freq,
            "robot_model": type(self.robot.robot_model).__name__,
            "gripper_class": self.gripper_class,
            "observation_eef_position_frame": "world_control_site",
            "observation_position_tolerance_m": self.observation_position_tolerance_m,
            "observation_pose_policy": "calibrate_against_live_robot_on_initial_frame_then_use_synchronized_public_checkpoints",
            "observation_eef_quaternion_frame": "world_eef_body_xyzw",
            "astra_eef_pose_frame": "world_control_site",
            "astra_quaternion_order": "wxyz",
            "rotation_delta_convention": self._site_rotation_delta_convention,
            "rotation_delta_source_verified": bool(self._rotation_delta_source_verified),
            "body_to_site_translation_m": pose.body_to_site_position_m.tolist(),
            "body_to_site_rotation": pose.body_to_site_rotation.tolist(),
            "gripper_action_semantics": {"-1": "open", "0": "keep", "+1": "closed"},
            "limits": {
                "max_target_position_norm_m": self.max_target_position_m,
                "max_target_rotation_norm_rad": self.max_target_rotation_rad,
                "max_control_step_position_norm_m": self.max_step_position_m,
                "max_control_step_rotation_norm_rad": self.max_step_rotation_rad,
            },
            "workspace_guard": "disabled",
            "absolute_geometry_workspace_audited": False,
        }

    @staticmethod
    def _robosuite_version() -> str:
        try:
            import robosuite
            return str(getattr(robosuite, "__version__", "unknown"))
        except Exception:
            return "unavailable"

    def _verify_rotation_delta_source(self) -> bool:
        try:
            from robosuite.utils import control_utils
            source = inspect.getsource(control_utils.set_goal_orientation)
        except Exception as error:
            raise AdapterError("OSC_rotation_frame_semantics_not_auditable") from error
        compact = "".join(source.split())
        if "np.dot(rotation_mat_error,current_orientation)" not in compact:
            raise AdapterError("OSC_rotation_delta_is_not_verified_world_left_multiply")
        return True

    def action_validator(self, action: np.ndarray) -> None:
        """Strict validator for actions generated by the Astra adapter."""
        action = _finite_vector(action, (7,), "native_action")
        if np.any(action < self.action_low - 1e-8) or np.any(action > self.action_high + 1e-8):
            raise AdapterError("native_action_out_of_LIBERO_action_spec")
        if np.any(action[:6] < self.input_min - 1e-8) or np.any(action[:6] > self.input_max + 1e-8):
            raise AdapterError("native_action_out_of_OSC_POSE_input_range")
        if action[6] not in (-1.0, 0.0, 1.0):
            raise AdapterError("invalid_Panda_gripper_command")

    @staticmethod
    def pi05_action_validator(action: np.ndarray) -> None:
        """Validate structure only; preserve π0.5's native action semantics.

        LIBERO's OSC controller clips the first six inputs to its configured
        input range, and PandaGripper maps the final continuous input by sign.
        Do not clip or reject finite π0.5 outputs here: pass them unchanged to
        the environment, which applies those native transformations.
        """
        _finite_vector(action, (7,), "pi05_native_action")

    def validate_pi05_chunk(self, actions: Any) -> np.ndarray:
        result = np.asarray(actions, dtype=np.float64)
        if result.ndim != 2 or result.shape[1] != 7 or result.shape[0] < 1:
            raise AdapterError("pi05_action_chunk_must_be_N_by_7")
        if not np.isfinite(result).all():
            raise AdapterError("pi05_action_chunk_nonfinite")
        return result.copy()

    def native_action_to_site_trajectory(self, actions: Any, pose: EefPose) -> list[dict[str, Any]]:
        # This is a prediction of what the existing LIBERO controller will
        # receive, not the action chunk itself. Robosuite's OSC scale_action
        # clips the six pose inputs before scaling; physical π0.5 actions
        # remain untouched and are still passed to env.step verbatim.
        rows = self.validate_pi05_chunk(actions)
        effective_rows = rows.copy()
        effective_rows[:, :6] = np.clip(
            effective_rows[:, :6], self.input_min[None, :], self.input_max[None, :]
        )
        body_pos = pose.body_position_m.copy()
        body_rot = pose.body_rotation_world.copy()
        b2s_pos = pose.body_to_site_position_m
        b2s_rot = pose.body_to_site_rotation
        site_pos = pose.position_m.copy()
        site_rot = pose.rotation_world.copy()
        trajectory = []
        for index, row in enumerate(effective_rows):
            delta = self._scale_input_to_output(row[:6])
            body_pos = body_pos + delta[:3]
            body_rot = Rotation.from_rotvec(delta[3:]).as_matrix() @ body_rot
            site_pos = body_pos + body_rot @ b2s_pos
            site_rot = body_rot @ b2s_rot
            gripper = "open" if row[6] < -1e-6 else "closed" if row[6] > 1e-6 else "keep"
            trajectory.append({
                "proposal_step": index,
                "predicted_control_site_position_m": site_pos.tolist(),
                "predicted_control_site_quaternion_wxyz": matrix_to_quat_wxyz(site_rot).tolist(),
                "nominal_delta_position_m": delta[:3].tolist(),
                "nominal_delta_rotation_vector_rad": delta[3:].tolist(),
                "gripper": gripper,
            })
        return trajectory

    def resolve_target(self, decision: Mapping[str, Any], current: EefPose) -> MotionTarget:
        mode = decision["mode"]
        if mode == "eef":
            target = decision["target"]
            position = _finite_vector(target["position"], (3,), "target_position")
            rotation = quat_wxyz_to_matrix(target["quaternion_wxyz"])
            gripper = "closed" if target["gripper_closed"] else "open"
        elif mode == "eef_delta":
            delta = decision["delta"]
            position = current.position_m + _finite_vector(delta["delta_position"], (3,), "delta_position")
            dr = _finite_vector(delta["delta_rotation_vector"], (3,), "delta_rotation_vector")
            rotation = Rotation.from_rotvec(dr).as_matrix() @ current.rotation_world
            gripper = str(delta["gripper"])
        else:
            raise AdapterError("non_motion_decision_passed_to_EEF_adapter")
        position_error = position - current.position_m
        rotation_error = Rotation.from_matrix(rotation @ current.rotation_world.T).as_rotvec()
        if float(np.linalg.norm(position_error)) > self.max_target_position_m + 1e-9:
            raise AdapterError("target_exceeds_total_position_safety_bound")
        if float(np.linalg.norm(rotation_error)) > self.max_target_rotation_rad + 1e-9:
            raise AdapterError("target_exceeds_total_rotation_safety_bound")
        return MotionTarget(position.copy(), rotation.copy(), gripper)

    def minimum_required_steps(self, target: MotionTarget, current: EefPose) -> int:
        pose = current
        for count in range(1, 7):
            pos_error, rot_error = self.target_errors(target, pose)
            if pos_error <= self.position_tolerance_m and rot_error <= self.rotation_tolerance_rad:
                return max(1, count - 1)
            action = self.action_toward_target(target, pose)
            physical = self._scale_input_to_output(action[:6])
            body_position = pose.body_position_m + physical[:3]
            body_rotation = Rotation.from_rotvec(physical[3:]).as_matrix() @ pose.body_rotation_world
            site_position = body_position + body_rotation @ pose.body_to_site_position_m
            site_rotation = body_rotation @ pose.body_to_site_rotation
            pose = EefPose(
                position_m=site_position,
                rotation_world=site_rotation,
                quaternion_wxyz=matrix_to_quat_wxyz(site_rotation),
                body_position_m=body_position,
                body_rotation_world=body_rotation,
                body_to_site_position_m=pose.body_to_site_position_m,
                body_to_site_rotation=pose.body_to_site_rotation,
            )
        pos_error, rot_error = self.target_errors(target, pose)
        if pos_error <= self.position_tolerance_m and rot_error <= self.rotation_tolerance_rad:
            return 6
        return 7

    def _scale_input_to_output(self, normalized: np.ndarray) -> np.ndarray:
        value = _finite_vector(normalized, (6,), "normalized_osc_action")
        if np.any(value < self.input_min - 1e-8) or np.any(value > self.input_max + 1e-8):
            raise AdapterError("normalized_osc_action_out_of_range")
        scale = (self.output_max - self.output_min) / (self.input_max - self.input_min)
        return (value - (self.input_max + self.input_min) / 2.0) * scale + (self.output_max + self.output_min) / 2.0

    def _site_subtarget_to_body(self, site_position: np.ndarray, site_rotation: np.ndarray,
                                current: EefPose) -> Tuple[np.ndarray, np.ndarray]:
        b2s_pos = current.body_to_site_position_m
        b2s_rot = current.body_to_site_rotation
        body_rotation = site_rotation @ b2s_rot.T
        body_position = site_position - body_rotation @ b2s_pos
        return body_position, body_rotation

    def action_toward_target(self, target: MotionTarget, current: EefPose) -> np.ndarray:
        site_dp = target.position_m - current.position_m
        site_dr = Rotation.from_matrix(target.rotation_world @ current.rotation_world.T).as_rotvec()
        alpha = 1.0
        if np.linalg.norm(site_dp) > self.max_step_position_m:
            alpha = min(alpha, self.max_step_position_m / float(np.linalg.norm(site_dp)))
        if np.linalg.norm(site_dr) > self.max_step_rotation_rad:
            alpha = min(alpha, self.max_step_rotation_rad / float(np.linalg.norm(site_dr)))

        # Find the largest feedback increment that remains inside both the
        # physical OSC output range and the conservative per-step norm limits.
        selected = None
        for _ in range(48):
            sub_pos = current.position_m + alpha * site_dp
            sub_rot = Rotation.from_rotvec(alpha * site_dr).as_matrix() @ current.rotation_world
            desired_body_pos, desired_body_rot = self._site_subtarget_to_body(sub_pos, sub_rot, current)
            body_dp = desired_body_pos - current.body_position_m
            body_dr = Rotation.from_matrix(desired_body_rot @ current.body_rotation_world.T).as_rotvec()
            position_limits = getattr(self.controller, "position_limits", None)
            within_position_limits = True
            if position_limits is not None:
                limits = np.asarray(position_limits, dtype=np.float64)
                within_position_limits = bool(
                    np.all(desired_body_pos >= limits[0] - 1e-9)
                    and np.all(desired_body_pos <= limits[1] + 1e-9)
                )
            within_step = (
                within_position_limits
                and
                np.linalg.norm(body_dp) <= self.max_step_position_m + 1e-9
                and np.linalg.norm(body_dr) <= self.max_step_rotation_rad + 1e-9
                and np.all(body_dp >= self.output_min[:3] - 1e-9)
                and np.all(body_dp <= self.output_max[:3] + 1e-9)
                and np.all(body_dr >= self.output_min[3:] - 1e-9)
                and np.all(body_dr <= self.output_max[3:] + 1e-9)
            )
            if within_step:
                try:
                    normalized = inverse_scale(
                        np.concatenate((body_dp, body_dr)), self.input_min, self.input_max,
                        self.output_min, self.output_max,
                    )
                    selected = normalized
                    break
                except ValueError:
                    pass
            alpha *= 0.5
        if selected is None or alpha < 1e-7:
            raise AdapterError("no_safe_native_action_represents_target_increment")
        if target.gripper == "keep":
            gripper_action = 0.0
        elif target.gripper == "open":
            gripper_action = -1.0
        elif target.gripper == "closed":
            gripper_action = 1.0
        else:
            raise AdapterError("invalid_gripper_semantic")
        action = np.concatenate((selected, np.asarray([gripper_action], dtype=np.float64)))
        self.action_validator(action)
        return action

    def target_errors(self, target: MotionTarget, current: EefPose) -> Tuple[float, float]:
        pos = float(np.linalg.norm(target.position_m - current.position_m))
        rot = float(np.linalg.norm(Rotation.from_matrix(target.rotation_world @ current.rotation_world.T).as_rotvec()))
        return pos, rot


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        return np.asarray(value).tolist()
    except Exception:
        return str(value)
