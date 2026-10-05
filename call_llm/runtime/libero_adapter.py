"""Audited robosuite 1.4.0 Panda world-frame control-site adapter."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import inspect
from pathlib import Path
import numpy as np

from .geometry import (vec, matrix_to_quat_wxyz, body_xyzw_to_site_wxyz,
                       resolve_delta, pose_error, bounded_increment, inverse_scale)


@dataclass(frozen=True)
class ControlPose:
    position: np.ndarray
    quaternion_wxyz: np.ndarray


@dataclass(frozen=True)
class Limits:
    workspace_min: np.ndarray
    workspace_max: np.ndarray
    max_translation_per_step: float = .01
    max_rotation_per_step: float = .05
    max_target_translation: float = .05
    max_target_rotation: float = .35
    position_tolerance: float = .002
    rotation_tolerance: float = .02

    def __post_init__(self):
        lo, hi = vec(self.workspace_min,3,'workspace_min'), vec(self.workspace_max,3,'workspace_max')
        if np.any(lo >= hi): raise ValueError('invalid_workspace')
        object.__setattr__(self,'workspace_min',lo)
        object.__setattr__(self,'workspace_max',hi)
        for key in ('max_translation_per_step','max_rotation_per_step','max_target_translation',
                    'max_target_rotation','position_tolerance','rotation_tolerance'):
            value=getattr(self,key)
            if not np.isfinite(value) or value<=0: raise ValueError('invalid_'+key)


@dataclass(frozen=True)
class FixedTarget:
    pose: ControlPose
    origin: ControlPose
    gripper_intent: str
    decision_id: str


def six(value, name):
    a=np.asarray(value,dtype=np.float64)
    if a.shape==(): a=np.repeat(a,6)
    return vec(a,6,name)


def audit_runtime(env, observation):
    import robosuite
    from robosuite.controllers.osc import OperationalSpaceController
    from robosuite.utils.control_utils import set_goal_orientation
    from robosuite.models.grippers.panda_gripper import PandaGripper
    if robosuite.__version__ != '1.4.0': raise ValueError('unsupported_robosuite_version')
    # This exact wrapper chain is verified against installed ControlEnv.
    core=env.env
    if len(core.robots)!=1: raise ValueError('single_robot_required')
    robot=core.robots[0]; c=robot.controller
    if not isinstance(c,OperationalSpaceController) or not isinstance(robot.gripper,PandaGripper):
        raise ValueError('unsupported_controller_or_gripper')
    if not c.use_delta or not c.use_ori or c.impedance_mode!='fixed':
        raise ValueError('unsupported_osc_mode')
    if c.position_limits is not None or c.orientation_limits is not None:
        raise ValueError('controller_internal_target_limits_not_supported')
    if c.interpolator_pos is not None or c.interpolator_ori is not None:
        raise ValueError('controller_interpolation_not_audited')
    low,high=(vec(x,7,'action_spec') for x in core.action_spec)
    if np.any(low>=high): raise ValueError('invalid_action_spec')
    imin,imax,omin,omax=(six(getattr(c,k),k) for k in ('input_min','input_max','output_min','output_max'))
    if np.any(omin>=0) or np.any(omax<=0): raise ValueError('zero_not_strictly_inside_output_range')
    native=inverse_scale(np.zeros(6),imin,imax,omin,omax)
    if not np.allclose(c.scale_action(native),0,atol=1e-12): raise ValueError('scale_roundtrip_failed')
    sid=robot.eef_site_id
    if isinstance(sid,dict): raise ValueError('single_control_site_required')
    p=np.asarray(core.sim.data.site_xpos[sid]).copy()
    rs=np.asarray(core.sim.data.site_xmat[sid]).reshape(3,3).copy()
    rb=np.asarray(core.sim.data.get_body_xmat(robot.robot_model.eef_name)).reshape(3,3).copy()
    fixed=rb.T@rs
    if not np.allclose(p,observation['robot0_eef_pos'],atol=1e-6,rtol=0):
        raise ValueError('observed_position_not_control_site')
    q=body_xyzw_to_site_wxyz(observation['robot0_eef_quat'],fixed)
    from .geometry import quat_wxyz_to_matrix
    if not np.allclose(quat_wxyz_to_matrix(q),rs,atol=1e-6,rtol=0):
        raise ValueError('observed_orientation_mapping_mismatch')
    # Verify installed rotation function against a noncommuting synthetic case.
    from scipy.spatial.transform import Rotation
    r0=Rotation.from_rotvec([.3,.1,0.]).as_matrix(); dr=np.array([0.,0.,.2])
    if not np.allclose(set_goal_orientation(dr,r0),Rotation.from_rotvec(dr).as_matrix()@r0,atol=1e-7):
        raise ValueError('world_rotation_semantics_changed')
    sources={}
    for obj in (OperationalSpaceController,type(c).__mro__[1],set_goal_orientation,PandaGripper):
        path=Path(inspect.getfile(obj))
        sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(schema_version='astra-libero-runtime-v1',robosuite_version=robosuite.__version__,
        frame='world',reference='control_site',quaternion_order='wxyz',
        control_site_name=core.sim.model.site_id2name(sid),eef_body_name=robot.robot_model.eef_name,
        body_to_site_rotation=fixed.tolist(),input_min=imin.tolist(),input_max=imax.tolist(),
        output_min=omin.tolist(),output_max=omax.tolist(),action_min=low.tolist(),action_max=high.tolist(),
        control_freq=float(core.control_freq),physics_timestep=float(core.sim.model.opt.timestep),
        gripper=dict(keep=0.,open=-1.,closed=1.),sources=sources,motion_verified=False,
        scale_roundtrip_verified=True,world_left_rotation_verified=True)


class LiberoEEFAdapter:
    def __init__(self, runtime, limits):
        if runtime['frame']!='world' or runtime['reference']!='control_site':
            raise ValueError('unsupported_frame')
        self.runtime=runtime; self.limits=limits
        self.fixed=np.asarray(runtime['body_to_site_rotation'],dtype=np.float64)
        matrix_to_quat_wxyz(self.fixed)
        for k,n in (('input_min',6),('input_max',6),('output_min',6),('output_max',6),
                    ('action_min',7),('action_max',7)):
            setattr(self,k,vec(runtime[k],n,k))

    def read_pose(self, raw):
        return ControlPose(vec(raw['robot0_eef_pos'],3,'eef_position'),
            body_xyzw_to_site_wxyz(raw['robot0_eef_quat'],self.fixed))

    def _workspace(self,p):
        if np.any(p<self.limits.workspace_min) or np.any(p>self.limits.workspace_max):
            raise ValueError('workspace_violation')

    def resolve_target(self,d,pose):
        self._workspace(pose.position)
        if d['mode']=='eef':
            target=ControlPose(vec(d['target']['position'],3,'target'),
                               vec(d['target']['quaternion_wxyz'],4,'target_quat'))
            grip='closed' if d['target']['gripper_closed'] else 'open'
        elif d['mode']=='eef_delta':
            delta=d['delta']
            if np.linalg.norm(delta['delta_rotation_vector'])>self.limits.max_target_rotation:
                raise ValueError('target_rotation_limit')
            p,q=resolve_delta(pose.position,pose.quaternion_wxyz,
                              delta['delta_position'],delta['delta_rotation_vector'])
            target=ControlPose(p,q); grip=delta['gripper']
        else: raise ValueError('not_an_eef_mode')
        self._workspace(target.position)
        dp,dr=pose_error(target.position,target.quaternion_wxyz,pose.position,pose.quaternion_wxyz)
        if np.linalg.norm(dp)>self.limits.max_target_translation or np.linalg.norm(dr)>self.limits.max_target_rotation:
            raise ValueError('target_total_limit')
        return FixedTarget(target,pose,grip,d['decision_id'])

    def error(self,target,pose):
        return pose_error(target.pose.position,target.pose.quaternion_wxyz,pose.position,pose.quaternion_wxyz)

    def pose_reached(self,target,pose):
        dp,dr=self.error(target,pose)
        return bool(np.linalg.norm(dp)<=self.limits.position_tolerance and
                    np.linalg.norm(dr)<=self.limits.rotation_tolerance)

    def next_action(self,target,pose):
        self._workspace(pose.position)
        travelled,rotated=pose_error(pose.position,pose.quaternion_wxyz,
            target.origin.position,target.origin.quaternion_wxyz)
        if np.linalg.norm(travelled)>self.limits.max_target_translation+self.limits.position_tolerance:
            raise ValueError('measured_translation_limit')
        if np.linalg.norm(rotated)>self.limits.max_target_rotation+self.limits.rotation_tolerance:
            raise ValueError('measured_rotation_limit')
        dp,dr=self.error(target,pose)
        delta=np.concatenate((
            bounded_increment(dp,self.limits.max_translation_per_step,self.output_min[:3],self.output_max[:3]),
            bounded_increment(dr,self.limits.max_rotation_per_step,self.output_min[3:],self.output_max[3:])))
        native=inverse_scale(delta,self.input_min,self.input_max,self.output_min,self.output_max)
        a=np.r_[native,float(self.runtime['gripper'][target.gripper_intent])]
        self.validate_native_action(a)
        return a

    def validate_native_action(self, action):
        a=vec(action,7,'native_action')
        if np.any(a<self.action_min) or np.any(a>self.action_max):
            raise ValueError('action_spec_violation')

    @staticmethod
    def synchronize_controller(env):
        # One explicit owner boundary, never at every step.
        controller=env.env.robots[0].controller
        controller.update(force=True)
        controller.reset_goal()
