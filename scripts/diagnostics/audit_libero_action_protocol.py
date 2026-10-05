"""Bounded robot-only protocol diagnostic; never loads a policy or an object state."""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from call_llm.runtime.libero_adapter import audit_runtime, LiberoEEFAdapter, Limits
from call_llm.runtime.geometry import matrix_to_quat_wxyz, quat_wxyz_to_matrix


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--motion',action='store_true')
    p.add_argument('--task-id',type=int,default=0)
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError('Diagnostic output already exists')
    os.environ.setdefault('MUJOCO_GL','egl')
    openpi_root_value = os.environ.get("OPENPI_ROOT")
    if openpi_root_value:
        sys.path.insert(0, str(Path(openpi_root_value) / "third_party" / "libero"))
    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark_dict
    from libero.libero.envs import OffScreenRenderEnv
    suite=get_benchmark_dict()['libero_10']()
    task=suite.get_task(args.task_id)
    path=Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
    env=OffScreenRenderEnv(bddl_file_name=str(path),camera_heights=128,camera_widths=128,
                           render_gpu_device_id=1)
    report={}
    try:
        env.seed(0)
        raw=env.reset()
        report=audit_runtime(env,raw)
        report['task_id']=args.task_id
        report['steps']=[]
        if args.motion:
            # Explicit robot-local diagnostic box, not a general task workspace.
            origin=np.asarray(raw['robot0_eef_pos']).copy()
            adapter=LiberoEEFAdapter(report,Limits(origin-.06,origin+.06))
            report['diagnostic_workspace']={'min':(origin-.06).tolist(),'max':(origin+.06).tolist(),
                                            'purpose':'bounded_initial_robot_pose_diagnostic_only'}
            adapter.synchronize_controller(env)
            # Small signed target increments for each translational/rotational axis.
            motions=[(axis,sign) for axis in range(6) for sign in (1.,-1.)]
            for axis,sign in motions:
                pose=adapter.read_pose(raw)
                dp=np.zeros(3); dr=np.zeros(3)
                if axis<3: dp[axis]=sign*.003
                else: dr[axis-3]=sign*.02
                decision=dict(mode='eef_delta',decision_id=f'axis-{axis}-{sign}',
                    delta=dict(delta_position=dp.tolist(),delta_rotation_vector=dr.tolist(),gripper='keep'))
                target=adapter.resolve_target(decision,pose)
                before=pose
                for _ in range(3):
                    command=adapter.next_action(target,adapter.read_pose(raw))
                    raw,_,done,_=env.step(command.tolist())
                    robot=env.env.robots[0]
                    actual=np.asarray(env.sim.data.site_xmat[robot.eef_site_id]).reshape(3,3)
                    mapped=quat_wxyz_to_matrix(adapter.read_pose(raw).quaternion_wxyz)
                    if not np.allclose(actual,mapped,atol=1e-6,rtol=0):
                        raise RuntimeError('body_site_mapping_not_fixed')
                    if done: raise RuntimeError('diagnostic_environment_terminated')
                after=adapter.read_pose(raw)
                from call_llm.runtime.geometry import pose_error
                measured_p,measured_r=pose_error(after.position,after.quaternion_wxyz,
                                                before.position,before.quaternion_wxyz)
                report['steps'].append(dict(axis=axis,sign=sign,
                    measured_translation=measured_p.tolist(),measured_rotation=measured_r.tolist(),
                    remaining_error=[x.tolist() for x in adapter.error(target,after)]))
            gripper=env.env.robots[0].gripper
            original=gripper.current_action.copy()
            try:
                outcomes={}
                for value,label in ((0.,'keep'),(-1.,'open'),(1.,'closed')):
                    gripper.current_action=np.zeros_like(original)
                    outcomes[label]=gripper.format_action(np.array([value])).tolist()
                report['gripper_format_action']=outcomes
                if outcomes['keep'] != [0.,0.]: raise RuntimeError('keep_semantics_changed')
                if not (outcomes['open'][0]>0 and outcomes['closed'][0]<0):
                    raise RuntimeError('gripper_sign_changed')
            finally: gripper.current_action=original
            # Motion requires directional evidence, not merely exception-free execution.
            signs=[]
            for row in report['steps']:
                axis=row['axis']; delta=row['measured_translation'] if axis<3 else row['measured_rotation']
                signs.append(delta[axis%3]*row['sign']>0)
            report['direction_checks']=signs
            report['motion_verified']=all(signs)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False))
        print(json.dumps(dict(output=str(args.output),motion_verified=report['motion_verified'],
                              frame=report['frame'],input_min=report['input_min'],output_max=report['output_max'])))
    finally: env.close()


if __name__=='__main__': main()
