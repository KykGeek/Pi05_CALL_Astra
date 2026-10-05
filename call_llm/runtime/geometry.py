"""World-frame control-site geometry and OSC affine inversion."""
import numpy as np
from scipy.spatial.transform import Rotation


def vec(value, n, name):
    a = np.asarray(value, dtype=np.float64)
    if a.shape != (n,) or not np.isfinite(a).all():
        raise ValueError(name + ':invalid_vector')
    return a.copy()


def quat_wxyz_to_matrix(q):
    q = vec(q, 4, 'quaternion_wxyz')
    norm = np.linalg.norm(q)
    if abs(norm - 1.0) > 1e-5:
        raise ValueError('quaternion_not_unit')
    return Rotation.from_quat((q / norm)[[1, 2, 3, 0]]).as_matrix()


def matrix_to_quat_wxyz(r):
    r = np.asarray(r, dtype=np.float64)
    if (r.shape != (3, 3) or not np.isfinite(r).all()
            or not np.allclose(r.T @ r, np.eye(3), atol=1e-7, rtol=0)
            or not np.isclose(np.linalg.det(r), 1., atol=1e-7, rtol=0)):
        raise ValueError('invalid_rotation_matrix')
    q = Rotation.from_matrix(r).as_quat()[[3, 0, 1, 2]]
    return -q if q[0] < 0 else q


def body_xyzw_to_site_wxyz(q_body, fixed_rotation):
    q = vec(q_body, 4, 'body_quaternion_xyzw')
    matrix_to_quat_wxyz(fixed_rotation)
    return matrix_to_quat_wxyz(quat_wxyz_to_matrix(q[[3, 0, 1, 2]]) @ fixed_rotation)


def resolve_delta(p_current, q_current, dp, dr):
    p = vec(p_current, 3, 'position') + vec(dp, 3, 'delta_position')
    r = Rotation.from_rotvec(vec(dr, 3, 'delta_rotation')).as_matrix()
    return p, matrix_to_quat_wxyz(r @ quat_wxyz_to_matrix(q_current))


def pose_error(p_target, q_target, p_current, q_current):
    dp = vec(p_target, 3, 'target') - vec(p_current, 3, 'position')
    r = quat_wxyz_to_matrix(q_target) @ quat_wxyz_to_matrix(q_current).T
    return dp, Rotation.from_matrix(r).as_rotvec()


def bounded_increment(error, max_norm, output_min, output_max):
    e = vec(error, 3, 'error')
    lo, hi = vec(output_min, 3, 'output_min'), vec(output_max, 3, 'output_max')
    if not np.isfinite(max_norm) or max_norm <= 0 or np.any(lo >= 0) or np.any(hi <= 0):
        raise ValueError('unsupported_increment_bounds')
    factor = min(1., max_norm / max(float(np.linalg.norm(e)), 1e-300))
    for j, x in enumerate(e):
        if x > 0:
            factor = min(factor, hi[j] / x)
        elif x < 0:
            factor = min(factor, lo[j] / x)
    return e * factor


def inverse_scale(delta, input_min, input_max, output_min, output_max):
    d, lo, hi, ol, oh = (vec(v, 6, name) for v, name in (
        (delta, 'delta'), (input_min, 'input_min'), (input_max, 'input_max'),
        (output_min, 'output_min'), (output_max, 'output_max')))
    if np.any(hi <= lo) or np.any(oh <= ol):
        raise ValueError('invalid_scale_range')
    if np.any(d < ol) or np.any(d > oh):
        raise ValueError('physical_delta_out_of_range')
    u = (d - (oh + ol) / 2) / ((oh - ol) / (hi - lo)) + (hi + lo) / 2
    if not np.isfinite(u).all() or np.any(u < lo - 1e-12) or np.any(u > hi + 1e-12):
        raise ValueError('native_action_out_of_range')
    return u
