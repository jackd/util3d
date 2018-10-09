"""Non-homogeneous transforms."""
import numpy as np


def coordinate_transform(x, R=None, t=None):
    assert(len(R.shape) == 2)
    if len(x.shape) == 1:
        if R is not None:
            x = np.matmul(R, x)
    else:
        if R is not None:
            x = np.matmul(x, R.T)
    if t is not None:
        x += t
    return x


def transform_transform(R, t, R1=None, t1=None):
    """Get R, t resulting from transform (R, t) followed by (R2, t2)."""
    if R1 is not None:
        R = np.matmul(R1, R)
    t = coordinate_transform(t, R1, t1)
    return R, t


def inverse_coordinate_transform(R, t, x):
    assert(len(R.shape) == 2)
    if len(x.shape) == 1:
        return np.matmul(R.T, x-t)
    else:
        return np.matmul(x-t, R)


def transform_inverse(R, t=None):
    Rt = R.T
    if t is None:
        return Rt, None
    else:
        return Rt, -np.matmul(Rt, t)


def transform_from_homogeneous(T):
    return T[:3, :3], T[:3, -3]


def transform_to_homogeneous(R, t, dtype=None):
    if dtype is None:
        dtype = R.dtype
    T = np.zeros((4, 4), dtype=dtype)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def vector_to_homogeneous(x):
    ones = np.ones(x.shape[:-1] + (1,), dtype=x.dtype)
    return np.concatenate((x, ones), axis=-1)


def vector_from_homogeneous(x):
    xyz, w = np.split(x, (3, 1), axis=-1)
    return xyz / w


def _get_eye_rotation_matrix(eye, center=None, world_up=None, axis=-1):
    if center is None:
        center = np.zeros_like(eye)
    if world_up is None:
        world_up = np.zeros_like(eye)
        world_up[..., -1] = 1
    n = eye - center
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    u = np.cross(world_up, n)
    u /= np.linalg.norm(u, axis=-1, keepdims=True)
    v = np.cross(n, u)
    v /= np.linalg.norm(v, axis=-1, keepdims=True)

    return np.stack((u, v, n), axis=axis)


def get_world_to_eye_transform(eye, center=None, world_up=None):
    R = _get_eye_rotation_matrix(eye, center, world_up, axis=-2)
    t = np.matmul(R, -eye)
    return R, t


def get_eye_to_world_transform(eye, center=None, world_up=None):
    R = _get_eye_rotation_matrix(eye, center, world_up, axis=-1)
    t = eye
    return R, t
