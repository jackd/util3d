import numpy as np


def compute_normals(positions, faces):
    ps = positions[faces]
    p0 = ps[..., 1] - ps[..., 0]
    p1 = ps[..., 2] - ps[..., 1]
    normals = np.cross(p0, p1)
    return normals
