import numpy as np


def compute_normals(positions, faces):
    ps = positions[faces]
    p0 = ps[..., 1] - ps[..., 0]
    p1 = ps[..., 2] - ps[..., 1]
    normals = np.cross(p0, p1)
    return normals


def triangulated_faces(faces):
    """
    Get a generator that iterates over the triangular faces.

    e.g.
    tuple(triangular_faces([[0, 1, 2, 3]])) == (0, 1, 2), (0, 2, 3)
    """
    for face in faces:
        for i in range(len(face) - 2):
            yield (face[0], face[i+1], face[i+2])
