import numpy as np


def get_centroids(vertices, faces):
    return np.mean(vertices[faces], axis=-2)


def get_normals(vertices, faces, normalize=True):
    ps = vertices[faces]
    p0 = ps[..., 1, :] - ps[..., 0, :]
    p1 = ps[..., 2, :] - ps[..., 0, :]
    normals = np.cross(p0, p1)
    if normalize:
        norms = np.sqrt(np.sum(normals**2, axis=-1, keepdims=True))
        norms[norms == 0] = 1e-8
        normals /= norms
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
