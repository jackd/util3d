import numpy as np


def guarded_norm(x, axis=-1, eps=1e-8, keepdims=False):
    norms = np.sqrt(np.sum(x**2, axis=axis, keepdims=keepdims))
    norms[norms < eps] = eps
    return norms


def guarded_normalized(x, axis=-1, eps=1e-8):
    return x / guarded_norm(x, keepdims=True, axis=axis, eps=eps)


def guarded_normalize(x, axis=-1, eps=1e-8):
    x /= guarded_norm(x, keepdims=True, axis=axis, eps=eps)


def get_centroids(vertices, faces):
    return np.mean(vertices[faces], axis=-2)


def get_normals(vertices, faces, normalize=True, guard_eps=1e-8):
    ps = vertices[faces]
    p0 = ps[..., 1, :] - ps[..., 0, :]
    p1 = ps[..., 2, :] - ps[..., 0, :]
    normals = np.cross(p0, p1)
    if normalize:
        guarded_normalize(normals, eps=guard_eps)
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


def combine_meshes(meshes):
    """
    Combine the iterable of (vertices, faces) into a single (vertices, faces).

    Args:
        meshes: iterable of (vertices, faces)

    Returns:
        vertices: (nv, 3) float numpy array of vertex coordinates
        faces: (nf, 3) int numpy array of face vertex indices.
    """
    vertices = []
    faces = []
    nv = 0
    for v, f in meshes:
        vertices.append(v)
        faces.append(f + nv)
        nv += len(v)
    return np.concat(vertices, axis=0), np.concat(faces, axis=0)
