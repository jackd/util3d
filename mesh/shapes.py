"""
Create a sphere by recursively subdividing each face.

Start with a tetrahedron
"""
import numpy as np


def batch_normalize(vertex):
    vertex /= np.sqrt(np.sum(vertex**2, axis=-1, keepdims=True))
    return vertex


def normalize(vertex):
    vertex /= np.sqrt(np.sum(vertex**2))
    return vertex


def get_tetrahedon_mesh(vertex_dtype=np.float32):
    """Get the vertices and faces of a unit tetrahedron."""
    vertices = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1]
    ], dtype=vertex_dtype)
    vertices /= np.sqrt(3)
    faces = (
        (0, 2, 1),
        (0, 3, 1),
        (1, 2, 3),
        (2, 0, 3)
    )
    return vertices, faces


def get_subdivided_sphere_mesh(n_subdivisions=2):
    from subdivide import subdivide
    vertices, faces = get_tetrahedon_mesh()
    for i in range(n_subdivisions):
        vertices, faces = subdivide(vertices, faces)
        # normalize
        vertices = np.array(vertices)
        batch_normalize(vertices)
    return vertices, faces


def get_edge_split_sphere_mesh(edge_length_threshold):
    from edge_splitter import split_to_threshold
    vertices, faces = get_tetrahedon_mesh()
    return split_to_threshold(
        vertices, faces, edge_length_threshold, normalize)
