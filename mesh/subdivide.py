from __future__ import division


def subdivide(vertices, faces):
    """
    Create a new mesh by subdividing each face into 4 based on edge midpoints.

    Args:
        vertices: (nv, 3) numpy array of vertex coordinates, or list of numpy
            arrays.
        faces: (nf, 3) numpy array (or list/tuple of lists/tuples) of indices
            of vertices forming faces.

    Returns: vertices, faces of subdivided mesh.
    """
    nf = len(faces)
    sub_vertices = list(vertices)
    sub_faces = [None] * (4 * nf)
    midpoints = {}

    def get_midpoint(i, j):
        if i > j:
            i, j = j, i
        edge = (i, j)
        if edge in midpoints:
            return midpoints[edge]
        else:
            v = (vertices[i] + vertices[j]) / 2
            n = midpoints[edge] = len(sub_vertices)
            sub_vertices.append(v)
            return n

    for index, face in enumerate(faces):
        start = 4*index
        i, j, k = face
        ij = get_midpoint(i, j)
        jk = get_midpoint(j, k)
        ki = get_midpoint(k, i)
        sub_faces[start: start + 4] = (
            (ij, jk, ki),
            (i, ij, ki),
            (j, jk, ij),
            (k, ki, jk),
        )

    return sub_vertices, sub_faces
