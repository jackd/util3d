import numpy as np
from point_cloud import rescale_points
from parser import parse_obj
from geom import compute_normals


def to_glumpy_buffers(
        positions, face_positions,
        texcoords=None, face_texcoords=None,
        normals=None, face_normals=None,
        rescale=False):
    """Convert `parse_obj_file` output to glumpy vertex/indices buffers."""
    from glumpy import gloo
    # Building the vertices
    face_positions = face_positions.reshape(-1)
    vtype = [('position', np.float32, 3)]

    if texcoords is not None:
        face_texcoords = face_texcoords.reshape(-1)
        vtype.append(('texcoord', np.float32, 2))
    if normals is not None:
        face_normals = face_normals.reshape(-1)
        vtype.append(('normal', np.float32, 3))

    vertices = np.empty(len(face_positions), vtype)
    vertices["position"] = positions[face_positions]
    # vertices = np.empty(len(positions), vtype)
    # vertices["position"] = positions
    if texcoords is not None:
        vertices["texcoord"] = texcoords[face_texcoords]
        # vertices["texcoord"] = texcoords
    if normals is not None:
        vertices["normal"] = normals[face_normals]
        # vertices["normal"] = normals
    vertices = vertices.view(gloo.VertexBuffer)

    if rescale:
        # Centering and scaling to fit the unit box
        rescale_points(vertices["position"])

    itype = np.uint32
    indices = np.arange(len(vertices), dtype=itype).reshape(-1, 3)
    # indices = face_positions
    indices = indices.view(gloo.IndexBuffer)
    return vertices, indices


def load_glumpy_data(file_or_filename, rescale=False, calculate_normals=False):
    positions, face_positions, texcoords, face_texcoords, \
        normals, face_normals = parse_obj(file_or_filename)
    if calculate_normals:
        assert(face_normals is None)
        normals = compute_normals(positions, face_positions)
        face_normals = np.array(
            [[i, i, i] for i in range(len(normals))], dtype=np.uint32)
    vertices, indices = to_glumpy_buffers(
            positions, face_positions, texcoords, face_texcoords, normals,
            face_normals, rescale=rescale)
    return vertices, indices
