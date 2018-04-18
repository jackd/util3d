"""Utility functions for parsing .wrl files and manipulating SceneGraphs."""
import numpy as np
from vrml.vrml97.parser import buildParser
_parser = buildParser()

parse_data = _parser.parse


def parse_file(path_or_file):
    if hasattr(path_or_file, 'read'):
        data = path_or_file.read()
    else:
        with open(path_or_file, 'r') as fp:
            data = fp.read()
    return parse_data(data)


def geometry_to_mesh(geometry):
    """Convert vrml geometry object to (vertices, faces)."""
    vertices = geometry.coord.point
    faces = np.reshape(geometry.coordIndex, (-1, 4))[:, :3]
    return vertices, faces


def scene_to_mesh(scene):
    from geom import combine_meshes
    return combine_meshes(
        (geometry_to_mesh(c.geometry) for c in scene.children))
