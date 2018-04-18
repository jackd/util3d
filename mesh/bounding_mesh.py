"""
Depends on bounding-mesh and vrml

bounding-mesh: https://github.com/gaschler/bounding-mesh
vrml: pip install PyVRML simpleparse
"""
import os
from util3d.temp_path import get_temp_dir


class BoundingMeshConfig(object):
    def __init__(
            self,
            bin_name='boundingmesh',
            direction=None,
            vertices=None,
            error=None,
            metric=None,
            init=None):
        self.bin = bin_name
        self.direction = direction
        self.vertices = vertices
        self.error = error
        self.metric = metric
        self.init = init
        args = []
        kwargs = {
            '-d': direction,
            '-v': vertices,
            '-e': error,
            '-m': metric,
            '-i': init
        }
        self.kwargs = {k: v for k, v in kwargs.items() if v is not None}
        args = ((k, v) for k, v in self.kwargs.items())
        self.args = tuple(item for sublist in args for item in sublist)

    def convert_file(self, filename_in, filename_out, verbose=True):
        import subprocess
        if verbose:
            FNULL = open(os.devnull, 'w')
            kwargs = dict(stdout=FNULL, stderr=subprocess.STDOUT)
        else:
            kwargs = {}
        if not os.path.isfile(filename_in):
            raise IOError('No file found at %s' % filename_in)
        args = (self.bin,) + self.args + (filename_in, filename_out)
        subprocess.call(args, **kwargs)
        return filename_out if filename_out is not None else (
            'boundingmesh_%s' % filename_in)

    def write_mesh(self, vertices, faces, filename_out, verbose=True):
        from .obj_io import write_obj_file
        with get_temp_dir() as temp_dir:
            path = os.path.join(temp_dir, 'model.obj')
            with open(path, 'w') as fp:
                write_obj_file(fp, vertices, faces)
            self.convert_file(path, filename_out, verbose=verbose)

    def convert_mesh(self, vertices, faces, verbose=True):
        from .obj_io import parse_obj_file
        with get_temp_dir() as temp_dir:
            path = os.path.join(temp_dir, 'model.obj')
            self.write_mesh(vertices, faces, path, verbose=verbose)
            with open(path, 'r') as fp:
                mesh = parse_obj_file(fp)[:2]
        return mesh
