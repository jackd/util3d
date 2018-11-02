from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from util3d.mesh.off import OffObject
from util3d.mesh.geom import triangulated_faces
from util3d.voxel.convert import mesh_to_voxels
from util3d.voxel.binvox import Voxels


def get_voxel_data(dim=256):
    folder = os.path.realpath(os.path.dirname(__file__))
    vox_path = os.path.join(folder, 'bunny%d.binvox' % dim)

    if not os.path.isfile(vox_path):
        bunny_path = os.path.join(folder, 'bunny.off')
        bunny = OffObject.from_path(bunny_path)

        vertices = bunny.vertices
        faces = tuple(triangulated_faces(bunny.faces))
        voxels = mesh_to_voxels(vertices, faces, dim)

        with open(vox_path, 'w') as fp:
            voxels.save_to_file(fp)
        del voxels

    with open(vox_path, 'r') as fp:
        voxels = Voxels.from_file(fp)
    return voxels
