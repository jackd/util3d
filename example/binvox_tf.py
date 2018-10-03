#!/usr/bin/python
import os
import numpy as np
from mayavi import mlab
from util3d.mayavi_vis import vis_voxels
# from util3d.mayavi_vis import vis_contours
from util3d.mesh.off import OffObject
from util3d.mesh.geom import triangulated_faces
from util3d.voxel.convert import mesh_to_voxels
from util3d.voxel.binvox import Voxels
from util3d.voxel.manip_tf import orthographic_filled_voxels
from util3d.voxel.manip import orthographic_filled_voxels as v0
import tensorflow as tf
tf.enable_eager_execution()

folder = os.path.realpath(os.path.dirname(__file__))
dim = 128
vox_path = os.path.join(folder, 'data', 'bunny%d.binvox' % dim)

if not os.path.isfile(vox_path):
    bunny_path = os.path.join(folder, 'data', 'bunny.off')
    bunny = OffObject.from_path(bunny_path)

    vertices = bunny.vertices
    faces = tuple(triangulated_faces(bunny.faces))
    voxels = mesh_to_voxels(vertices, faces, dim)

    with open(vox_path, 'w') as fp:
        voxels.save_to_file(fp)
    del voxels

with open(vox_path, 'r') as fp:
    voxels = Voxels.from_file(fp)

dense = voxels.dense_data()
filled = np.array(orthographic_filled_voxels(dense))
filled_v0 = v0(dense)
print(np.sum(dense), np.sum(filled), np.all(filled_v0 == filled))
mlab.figure()
# vis_contours(dense, contours=[0.5], color=(0, 0, 1))
vis_voxels(dense, color=(0, 0, 1), axis_order='xyz')
mlab.figure()
vis_voxels(filled, color=(0, 1, 0), axis_order='xyz')
mlab.show()
