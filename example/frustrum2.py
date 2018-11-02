#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from util3d.transform.frustrum import voxel_values_to_frustrum
from util3d.transform.frustrum import frustrum_values_to_voxels
from util3d.transform.nonhom import get_eye_to_world_transform
# from util3d.transform.nonhom import get_world_to_eye_transform
from util3d.voxel.manip import orthographic_filled_voxels

from data import get_voxel_data

dim = 256

dense = get_voxel_data(dim).dense_data()
filled = orthographic_filled_voxels(dense)


eye = np.ones((3,), dtype=np.float32)
f = 2.0
r3 = np.sqrt(3)
z_near = np.linalg.norm(eye) - 0.5*r3
z_far = z_near + r3

R, t = get_eye_to_world_transform(eye)

ray_shape = (dim,)*3
vox_shape = ray_shape
include_corners = True


def vis(*v):
    from mayavi import mlab
    from util3d.mayavi_vis import vis_contours
    for i, vi in enumerate(v):
        mlab.figure()
        color = [0, 0, 0]
        color[i] = 1
        vis_contours(vi, contours=[0.5], color=tuple(color))
    mlab.show()


def double_transform(voxel_values):
    frust_vals, frust_ins = voxel_values_to_frustrum(
        voxel_values, R, t, f, z_near, z_far, ray_shape, include_corners,
        interpolation_order=1)

    vox_vals, vox_ins = frustrum_values_to_voxels(
        frust_vals, R, t, f, z_near, z_far, vox_shape, include_corners,
        interpolation_order=1)

    vis(voxel_values, frust_vals, vox_vals)


double_transform(filled)
