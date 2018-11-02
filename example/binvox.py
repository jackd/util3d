#!/usr/bin/python
import numpy as np
from mayavi import mlab
# from util3d.mayavi_vis import vis_contours
# from util3d.mayavi_vis import vis_voxels
from util3d.mayavi_vis import vis_contours
from util3d.voxel.manip import orthographic_filled_voxels

from data import get_voxel_data

voxels = get_voxel_data(dim=256)
dense = voxels.dense_data()
filled = orthographic_filled_voxels(dense)
print(np.sum(dense), np.sum(filled))
mlab.figure()
# vis_contours(dense, contours=[0.5], color=(0, 0, 1))
# vis_voxels(dense, color=(0, 0, 1), axis_order='xyz')
vis_contours(dense, contours=[0.5], color=(0, 0, 1))
mlab.figure()
# vis_contours(filled, contours=[0.5], color=(0, 1, 0))
# vis_voxels(filled, color=(0, 1, 0), axis_order='xyz')
vis_contours(filled, contours=[0.5], color=(0, 0, 1))
mlab.show()
