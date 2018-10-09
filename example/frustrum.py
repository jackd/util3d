#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from util3d.transform.frustrum import get_ray_eye_coordinates
from util3d.transform.nonhom import get_eye_to_world_transform
# from util3d.transform.nonhom import get_world_to_eye_transform
from util3d.transform.nonhom import coordinate_transform


def vis(cloud):
    from mayavi import mlab
    from util3d.mayavi_vis import vis_point_cloud, vis_axes
    vis_axes()
    vis_point_cloud(cloud, color=(0, 0, 1), scale_factor=0.2)
    mlab.show()


ray_shape = (8,)*3
f = 32 / 35
cloud = get_ray_eye_coordinates(ray_shape, f, 0.5, 1.5, include_corners=False)
eye = np.array([1, 0, 0.6])
# transform = get_world_to_eye_transform(eye)
transform = get_eye_to_world_transform(eye)
cloud = coordinate_transform(cloud, *transform)
vis(cloud)
