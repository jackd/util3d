"""
Visualization functions using mayavi.mlab.

Calls should be following by `mayavi.mlab.show()`.
"""
import numpy as np
from mayavi import mlab

_dim = {'x': 0, 'y': 1, 'z': 2}


def permute_xyz(x, y, z, order='xyz'):
    data = (x, y, z)
    return tuple(data[_dim[k]] for k in order)


def vis_point_cloud(points, axis_order='xyz', value=None, **kwargs):
    data = permute_xyz(*points.T, order=axis_order)
    if value is not None:
        data = data + (value,)
    mlab.points3d(*data, **kwargs)


def vis_segmented_cloud(points, idxs, colors=None, **kwargs):
    if colors is None:
        colors = (
            (0, 0, 0),
            (0, 0, 1),
            (0, 1, 0),
            (0, 1, 1),
            (1, 0, 0),
            (1, 0, 1),
            (1, 1, 0),
            (1, 1, 1),
        )
    n_means = np.max(idxs)+1
    clouds = [points[idxs == i] for i in range(n_means)]
    for i in range(n_means):
        cloud = clouds[i]
        color = colors[i % len(colors)]
        vis_point_cloud(cloud, color=color, **kwargs)


def vis_voxels(voxels, axis_order='xzy', **kwargs):
    data = permute_xyz(*np.where(voxels), order=axis_order)
    if len(data[0]) == 0:
        # raise ValueError('No voxels to display')
        Warning('No voxels to display')
    else:
        if 'mode' not in kwargs:
            kwargs['mode'] = 'cube'
        mlab.points3d(*data, **kwargs)


def vis_mesh(
        vertices, faces, axis_order='xyz', include_wireframe=True,
        color=(0, 0, 1), **kwargs):
    if len(faces) == 0:
        print('Warning: no faces')
        return
    x, y, z = permute_xyz(*vertices.T, order=axis_order)
    mlab.triangular_mesh(x, y, z, faces, color=color, **kwargs)
    if include_wireframe:
        mlab.triangular_mesh(
            x, y, z, faces, color=(0, 0, 0), representation='wireframe')
