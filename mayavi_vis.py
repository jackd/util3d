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


def vis_normals(positions, normals, axis_order='xyz', **kwargs):
    x, y, z = permute_xyz(*positions.T, order=axis_order)
    u, v, w = permute_xyz(*normals.T, order=axis_order)
    mlab.quiver3d(x, y, z, u, v, w, **kwargs)


_colors = (
    (0, 0, 0),
    (0, 0, 1),
    (0, 1, 0),
    (0, 1, 1),
    (1, 0, 0),
    (1, 0, 1),
    (1, 1, 0),
    (1, 1, 1),
)


def vis_segmented_cloud(points, idxs, colors=None, **kwargs):
    if colors is None:
        colors = _colors
    n_means = np.max(idxs)+1
    clouds = [points[idxs == i] for i in range(n_means)]
    for i in range(n_means):
        cloud = clouds[i]
        color = colors[i % len(colors)]
        vis_point_cloud(cloud, color=color, **kwargs)


def vis_multi_clouds(clouds, colors=None, **kwargs):
    if colors is None:
        colors = _colors
    nc = len(colors)
    for i, cloud in enumerate(clouds):
        color = colors[i % nc]
        vis_point_cloud(cloud, color=color, **kwargs)


def vis_voxels(voxels, axis_order='xzy', **kwargs):
    data = permute_xyz(*np.where(voxels), order=axis_order)
    if len(data[0]) == 0:
        # raise ValueError('No voxels to display')
        Warning('No voxels to display')
    else:
        kwargs.setdefault('mode', 'cube')
        mlab.points3d(*data, **kwargs)


def vis_sliced(data, axis_order='xzy', **kwargs):
    if axis_order != 'xyz':
        data = data.transpose(tuple(_dim[w] for w in axis_order))
    mlab.volume_slice(data, **kwargs)


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


def vis_colored_point_cloud(points, colors, **kwargs):
    if colors.shape[-1] != 4:
        raise ValueError('colors must be an (n, 4) array of rgba values')
    n = len(points)
    if len(colors) != n:
        raise ValueError('colors must be the same length as points')
    scalars = np.arange(n)
    ones = np.ones((n,))
    x, y, z = points.T

    pts = mlab.quiver3d(
        x, y, z, ones, ones, ones, scalars=scalars, mode='sphere',
        **kwargs)
    pts.glyph.color_mode = 'color_by_scalar'  # Color by scalar
    pts.module_manager.scalar_lut_manager.lut.table = colors
    mlab.draw()
