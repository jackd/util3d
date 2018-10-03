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


def vis_voxels(voxels, axis_order='xzy', scale=None, shift=None, **kwargs):
    data = permute_xyz(*np.where(voxels), order=axis_order)
    if len(data[0]) == 0:
        # raise ValueError('No voxels to display')
        Warning('No voxels to display')
    else:
        if scale is not None or shift is not None:
            data = np.stack(data, axis=-1)
            if scale is not None:
                data = [d / scale for d in data]
            if shift is not None:
                data = [d - shift for d in data]
            data = np.unstack(data, axis=-1)
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


def vis_contours(data, contours=[0], opacity=0.5, axis_order='xyz', **kwargs):
    if axis_order != 'xyz':
        data = data.transpose(tuple(_dim[w] for w in axis_order))
    if data.dtype not in (np.float32, np.float64):
        data = data.astype(np.float32)
    mlab.contour3d(data, contours=contours, opacity=opacity, **kwargs)


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


def vis_axes(length=1):
    mlab.quiver3d([0], [0], [0], [length], [0], [0], color=(1, 0, 0))
    mlab.quiver3d([0], [0], [0], [0], [length], [0], color=(0, 1, 0))
    mlab.quiver3d([0], [0], [0], [0], [0], [length], color=(0, 0, 1))
