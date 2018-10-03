from __future__ import division
import os
import numpy as np

_bv_path = os.path.join(
    os.path.realpath(os.path.dirname(__file__)), 'bin', 'binvox')


def voxels_to_point_cloud(voxels, dtype=np.float32):
    """
    Converts voxels to a point cloud, or (N, 3) array.

    `N == np.sum(voxels).`
    """
    return np.array(np.where(voxels), dtype=dtype).T


def indices_to_array(i, j, k, dims):
    """
    Convert array-like i, j and k to voxel grid.

    Args:
        `i`, `j`, `k`: array-like, length `N`.
        `dims`: shape of output voxel grid.

    Returns:
        `(N, 3)` numpy array, satisfying `np.all(out[i, j, k] == True)`.
    """
    out = np.zeros(dims, dtype=np.bool)
    out[i, j, k] = True
    return out


def point_cloud_to_array(xyz, dims):
    """
    Convert a point cloud (`xyz`) to voxel grid.

    Args:
        `xyz`: (N, 3) coordinates of points all in [0, dims).
        `dims`: shape of final voxel grid, 3-tuple.

    Returns
        voxel grid of shape `dims`.
    """

    i, j, k = xyz.astype(np.int32).T
    return indices_to_array(i, j, k, dims)


def point_cloud_to_voxel_indices_converter(dims, mins, maxs, ensure_valid):
    """
    Get a converter function that converts a point cloud to voxel indices.

    Args:
        `dims`: 3-tuple, shape of voxel grid for which indices apply
        `mins`: 3-element array-like, min values for bounding cube on voxel
            voxel_grid
        `mins`: 3-element array-like, max values for bounding cube on voxel
            voxel_grid
        `ensure_valid`: if True, only returns indices for points inside
            the bounding cube defined by `[mins, maxs]`.

    Returns:
        Function mapping (N, 3) array of point cloud positions to (M, 3) int
        array of voxel indices. If `ensure_valid`, `M <= N` and outputs are
        all in the range `[0, dims)`. Otherwise, `N == M` and invalid
        voxel indices are allowed.
    """
    mins = np.array(mins)
    maxs = np.array(maxs)
    dims = np.array(dims)

    if not np.all(maxs - mins > 0):
        raise ValueError('maxes must all be greater than mins.')

    def convert(cloud):
        cloud = np.array(cloud)
        if len(cloud.shape) != 2:
            raise ValueError(
                'cloud must be N by 3, got shape %s' % str(cloud.shape))
        cloud -= mins
        cloud *= dims / (maxs - mins)
        cloud = cloud.astype(np.int32)
        if ensure_valid:
            valid = np.all(
                np.logical_and(0 <= cloud, cloud < dims), axis=1)
            cloud = cloud[valid]
        return cloud[:, 0], cloud[:, 1], cloud[:, 2]

    return convert


def point_cloud_to_voxel_indices(
        cloud, dims, mins=None, maxs=None, ensure_valid=False):
    """
    Get voxel indices for a point cloud.

    Args:
        `cloud`: `(n_points, 3)` numpy array
        dims: scalar or length 3 list/tuple/ndarray of output voxel
            dimensions
        `mins`: scalar or length 3 list/tuple/ndarray of minx, miny, minz
        `maxs`: scalar or length 3 list/tuple/ndarray of maxx, maxy, maxz
        `ensure_valid`: if True, removes all rows that correspond to points
            outside of the voxel grid `[0, dims)`

    Returns:
        `x`, `y`, `z` numpy arrays indicating occupied voxels, with number of
            elements equal to the number of valid points in the original cloud
            if `ensure_valid` is `True`, otherwise `n_points`.
    """
    if mins is None:
        mins = np.min(cloud, axis=0)
    elif maxs is None:
        maxs = np.max(cloud, axis=0)
        maxs += np.abs(maxs*1e-7)
    converter = point_cloud_to_voxel_indices_converter(
            dims, mins, maxs, ensure_valid)

    return converter(cloud)


def obj_to_binvox(
        obj_path, binvox_path, voxel_dim=32,
        bounding_box=(-0.5, -0.5, -0.5, 0.5, 0.5, 0.5),
        pb=True, exact=True, dc=True, aw=True, c=False, v=False):
    import subprocess
    _FNULL = open(os.devnull, 'w')
    if not os.path.isfile(obj_path):
        raise IOError('No obj file at %s' % obj_path)
    original_bv_path = obj_path[:-4] + '.binvox'
    if os.path.isfile(original_bv_path):
        raise IOError('File already exists at %s' % original_bv_path)
    args = [_bv_path, '-d', str(voxel_dim), '-bb']
    args.extend([str(b) for b in bounding_box])
    for condition, flag in (
            (pb, '-pb'), (exact, '-e'), (dc, '-dc'), (aw, '-aw'), (c, '-c'),
            (v, '-v')):
        if condition:
            args.append(flag)
    args.append(obj_path)
    subprocess.call(args, stdout=_FNULL, stderr=subprocess.STDOUT)
    if not os.path.isfile(original_bv_path):
        raise IOError('No binvox file at %s' % original_bv_path)
    os.rename(original_bv_path, binvox_path)


def mesh_to_binvox(
        vertices, faces, binvox_path, voxel_dim, *args, **kwargs):
    """
    Create a .binvox file at `binvox_path` based on mesh provided.

    `voxel_dim`, `args`, and `kwargs` are passed to `obj_to_binvox`.
    """
    from dids.file_io.temp import TempPath
    from util3d.mesh.obj_io import write_obj_file
    with TempPath(extension='.obj') as obj_path:
        with open(obj_path, 'w') as fp:
            write_obj_file(fp, vertices, faces)
        obj_to_binvox(obj_path, binvox_path, voxel_dim, *args, **kwargs)


def mesh_to_voxels(vertices, faces, voxel_dim, *args, **kwargs):
    """
    Get a `Voxels` instance based on the provided mesh.

    `voxel_dim`, `args` and `kwargs` are passed to `obj_to_binvox`.

    Returns a `Voxels` instance.
    """
    from dids.file_io.temp import TempPath
    from binvox import DenseVoxels
    with TempPath(extension='.binvox') as binvox_path:
        mesh_to_binvox(
            vertices, faces, binvox_path, voxel_dim, *args, **kwargs)
        with open(binvox_path, 'rb') as fp:
            voxels = DenseVoxels.from_file(fp)
    return voxels
