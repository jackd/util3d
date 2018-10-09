from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from . import nonhom


def get_grid_points(n, include_corners, dtype=np.float32):
    if include_corners:
        return np.linspace(0, 1, n, dtype=dtype)
    else:
        x = np.arange(n, dtype=dtype)
        x += 0.5
        x /= n
        return x


def get_voxel_world_coords(
        dims, include_corners=False, dtype=np.float32, axis=-1):
    xyz = tuple(get_grid_points(d, include_corners, dtype) for d in dims)
    for x in xyz:
        x -= 0.5
    return np.stack(np.meshgrid(*xyz, indexing='ij'), axis=axis)


def get_ray_eye_coordinates(
        ray_shape, f, z_near, z_far, include_corners=False, axis=-1):
    nx, ny, nz = ray_shape
    x, y, z = (get_grid_points(n, include_corners) for n in ray_shape)

    x -= 0.5
    y -= 0.5

    if isinstance(f, (int, float)) or f.shape == ():
        fx, fy = f, f
    else:
        fx, fy = f

    x *= fx
    y *= fy

    z *= (z_far - z_near)
    z += z_near
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    X *= Z
    Y *= Z
    Z *= -1
    XYZ_eye = np.stack((X, Y, Z), axis=axis)
    return XYZ_eye


def get_ray_interpolation_args(
        R, t, f, z_near, z_far, ray_shape, voxel_shape, include_corners=False):
    voxel_shape = tuple(voxel_shape)
    ray_shape = tuple(ray_shape)
    XYZ_eye = get_ray_eye_coordinates(
        ray_shape, f, z_near, z_far, include_corners)
    XYZ_world = nonhom.coordinate_transform(XYZ_eye, R=R, t=t)
    XYZ_world += 0.5
    XYZ_world *= voxel_shape
    ijk_world = XYZ_world.astype(np.int32)
    inside = np.all((0 <= ijk_world) & (ijk_world < voxel_shape), axis=-1)
    ijk_world[np.logical_not(inside)] = 0
    return ijk_world, inside


def voxel_values_to_frustrum(
        voxel_values, R, t, f, z_near, z_far, ray_shape,
        include_corners=False):
    ijk_world, inside = get_ray_interpolation_args(
        R, t, f, z_near, z_far, ray_shape, voxel_values.shape,
        include_corners=include_corners)
    i, j, k = (
        np.squeeze(ii, axis=-1) for ii in np.split(ijk_world, 3, axis=-1))
    return voxel_values[i, j, k], inside
