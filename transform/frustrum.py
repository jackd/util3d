from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from . import nonhom


def _expand_f(f):
    if isinstance(f, (int, float)) or f.shape == ():
        fx, fy = f, f
    else:
        fx, fy = f
    return fx, fy


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
    x, y, z = (get_grid_points(n, include_corners) for n in ray_shape)

    x -= 0.5
    y -= 0.5

    fx, fy = _expand_f(f)

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


def get_voxel_coordinates(
        voxel_shape, include_corners=False, dtype=np.float32):
    x, y, z = (get_grid_points(n, include_corners, dtype) for n in voxel_shape)
    xyz = np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1)
    xyz -= 0.5
    return xyz


def frustrum_to_eye_coordinates(xyz_frustrum, f, z_near, z_far):
    x, y, z = (np.squeeze(ii, axis=-1).copy()
               for ii in np.split(xyz_frustrum, 3, axis=-1))
    y -= 0.5
    x -= 0.5
    z *= (z_far - z_near)
    z *= z_near

    fx, fy = _expand_f(f)
    x *= fx
    y *= fy
    z *= -1

    return np.stack((x, y, z), axis=-1)


def eye_to_frustrum_coordinates(xyz_eye, f, z_near, z_far):
    x, y, z = (np.squeeze(ii, axis=-1).copy()
               for ii in np.split(xyz_eye, 3, axis=-1))

    z *= -1
    x /= z
    y /= z

    fx, fy = _expand_f(f)

    x /= fx
    y /= fy

    z -= z_near
    z /= z_far - z_near
    x += 0.5
    y += 0.5

    xyz_frust = np.stack((x, y, z), axis=-1)
    return xyz_frust


def get_frustrum_coordinates(
        R, t, f, z_near, z_far, voxel_shape, include_corners=False):
    """Get frustrum coordinates of voxel grid points."""
    xyz_world = get_voxel_coordinates(
        voxel_shape, include_corners=include_corners)
    xyz_eye = nonhom.inverse_coordinate_transform(x=xyz_world, R=R, t=t)
    return eye_to_frustrum_coordinates(xyz_eye, f, z_near, z_far)


def get_frustrum_interpolation_args(
        R, t, f, z_near, z_far, ray_shape, voxel_shape, include_corners=False):
    xyz_frust = get_frustrum_coordinates(
        R, t, f, z_near, z_far, voxel_shape, include_corners=include_corners)

    inside = np.logical_and(
        np.all(xyz_frust >= 0, axis=-1),
        np.all(xyz_frust <= 1))
    xyz_frust *= tuple(ray_shape)
    ijk_frust = xyz_frust.astype(np.int32)
    ijk_frust[np.logical_not(inside)] = 0
    return ijk_frust, inside


_methods = ['nearest', 'linear']


def _fix_outside(out, out_val=0):
    outside = np.isnan(out)
    inside = np.logical_not(outside)
    out[outside] = out_val
    return out, inside


def frustrum_values_to_voxels(
        frustrum_values, R, t, f, z_near, z_far, voxel_shape,
        include_corners=False, interpolation_order=0):
    from scipy.interpolate import RegularGridInterpolator
    # ijk_frust, inside = get_frustrum_interpolation_args(
    #     R, t, f, z_near, z_far, frustrum_values.shape, voxel_shape,
    #     include_corners=include_corners)
    # i, j, k = (
    #     np.squeeze(ii, axis=-1) for ii in np.split(ijk_frust, 3, axis=-1))
    # return frustrum_values[i, j, k], inside
    xyz_frust = get_frustrum_coordinates(
        R, t, f, z_near, z_far, voxel_shape, include_corners=include_corners)
    ray_shape = frustrum_values.shape
    xf, yf, zf = (get_grid_points(s, include_corners) for s in ray_shape)
    method = _methods[interpolation_order]
    # if interpolation_order == 1:
    #     frustrum_values = frustrum_values.astype(np.float32)
    interpolator = RegularGridInterpolator(
        (xf, yf, zf), frustrum_values, method, bounds_error=False)
    out = interpolator(xyz_frust)
    return _fix_outside(out)


def voxel_values_to_frustrum(
        voxel_values, R, t, f, z_near, z_far, ray_shape,
        include_corners=False, interpolation_order=0):
    # ijk_world, inside = get_ray_interpolation_args(
    #     R, t, f, z_near, z_far, ray_shape, voxel_values.shape,
    #     include_corners=include_corners)
    #
    # i, j, k = (
    #     np.squeeze(ii, axis=-1) for ii in np.split(ijk_world, 3, axis=-1))
    # return voxel_values[i, j, k], inside
    from scipy.interpolate import RegularGridInterpolator

    xyz_eye = get_ray_eye_coordinates(
        ray_shape, f, z_near, z_far, include_corners)
    xyz_world = nonhom.coordinate_transform(xyz_eye, R=R, t=t)

    xw, yw, zw = (
        get_grid_points(s, include_corners) - 0.5 for s in voxel_values.shape)
    method = _methods[interpolation_order]
    interpolator = RegularGridInterpolator(
        (xw, yw, zw), voxel_values, method, bounds_error=False)
    out = interpolator(xyz_world)
    return _fix_outside(out)
