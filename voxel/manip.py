from __future__ import division
import numpy as np
from scipy.ndimage import zoom
from scipy.ndimage.filters import convolve


def _get_zoom(shape, target_shape):
    if isinstance(target_shape, (int, float)):
        # target_shape = [target_shape for _ in voxels.shape]
        # zoom = tuple((s / t for s, t in zip(voxels.shape, target_shape)))
        s0 = shape[0]
        if all([s == s0 for s in shape[1:]]):
            zoom = s0 / target_shape
        else:
            zoom = tuple(s / target_shape for s in shape)
    elif len(shape) == len(target_shape):
        zoom = tuple((s / t for s, t in zip(shape, target_shape)))
    else:
        raise ValueError('target_shape must be same length as voxels.shape')
    return zoom


def resize(voxels, target_shape):
    """Resize voxels via `scipy.ndimage.zoom`."""
    factor = _get_zoom(voxels.shape, target_shape)
    return zoom(voxels, factor)


def fast_resize(voxels, target_shape):
    """
    Resize voxel grid by sampling every nth voxel.

    Faster than `resize`.

    Args:
        `voxels`: voxel grid, (P, Q, R) array
        `target_shape`: int, or 3-tuple. Each value of `voxel_grid.shape`
            should be divisible by target_shape (if int) or the relevant entry
            (if tuple).

    Returns:
    resized voxel grid.
    """
    zoom = _get_zoom(voxels.shape, target_shape)
    if isinstance(zoom, (int, float)):
        zoom = int(np.round(zoom))
        return voxels[::zoom, ::zoom, ::zoom]
    else:
        zoom = [int(np.round(z)) for z in zoom]
        return voxels[::zoom[0], ::zoom[1], ::zoom[2]]


def get_surface_voxels(voxels):
    """
    Get a voxel grid of just surface voxels.

    A surface voxel is an occupied voxel cell with an unoccupied neighbour.
    """
    weights = np.ones((3, 3, 3), dtype=np.bool)
    weights[1, 1, 1] = False
    empty_space = np.logical_not(voxels)
    n = convolve(empty_space, weights, mode='constant', cval=True)
    return np.logical_and(voxels, n)


def dfs(starts, neighbours_fn):
    """
    Iterable of nodes in a graph defined by `neighbours_fn`.

    All node ids must be hashable.

    Args:
        `starts`: iterable of starting node ids.
        `neighbours_fn`: function with signature `NodeId -> list<NodeId>`

    Returns:
        Iterable of nodes reachable from start in dfs order.
    """
    stack = list(starts)
    visited = set()
    while len(stack) > 0:
        current = stack.pop()
        if current not in visited:
            yield current
            visited.add(current)
            stack.extend(neighbours_fn(current))


def outer_voxels_dfs(voxels):
    """
    Get the voxel grid outside the supplied voxels.

    Calculation is done via depth first search enumeration. While
    computationally more efficient (less operations) than the convolution
    approach (see `outer_voxels_conv`), parallelization is worse. In general,
    use `outer_voxels_conv`.

    Args:
        `voxels`: input voxel grid

    Returns:
        voxel grid, where True indicates the voxel is reachable from the
        outside.
    """
    from itertools import chain
    shape = voxels.shape
    outer = np.zeros(shape, dtype=np.bool)
    si, sj, sk = shape
    starts = chain(
        ((0, j, k) for j in range(sj) for k in range(sk)),
        ((si-1, j, k) for j in range(sj) for k in range(sk)),
        ((i, 0, k) for i in range(si) for k in range(sk)),
        ((i, sj-1, k) for i in range(si) for k in range(sk)),
        ((i, j, 0) for i in range(si) for j in range(sj)),
        ((i, j, sk-1) for i in range(si) for j in range(sj)),
    )

    def is_empty(ijk):
        return not voxels[ijk]

    starts = (n for n in starts if is_empty(n))

    def neighbours_fn_any(current):
        i, j, k = current
        if i > 0:
            yield (i-1, j, k)
        if i < si-1:
            yield (i+1, j, k)
        if j > 0:
            yield (i, j-1, k)
        if j < sj-1:
            yield (i, j+1, k)
        if k > 0:
            yield (i, j, k-1)
        if k < sk - 1:
            yield (i, j, k+1)

    def neighbours_fn(current):
        return (n for n in neighbours_fn_any(current) if is_empty(n))

    for n in dfs(starts, neighbours_fn):
        outer[n] = True
    return outer


def outer_voxels_conv(voxels):
    """
    Get the voxel grid outside the supplied voxels.

    Same as `outer_voxels_dfs`, but calculation is done with repeated
    convolutions. This leads to more operations (O(N^4) rather than O(N^3)),
    but the convolution implementation (O(N^3)) is significantly faster than
    naive loops of the `outer_voxels_dfs` implementation.

    For very large N this -may- be slower than `outer_voxels_dfs`, but
    unlikely.

    Args:
        `voxels`: input voxel grid

    Returns:
        voxel grid, where True indicates the voxel is reachable from the
        outside.
    """
    outer = np.zeros_like(voxels)
    weights = np.zeros((3, 3, 3), dtype=np.bool)
    weights[2, 1, 1] = 1
    weights[0, 1, 1] = 1
    weights[1, 2, 1] = 1
    weights[1, 0, 1] = 1
    weights[1, 1, 2] = 1
    weights[1, 1, 0] = 1
    weights[1, 1, 1] = 1
    n = 0
    while True:
        outer = convolve(outer, weights, mode='constant', cval=1)
        outer = np.logical_and(outer, np.logical_not(voxels))
        new_n = np.sum(outer)
        if new_n == n:
            break
        n = new_n
    return outer


outer_voxels = outer_voxels_conv


def closed_voxels(voxels, thresh=4):
    """
    Simple hole-closing implementation.

    Returns a voxel grid created by filling in voxels with `thresh` occupied
    neighbours in any one plane.
    """
    w0 = np.zeros((3, 3, 3), dtype=np.bool)
    w0[[0, 2, 1, 1], 1, [1, 1, 0, 2]] = 1
    c0 = convolve(voxels, w0, mode='constant', cval=0) == thresh

    w1 = np.zeros((3, 3, 3), dtype=np.bool)
    w1[[0, 2, 1, 1], 1, [1, 1, 0, 2]] = 1
    c1 = convolve(voxels, w1, mode='constant', cval=0) == thresh

    w2 = np.zeros((3, 3, 3), dtype=np.bool)
    w2[[0, 2, 1, 1], [1, 1, 0, 2], 1] = 1
    c2 = convolve(voxels, w2, mode='constant', cval=0) == thresh

    return np.any([voxels, c0, c1, c2], axis=0)


def filled_voxels(voxels):
    """
    Get the filled in voxel grid.

    This is the set of voxels with no free path to the outside, including
    the voxels themselves.
    """
    return np.logical_not(outer_voxels(voxels))


def get_sign_change_indices_1d(data):
    return get_value_change_indices_1d(np.sign(data))


def get_value_change_indices_1d(data):
    return np.where(data[:-1] != data[1:])[0]


def get_root_frac_1d(data, lower_indices):
    lv = data[lower_indices]
    uv = data[lower_indices + 1]
    frac = lv / (lv - uv)
    return frac


def get_interpolated_roots_1d(data):
    lower = get_sign_change_indices_1d(data)
    frac = get_root_frac_1d(data, lower)
    return lower, frac


def get_interpolated_roots_2d(data):
    for di in data:
        yield get_interpolated_roots_1d(di)


# def orthographic_filled_voxels(voxels):
#     """
#     Fill voxels by orthographic projection.
#
#     For each of the x, y and z axes, the min and max voxel depths are
#     calculated for each "pixel" after "orthographic projection", creating a
#     "tube". The returned volume is the intersection of all voxels inside each
#     of these "tubes".
#     """
#     if len(voxels.shape) != 3:
#         raise ValueError('voxels must be rank 3')
#     if voxels.dtype != np.bool:
#         raise ValueError('voxels must be bool')
#
#     voxels = np.pad(
#         voxels,
#         pad_width=((1, 1), (1, 1), (1, 1)),
#         mode='constant',
#         constant_values=0)
#
#     shape = voxels.shape
#
#     def tubed(x, axis):
#         n = shape[axis]
#         indices = range(n)
#         indices = np.array(indices, dtype=np.int32)
#         s0 = [1, 1, 1]
#         s0[axis] = n
#         indices = np.reshape(indices, s0)
#         s0 = list(shape)
#         s0[axis] = 1
#         indices = np.tile(indices, s0)
#         mins = np.min(
#             np.where(x, indices, n*np.ones_like(indices)), axis=axis,
#             keepdims=True)
#         maxs = np.max(
#             np.where(x, indices, -np.ones_like(indices)), axis=axis,
#             keepdims=True)
#         out = (indices >= mins) & (indices <= maxs)
#         return out
#
#     voxels = tubed(voxels, 0) & tubed(voxels, 1) & tubed(voxels, 2)
#     voxels = voxels[1:-1, 1:-1, 1:-1]
#     return voxels


class OrthographicTuber(object):
    def __init__(self, dims, axis):
        n = dims[axis]
        indices = range(n)
        indices = np.array(indices, dtype=np.int32)
        s0 = [1, 1, 1]
        s0[axis] = n
        indices = np.reshape(indices, s0)
        s0 = list(dims)
        s0[axis] = 1
        indices = np.tile(indices, s0)
        self._axis = axis
        self._dims = dims
        self._indices = indices
        self._upper = n * np.ones(shape=dims, dtype=np.int32)
        self._lower = -np.ones(shape=dims, dtype=np.int32)

    def __call__(self, values):
        assert(values.shape == self._dims)
        mins = np.min(
            np.where(values, self._indices, self._upper), axis=self._axis,
            keepdims=True)
        maxs = np.max(
            np.where(values, self._indices, self._lower), axis=self._axis,
            keepdims=True)
        return (self._indices >= mins) & (self._indices <= maxs)


class OrthographicFiller(object):
    def __init__(self, dims):
        self._tubers = [OrthographicTuber(dims, i) for i in range(3)]

    def __call__(self, values):
        return np.all([t(values) for t in self._tubers], axis=0)


def orthographic_filled_voxels(voxels):
    """
    Fill voxels by orthographic projection.

    For each of the x, y and z axes, the min and max voxel depths are
    calculated for each "pixel" after "orthographic projection", creating a
    "tube". The returned volume is the intersection of all voxels inside each
    of these "tubes".
    """
    return OrthographicFiller(voxels.shape)(voxels)
