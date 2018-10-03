from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from . import manip


class _OrthographicTuber(manip.OrthographicTuber):
    def __init__(self, dims, axis):
        super(_OrthographicTuber, self).__init__(dims, axis)
        with tf.name_scope('tuber%d_setup' % axis):
            self._indices = tf.constant(self._indices)
            self._upper = tf.constant(self._upper)
            self._lower = tf.constant(self._lower)

    def __call__(self, values):
        shape = values.shape
        if hasattr(shape, 'as_list'):
            shape = tuple(values.shape.as_list())
        assert(shape == self._dims)
        with tf.name_scope('tuper%d_call' % self._axis):
            mins = tf.reduce_min(
                tf.where(values, self._indices, self._upper),
                axis=self._axis, keepdims=True)
            maxs = tf.reduce_max(
                tf.where(values, self._indices, self._lower),
                axis=self._axis, keepdims=True)
            return tf.logical_and(self._indices >= mins, self._indices <= maxs)


def orthographic_filled_voxels(voxels):
    shape = voxels.shape
    if hasattr(shape, 'as_list'):
        shape = tuple(shape.as_list())
    with tf.name_scope('orthographic_filled_voxels'):
        x, y, z = (_OrthographicTuber(shape, i)(voxels) for i in range(3))
        out = tf.logical_and(tf.logical_and(x, y), z)
    return out
