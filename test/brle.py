#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import numpy as np
from util3d.voxel import brle

dense_simple = np.array((0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0), dtype=np.bool)
brl_simple = np.array((3, 2, 2, 4, 1, 0), dtype=np.uint8)

dense = np.array([0] * 10 + [1] * 20 + [0] * 30 + [1] * 300, dtype=np.bool)
brl = np.array([10, 20, 30, 255, 0, 45], dtype=np.uint8)


class RleTest(unittest.TestCase):
    def test_simple_dense_to_rle(self):
        np.testing.assert_equal(
            np.array(tuple(brle.dense_to_brle(dense_simple))), brl_simple)

    def test_simple_brle_to_dense(self):
        np.testing.assert_equal(
            np.array(tuple(brle.brle_to_dense(brl_simple))), dense_simple)

    def test_dense_to_brle(self):
        np.testing.assert_equal(brle.dense_to_brle(dense), brl)

    def test_brle_to_dense(self):
        np.testing.assert_equal(
            np.array(tuple(brle.brle_to_dense(brl))), dense)

    def test_reduce_sum(self):
        self.assertEqual(brle.reduce_brle_sum(brl), np.sum(dense))

    def test_length(self):
        self.assertEqual(brle.length(brl), len(dense))

    def test_reverse(self):
        reversed = brle.reverse(brl)
        reversed = brle.brle_to_dense(reversed)
        np.testing.assert_equal(reversed, dense[-1::-1])

    def test_zeros(self):
        n = 1000
        expected = np.zeros((n,), dtype=np.bool)
        actual = brle.brle_to_dense(brle.zeros(n))
        np.testing.assert_equal(actual, expected)

    def test_sparse(self):
        actual = brle.brle_to_sparse(brl_simple)
        expected = np.array(np.where(dense_simple)[0], dtype=np.int32)
        np.testing.assert_equal(actual, expected)


unittest.main()
