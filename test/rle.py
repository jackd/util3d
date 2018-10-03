#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import unittest
import numpy as np
from util3d.voxel import rle

dense_simple = np.array((0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0), dtype=np.bool)
rl_simple = np.array((0, 3, 1, 2, 0, 2, 1, 4, 0, 1), dtype=np.uint8)

dense = np.array([0] * 10 + [1] * 20 + [0] * 30 + [1] * 300, dtype=np.bool)
rl = np.array([0, 10, 1, 20, 0, 30, 1, 255, 1, 45], dtype=np.uint8)


class RleTest(unittest.TestCase):
    def test_simple_dense_to_rle(self):
        np.testing.assert_equal(
            np.array(tuple(rle.dense_to_rle(dense_simple))), rl_simple)

    def test_simple_rle_to_dense(self):
        np.testing.assert_equal(
            np.array(tuple(rle.rle_to_dense(rl_simple))), dense_simple)

    def test_dense_to_rle(self):
        np.testing.assert_equal(np.array(tuple(rle.dense_to_rle(dense))), rl)

    def test_rle_to_dense(self):
        np.testing.assert_equal(np.array(tuple(rle.rle_to_dense(rl))), dense)

    def test_reduce_sum(self):
        self.assertEqual(rle.reduce_rle_sum(rl), np.sum(dense))

    def test_length(self):
        self.assertEqual(rle.length(rl), len(dense))

    def test_sorted_gather_1d(self):
        n = len(dense)
        ns = 10
        indices = random.sample(range(n), ns)
        indices.sort()
        np.testing.assert_equal(
            np.array(tuple(rle.sorted_gather_1d(rl, indices))), dense[indices])

    def test_gather_1d(self):
        n = len(dense)
        ns = 10
        indices = random.sample(range(n), ns)
        np.testing.assert_equal(
            np.array(tuple(rle.gather_1d(rl, indices))), dense[indices])

    def test_reverse(self):
        reversed = rle.reverse(rl)
        reversed = rle.rle_to_dense(reversed)
        np.testing.assert_equal(reversed, dense[-1::-1])

    def test_gatherer(self):
        n = len(dense)
        ns = 10
        indices = random.sample(range(n), ns)
        gatherer = rle.gatherer_1d(indices)

        np.testing.assert_equal(gatherer(rl), dense[indices])
        np.testing.assert_equal(
            gatherer(rle.reverse(rl)), dense[-1::-1][indices])

    def test_dense_to_rle2(self):
        rl2 = rle.dense_to_rle2(dense)
        np.testing.assert_equal(rl, rl2)

    def test_zeros(self):
        n = 1000
        expected = np.zeros((n,), dtype=np.bool)
        actual = rle.rle_to_dense(rle.zeros(n))
        np.testing.assert_equal(actual, expected)


unittest.main()
