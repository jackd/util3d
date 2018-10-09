"""Homogeneous transforms."""
import numpy as np


def coordinate_transform(x, T):
    return np.matmul(T, x)


def inverse_coordinate_transform(x, T):
    return np.linalg.solve(T, x)


def transform_transform(T, T1):
    """
    Transform associated with applying T1 to T.

    Applied to x, this is the same as
    coordinate_transform(coordinate_transform(x, T), T1)
    """
    return np.matmul(T1, T)
