import numpy as np


def sample_points(x, n_samples, axis=0):
    """Sample `n_samples` points from cloud `x` with replacement."""
    n_original = x.shape[axis]
    indices = np.random.uniform(
        0, n_original, size=(n_samples)).astype(np.int32)
    return x.take(indices, axis=axis)
