import numpy as np


def sample_points(x, n_samples, axis=0, replace=True):
    """Sample `n_samples` points from cloud `x` with or without replacement."""
    n_original = x.shape[axis]
    indices = np.random.choice(n_original, n_samples, replace=replace)
    # indices = np.random.uniform(
    #     0, n_original, size=(n_samples)).astype(np.int32)
    return x.take(indices, axis=axis)
