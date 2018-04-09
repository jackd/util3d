import numpy as np


def sample_triangle(v, n, dtype=np.float32):
    a = np.random.uniform(size=(n, 2)).astype(dtype)
    mask = np.sum(a, axis=-1) > 1
    a[mask] *= -1
    a[mask] += 1
    a = np.expand_dims(a, axis=-1)
    return np.sum(a*v, axis=-2)


def sample_faces(vertices, faces, n_total):
    if len(faces) == 0:
        raise ValueError('Cannot sample points from zero faces.')
    tris = vertices[faces]
    n_faces = len(faces)
    d0 = tris[..., 0:1, :]
    ds = tris[..., 1:, :] - d0
    assert(ds.shape[1:] == (2, 3))
    areas = 0.5 * np.sqrt(np.sum(np.cross(ds[:, 0], ds[:, 1])**2, axis=-1))
    cum_area = np.cumsum(areas)
    cum_area *= (n_total / cum_area[-1])
    cum_area = np.round(cum_area).astype(np.int32)

    positions = []
    last = 0
    for i in range(n_faces):
        n = cum_area[i] - last
        last = cum_area[i]
        if n > 0:
            positions.append(d0[i] + sample_triangle(ds[i], n))
    return np.concatenate(positions, axis=0)
