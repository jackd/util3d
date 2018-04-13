import numpy as np


def sample_triangle(v, n=None):
    if n is None:
        size = v.shape[:-2] + (2,)
    elif isinstance(n, int):
        size = (n, 2)
    elif isinstance(n, tuple):
        size = n + (2,)
    elif isinstance(n, list):
        size = tuple(n) + (2,)
    else:
        raise TypeError('n must be int, tuple or list, got %s' % str(n))
    assert(v.shape[-2] == 2)
    a = np.random.uniform(size=size)
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


def sample_faces_with_normals(vertices, faces, n_total):
        if len(faces) == 0:
            raise ValueError('Cannot sample points from zero faces.')
        tris = vertices[faces]
        d0 = tris[..., 0:1, :]
        ds = tris[..., 1:, :] - d0
        d0 = np.squeeze(d0, axis=-2)
        assert(ds.shape[1:] == (2, 3))
        normals = np.cross(ds[:, 0], ds[:, 1])
        norm = np.sqrt(np.sum(normals**2, axis=-1, keepdims=True))
        areas = np.squeeze(norm, axis=-1)
        total_area = np.sum(areas)*(1 + 1e-4)
        areas /= total_area
        norm_eps = 1e-8
        norm[norm < norm_eps] = norm_eps
        normals /= norm

        counts = np.random.multinomial(n_total, areas)
        indices = np.concatenate(
            tuple((i,)*c for i, c in enumerate(counts)),
            axis=0).astype(np.int32)
        positions = d0[indices] + sample_triangle(ds[indices])
        normals = normals[indices]
        return positions, normals


if __name__ == '__main__':

    def vis_mesh(vertices, faces):
        from mayavi import mlab
        from util3d.mayavi_vis import vis_mesh
        from util3d.mayavi_vis import vis_normals
        from util3d.mesh.graph import make_cloud_normals_consistent
        # from util3d.mesh.geom import get_centroids, get_normals
        # from geom import get_normals
        # centroids = get_centroids(vertices, faces)
        # normals = get_normals(vertices, faces)
        # original_normals = normals.copy()

        positions, normals = sample_faces_with_normals(
            vertices, faces, 2048)
        original_normals = normals.copy()
        thresh = (np.max(positions) - np.min(positions)) * 0.2
        print('making normals consistent...')
        nc, cc = make_cloud_normals_consistent(
            positions, normals, thresh=thresh)
        # face_normals = get_normals(vertices, faces)
        # faces = faces[face_normals[:, 2] > 0]
        for norms in (original_normals, normals):
            mlab.figure()
            vis_mesh(vertices, faces, color=(0, 0, 1), opacity=0.2)
            vis_normals(
                positions, norms, scale_factor=0.000005, color=(0, 1, 0))
        mlab.show()

    def mesh_dataset():
        # from shapenet.core.meshes import get_mesh_dataset
        # from shapenet.core import cat_desc_to_id
        # cat_id = cat_desc_to_id('plane')
        # return get_mesh_dataset(cat_id)
        from modelnet.parsed import get_saved_dataset
        return get_saved_dataset('ModelNet40', 'train', 'airplane')

    def get_mesh():
        from util3d.mesh.obj_io import parse_obj
        return parse_obj('/home/jackd/tmp/airplane_0714.obj')[:2]

    vertices, faces = get_mesh()
    vis_mesh(vertices, faces)

    # with mesh_dataset() as ds:
    #     for example_id in ds:
    #         mesh = ds[example_id]
    #         vertices, faces = (
    #             np.array(mesh[k]) for k in ('vertices', 'faces'))
    #
    #         vis(vertices, faces)
