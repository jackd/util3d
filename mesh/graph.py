import numpy as np
from itertools import chain
from scipy.sparse.csgraph import connected_components
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.csgraph import depth_first_order
from scipy.sparse import coo_matrix


def _sorted(a, b):
    return (a, b) if a < b else (b, a)


def edges(face):
    for i in range(len(face) - 1):
        yield face[i], face[i+1]
    yield face[-1], face[0]


def get_face_neighbors(faces):
    face_neighbors = tuple(
        tuple(_sorted(*e) for e in edges(f)) for f in faces)
    edge_neighbors = {}
    for i, fn in enumerate(face_neighbors):
        for ab in fn:
            edge_neighbors.setdefault(ab, set()).add(i)
    # print(edge_neighbors)
    fn2 = tuple(set(chain(*(edge_neighbors[ab] for ab in fn)))
                for fn in face_neighbors)
    for i, fn in enumerate(fn2):
        fn.remove(i)
    return fn2


def find_close_vertices(vertices, thresh=1e-8):
    v0 = np.expand_dims(vertices, axis=-2)
    v1 = np.expand_dims(vertices, axis=-3)
    dists = np.sum((v0 - v1)**2, axis=-1)
    ii, jj = np.where(dists <= thresh)
    nv = len(vertices)
    m = coo_matrix((np.ones((len(ii),), dtype=np.bool), (ii, jj)), (nv, nv))
    nc, cc = connected_components(m)
    return nc, cc


def merge_close_vertices(vertices, faces, thresh=1e-8):
    nc, cc = find_close_vertices(vertices, thresh=thresh)
    components = [[] for _ in range(nc)]
    for v, c in zip(vertices, cc):
        components[c].append(v)

    ret_vertices = np.array(
        tuple(np.mean(comp, axis=0) for comp in components))
    ret_faces = cc[faces]
    return ret_vertices, ret_faces


def make_face_normals_consistent(faces, normals):
    neighbors = get_face_neighbors(faces)
    ii = []
    jj = []
    vv = []
    eps = 1e-4
    for i, n in enumerate(neighbors):
        ni = normals[i]
        for j in n:
            ii.append(i)
            jj.append(j)
            nj = normals[j]
            weight = 1 + eps - abs(np.dot(ni, nj))
            vv.append(weight)
    nf = len(faces)

    m = coo_matrix((vv, (ii, jj)), shape=(nf, nf))
    return _make_all_normals_consistent(m, normals)


def _make_all_normals_consistent(m, normals):
    minimum_spanning_tree(m, overwrite=True)
    ncc, cc = connected_components(m)
    done = set()
    for i, ci in enumerate(cc):
        if ci not in done:
            _make_normals_consistent(m, normals, i)
            done.add(ci)
    return ncc, cc


def _make_normals_consistent(m, normals, start_index):
    n, _ = depth_first_order(m, start_index)
    for i in n:
        row = m.getrow(i)
        norm_i = normals[i]
        for j in row.nonzero()[1]:
            norm_j = normals[j]
            dot = np.dot(norm_i, norm_j)
            if dot < 0:
                # print('Flipping %d - %d' % (i, j))
                # norm_j *= -1
                normals[j] = -norm_j


def get_close_points(points, thresh=1e-8):
    from scipy.spatial import cKDTree
    return cKDTree(points).query_pairs(thresh)


def make_cloud_normals_consistent(points, normals, thresh=1e-3):
    close_points = get_close_points(points, thresh=thresh)

    ii, jj = zip(*close_points)
    print(len(ii))
    # iii = np.array(ii + jj)
    # jjj = np.array(jj + ii)
    iii = np.array(ii)
    jjj = np.array(jj)
    weights = 1 + 1e-3 - np.abs(np.sum(normals[iii]*normals[jjj], axis=-1))
    n = len(points)
    m = coo_matrix((weights, (iii, jjj)), (n, n))
    return _make_all_normals_consistent(m, normals)


if __name__ == '__main__':
    from geom import get_normals, get_centroids
    # vertices = np.array([
    #     [0, 0, 0],
    #     [0, 1, 0],
    #     [0, 1, 1],
    #     [0, 0, 1],
    #     [0, -1, 1],
    #     [1, -1, -2]
    # ], dtype=np.float32)
    # faces = np.array(
    #     ((0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 5, 4)), dtype=np.int32)

    def get_mesh():
        from util3d.mesh.obj_io import parse_obj
        return parse_obj('/home/jackd/tmp/airplane_0714.obj')[:2]

    vertices, faces = get_mesh()
    r = np.max(vertices) - np.min(vertices)
    print(r)
    vertices, faces = merge_close_vertices(vertices, faces, r*1e-5)
    # ii, jj = find_close_vertices(vertices)
    # print(len(vertices))
    # print(len(ii))
    # exit()

    colors = (
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 0),
        (0, 1, 1),
        (1, 0, 0),
        (1, 0, 1),
        (1, 1, 0),
        (1, 1, 1),
    )

    def vis(vertices, faces, split_faces, centroids, normals,
            original_normals):
        from mayavi import mlab
        from util3d.mayavi_vis import vis_mesh, vis_normals
        for n in (normals, normals_original):
            mlab.figure()
            for i, f in enumerate(split_faces):
                vis_mesh(vertices, f, color=colors[i % len(colors)])
            vis_normals(centroids, n, scale_factor=10)
        mlab.show()

    centroids = get_centroids(vertices, faces)
    normals = get_normals(vertices, faces, normalize=True)
    normals_original = normals.copy()
    nc, cc = make_face_normals_consistent(faces, normals)
    split_faces = [[] for _ in range(nc)]
    for ci, face in zip(cc, faces):
        split_faces[ci].append(face)
    # print(get_face_neighbors(faces))

    areas = []
    split_faces = tuple(np.array(f) for f in split_faces)
    for f in split_faces:
        print(f.shape)
        n = get_normals(vertices, f, normalize=False)
        area = np.sum(np.sqrt(np.sum(n**2, axis=-1))) / 2
        areas.append(area)

    vis(vertices, faces, split_faces, centroids, normals, normals_original)
