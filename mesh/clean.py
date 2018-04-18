"""
Based on pymesh.
git: https://github.com/qnzhou/PyMesh.git

Additional install notes:
apt-get install libptscotch-dev

Documentation: https://media.readthedocs.org/pdf/pymesh/latest/pymesh.pdf
"""
from __future__ import division
import pymesh
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.csgraph import depth_first_order
from scipy.sparse import coo_matrix
from util3d.mesh.geom import get_normals, get_centroids, guarded_normalized, \
    guarded_normalize


def remove_duplicated_faces_raw(faces):
    # return np.array(
    #     tuple(set(((tuple(sorted(f)) for f in faces)))), dtype=np.int32)
    survivors = set(((tuple(sorted(f)) for f in faces)))
    returned = []
    for f in faces:
        fs = tuple(sorted(f))
        if fs in survivors:
            survivors.remove(fs)
            returned.append(f)

    return np.array(returned, dtype=np.int32)


def remove_duplicated_faces(mesh):
    faces = mesh.faces
    faces = remove_duplicated_faces_raw(faces)
    mesh = pymesh.meshio.form_mesh(mesh.vertices, faces)
    return mesh,


def make_face_normals_consistent(mesh):
    from graph import get_face_neighbors
    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    print(
        'Making face normals consistent, nv = %d, nf = %d'
        % (len(vertices), len(faces)))
    print('Calculating normals...')
    normals = get_normals(vertices, faces, normalize=False)
    normalized_normals = guarded_normalized(normals)
    print('Calculating neighbors...')
    neighbors = get_face_neighbors(faces)
    print('Creating adjacency matrix...')
    ii = []
    jj = []
    vv = []
    eps = 1e-4
    for i, n in enumerate(neighbors):
        ni = normalized_normals[i]
        for j in n:
            ii.append(i)
            jj.append(j)
            nj = normalized_normals[j]
            weight = 1 + eps - abs(np.dot(ni, nj))
            vv.append(weight)
    nf = len(faces)

    m = coo_matrix((vv, (ii, jj)), shape=(nf, nf))
    print('Calculating minimum spanning tree...')
    minimum_spanning_tree(m, overwrite=True)
    print('Flipping inconsistent faces...')
    to_flip = []
    n = depth_first_order(m, 0)[0]
    for i in n:
        row = m.getrow(i)
        norm_i = normalized_normals[i]
        js = row.nonzero()[1]
        norm_js = normalized_normals[js]
        dots = np.dot(norm_js, norm_i)
        for j, dot in zip(js, dots):
            if dot < 0:
                normalized_normals[j] *= -1
                to_flip.append(j)
                faces[j] = faces[j, -1::-1]
    faces[to_flip] = faces[to_flip, -1::-1]
    print('Evaluating flux...')
    c = (np.max(vertices, axis=0) + np.min(vertices, axis=0)) / 2
    centroids = get_centroids(vertices, faces) - c
    guarded_normalize(centroids)
    flux = np.sum(centroids * normals)
    if flux < 0:
        print('Flipping all faces')
        faces = faces[:, -1::-1]
    return pymesh.form_mesh(vertices, faces)


def clean(vertices, faces, duplicate_tol=1e-12):
    mesh = pymesh.meshio.form_mesh(vertices, faces)
    mesh = pymesh.remove_isolated_vertices(mesh)[0]
    mesh = pymesh.remove_duplicated_vertices(mesh, tol=duplicate_tol)[0]
    mesh = remove_duplicated_faces(mesh)[0]
    # mesh = pymesh.remove_duplicated_faces(mesh, fins_only=True)[0]
    mesh = pymesh.remove_degenerated_triangles(mesh)[0]
    mesh = pymesh.resolve_self_intersection(mesh)
    # meshes = pymesh.separate_mesh(mesh)
    # for i, mesh in enumerate(meshes):
    #     meshes[i] = make_face_normals_consistent(mesh)
    # mesh = pymesh.merge_meshes(meshes)
    return mesh.vertices, mesh.faces


if __name__ == '__main__':
    from util3d.mesh.obj_io import write_obj

    def get_mesh():
        import numpy as np
        from modelnet.parsed import get_saved_dataset
        dataset = get_saved_dataset('ModelNet40', 'train', 'toilet')
        with dataset as ds:
            key = tuple(ds.keys())[2]
            mesh = ds[key]
            vertices, faces = (
                np.array(mesh[k]) for k in ('vertices', 'faces'))
        return vertices, faces

    vertices, faces = get_mesh()

    v2, f2 = clean(vertices, faces)
    write_obj('/tmp/original.obj', vertices, faces)
    write_obj('/tmp/cleaned.obj', v2, f2)
    print(vertices.shape)
    print(faces.shape)
    print(v2.shape)
    print(f2.shape)

    # def vis(v0, f0, v1, f1):
    #     from mayavi import mlab
    #     from util3d.mayavi_vis import vis_mesh
    #     mlab.figure()
    #     vis_mesh(v0, f0, color=(0, 0, 1))
    #     mlab.figure()
    #     vis_mesh(v1, f1, color=(0, 1, 0))
    #     mlab.show()
    #
    # vis(vertices, faces, v2, f2)
