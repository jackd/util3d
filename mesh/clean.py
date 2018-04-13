"""
Based on pymesh.
git: https://github.com/qnzhou/PyMesh.git

Additional install notes:
apt-get install libptscotch-dev

Documentation: https://media.readthedocs.org/pdf/pymesh/latest/pymesh.pdf
"""
import pymesh


def clean(vertices, faces, duplicate_tol=1e-12):
    mesh = pymesh.meshio.form_mesh(vertices, faces)
    mesh = pymesh.remove_isolated_vertices(mesh)[0]
    mesh = pymesh.remove_duplicate_vertices(mesh, tol=duplicate_tol)[0]
    mesh = pymesh.remove_duplicate_faces(mesh)[0]
    mesh = pymesh.remove_degenerate_triangles(mesh)[0]
    mesh = pymesh.resolve_self_intersection(mesh)[0]
    return mesh.vertices, mesh.faces


if __name__ == '__main__':

    def get_mesh():
        from util3d.mesh.obj_io import parse_obj
        return parse_obj('/home/jackd/tmp/airplane_0714.obj')[:2]

    vertices, faces = get_mesh()

    v2, f2 = clean(vertices, faces)
    print(vertices.shape)
    print(faces.shape)
    print(v2.shape)
    print(f2.shape)
