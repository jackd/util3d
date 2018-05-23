import os
from mayavi import mlab
from util3d.mayavi_vis import vis_voxels
from util3d.mesh.off import OffObject
from util3d.mesh.geom import triangulated_faces
from util3d.voxel.convert import mesh_to_voxels
from util3d.voxel.binvox import Voxels

folder = os.path.realpath(os.path.dirname(__file__))
vox_path = os.path.join(folder, 'data', 'bunny.binvox')

if not os.path.isfile(vox_path):
    bunny_path = os.path.join(folder, 'data', 'bunny.off')
    bunny = OffObject.from_path(bunny_path)

    vertices = bunny.vertices
    faces = tuple(triangulated_faces(bunny.faces))
    voxels = mesh_to_voxels(vertices, faces, 64)

    with open(os.path.join(folder, 'data', 'bunny.binvox'), 'w') as fp:
        voxels.save_to_file(fp)
    del voxels

with open(vox_path, 'r') as fp:
    voxels = Voxels.from_file(fp)


vis_voxels(voxels.dense_data())
mlab.show()
