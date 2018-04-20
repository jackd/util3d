import os
from mayavi import mlab
from util3d.mayavi_vis import vis_mesh
from util3d.mesh.off import OffObject
from util3d.mesh.geom import triangulated_faces
from util3d.mesh.sch import get_convex_hull

folder = os.path.realpath(os.path.dirname(__file__))
bunny_path = os.path.join(folder, 'data', 'bunny.off')
bunny = OffObject.from_path(bunny_path)

vertices = bunny.vertices
faces = tuple(triangulated_faces(bunny.faces))

mlab.figure()
vis_mesh(vertices, faces)

cf = get_convex_hull(vertices)
# mlab.figure()
vis_mesh(vertices, cf, color=(1, 0, 0), opacity=0.1)
mlab.show()
