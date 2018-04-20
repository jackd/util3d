"""
Code for calculating the semi-convex hull of an object.

Implementation of Algorithm 1 from Guney et al.
http://www.cvlibs.net/publications/Guney2015CVPR.pdf

Will likely use PyACVD for remeshing: https://pypi.org/project/PyACVD/
"""
# import tensorflow as tf
# from sample import sample_faces


def get_convex_hull(points):
    from scipy.spatial import ConvexHull
    from geom import triangulated_faces
    hull = ConvexHull(points)
    faces = hull.simplices
    faces = tuple(triangulated_faces(faces))
    return faces


# class SemiConvexHullApproximator(object):
#     def __enter__(self):
#         self.sess = tf.Session()
#
#     def __exit__(self, *args, **kwargs):
#         self.sess.close()
#
#
# def get_semi_convex_hull(vertices, faces, n_points=4096):
#     p = sample_faces(vertices, faces, n_points)
#     cf = get_convex_hull(vertices)
#     cv, cf = remesh(vertices, faces)
#     converged = False
#
#     while not converged:
#         grad_norm, nn = sess.run()
