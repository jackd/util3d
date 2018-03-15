"""Provides EdgeSplitter class and convenience function split_edges."""
from __future__ import division
import numpy as np
import sortedcontainers
from collections import defaultdict


def _edges(face):
    i, j, k = face
    return ((i, j), (j, k), (k, i))


def _split_face(indices, start, n):
    f0 = list(indices)
    f1 = list(indices)
    f0[start] = n
    f1[(start + 1) % 3] = n
    return tuple(f0), tuple(f1)


def _sorted_edge(i0, i1):
    return (i0, i1) if i0 < i1 else (i1, i0)


def has_unique_vertices(face):
    a, b, c = face
    return a != b and a != c and b != c


class EdgeSplitter(object):
    """
    Class for splitting edges until all lengths are all below a threshold.

    Edges are split in descending order according to length.
    """

    def __init__(
            self, vertices, faces, vertex_map_fn=None, remove_bad_faces=False):
        self._vertices = list(vertices)
        self._edge_lengths = {}
        self._edge_faces = defaultdict(dict)
        self._edges = sortedcontainers.SortedListWithKey(
            key=lambda x: self._edge_lengths[x])
        self._faces = set()
        self._vertex_map_fn = vertex_map_fn
        if remove_bad_faces:
            faces = (f for f in faces if has_unique_vertices(f))
        for face in faces:
            self.add_face(tuple(face))

    @property
    def faces(self):
        return tuple(self._faces)

    @property
    def vertices(self):
        return tuple(self._vertices)

    def add_face(self, face):
        u, v, w = face
        if u == v or u == w or v == w:
            print('Warning: line face. Ignoring')
            return
        for i, edge in enumerate(_edges(face)):
            self._add_edge(edge)
            self._edge_faces[_sorted_edge(*edge)][face] = i
        self._faces.add(face)

    def add_vertex(self, vertex):
        if self._vertex_map_fn is not None:
            vertex = self._vertex_map_fn(vertex)
        self._vertices.append(vertex)

    def split_edge(self, edge):
        n = len(self._vertices)
        i0, i1 = edge
        if i0 > i1:
            i0, i1 = i1, i0
            edge = (i0, i1)
        self.add_vertex((self._vertices[i0] + self._vertices[i1])/2)
        for face, i in self._edge_faces[edge].items():
            self._split_face(face, i, n)
        self._remove_edge(edge)

    def split_longest_edge(self):
        self.split_edge(self._edges[-1])

    def split_to_threshold(self, threshold):
        if threshold <= 0:
            raise ValueError('threshold must be positive.')
        if len(self._edges) == 0:
            return
        max_l2 = threshold**2
        while self._edge_lengths[self._edges[-1]] >= max_l2:
            self.split_longest_edge()

    def split_to_n_vertices(self, n_vertices):
        while len(self._vertices) < n_vertices:
            self.split_longest_edge()

    def _add_edge(self, edge):
        edge = _sorted_edge(*edge)
        i0, i1 = edge
        if edge in self._edge_lengths:
            return
        l2 = np.sum((self._vertices[i0] - self._vertices[i1])**2)
        self._edge_lengths[edge] = l2
        self._edges.add(edge)

    def _remove_edge(self, edge):
        i0, i1 = edge
        assert(i0 < i1)
        self._edges.remove(edge)
        del self._edge_lengths[edge]

    def _remove_face(self, face):
        self._faces.remove(face)
        for edge in _edges(face):
            faces = self._edge_faces[_sorted_edge(*edge)]
            del faces[face]

    def _split_face(self, face, start, n):
        self._remove_face(face)
        f0, f1 = _split_face(face, start, n)
        self.add_face(f0)
        self.add_face(f1)


def split_to_threshold(
        vertices, faces, edge_length_threshold, vertex_map_fn=None):
    splitter = EdgeSplitter(vertices, faces, vertex_map_fn)
    splitter.split_to_threshold(edge_length_threshold)
    return splitter.vertices, splitter.faces


def split_to_n_vertices(vertices, faces, n_vertices, vertex_map_fn=None):
    splitter = EdgeSplitter(vertices, faces, vertex_map_fn)
    splitter.split_to_n_vertices(n_vertices)
    return splitter.vertices, splitter.faces
