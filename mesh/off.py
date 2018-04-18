import numpy as np


def _parse_off_vertex(line):
    return tuple(float(s) for s in line.rstrip().split(' '))


def _parse_off_face(line):
    return tuple(int(s) for s in line.rstrip().split(' ')[1:])


class OffObject(object):
    def __init__(self, vertices, faces):
        self._n_edges = None
        self._vertices = np.array(vertices)
        self._faces = faces

    @property
    def vertices(self):
        return self._vertices

    @property
    def faces(self):
        return self._faces

    @property
    def n_edges(self):
        if self._n_edges is None:
            self._n_edges = sum(len(f) for f in self.faces)
        return self._n_edges

    @property
    def n_vertices(self):
        return len(self.vertices)

    @property
    def n_faces(self):
        return len(self.faces)

    @staticmethod
    def from_file(fp):
        line_iter = iter(fp.readlines())
        try:
            line = next(line_iter)
            while line.startswith('#'):
                line = next(line_iter)
                continue
            if line[:3] != 'OFF':
                raise IOError('Invalid .off file: must start with OFF')
            line = line[3:]
            if line == '\n':
                line = next(line_iter)
        except StopIteration:
            raise IOError('Invalid off file - no header found.')

        nv, nf, ne = (int(i) for i in line.rstrip().split(' '))
        try:
            vertices = tuple(
                _parse_off_vertex(next(line_iter)) for _ in range(nv))
            faces = tuple(
                _parse_off_face(next(line_iter)) for _ in range(nf))
        except StopIteration:
            raise IOError('Invalid off file - insufficient number of lines.')
        try:
            next(line_iter)
            raise IOError('Invalid off file - too many lines')
        except StopIteration:
            return OffObject(vertices, faces)

    def _to_file(self, fp):
        fp.write('OFF\n')
        fp.write('%d %d %d\n' % (self.n_vertices, self.n_faces, self.n_edges))
        fp.writelines(
            '%s\n' % ' '.join(str(vi) for vi in v) for v in self.vertices)
        fp.writelines(
            '%d %s' % (len(f), ' '.join(str(fi) for fi in f))
            for f in self.faces)

    def to_file(self, path_or_file):
        if hasattr(path_or_file, 'write'):
            self._to_file(path_or_file)
        else:
            with open(path_or_file, 'w') as fp:
                self._to_file(fp)

    @staticmethod
    def from_path(path):
        with open(path, 'r') as fp:
            return OffObject.from_file(fp)
