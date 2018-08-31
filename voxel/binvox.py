from __future__ import division
import numpy as np
import rle
import os


def read_header(fp):
    """
    Read binvox header.

    Mostly meant for internal use.
    """
    line = fp.readline().strip()
    if not line.startswith(b'#binvox'):
        raise IOError('Not a binvox file')
    dims = tuple(int(s) for s in fp.readline().strip().split(b' ')[1:])
    translate = tuple(float(s) for s in fp.readline().strip().split(b' ')[1:])
    scale = float(fp.readline().strip().split(b' ')[1])
    fp.readline()
    return dims, translate, scale


class Voxels(object):
    def __init__(self, dims, translate=(0, 0, 0), scale=1):
        if isinstance(dims, int):
            self._dims = (dims,) * 3
        elif len(dims) != 3:
            raise ValueError('dims must have 3 elements.')
        else:
            self._dims = tuple(dims)
        self.translate = np.array(translate)
        self.scale = scale

    @staticmethod
    def from_file(fp):
        """Deprecated. Use Voxels.from_binvox instead."""
        return Voxels.from_binvox_file(fp)

    @staticmethod
    def from_binvox_file(fp):
        dims, translate, scale = read_header(fp)
        rle_data = np.frombuffer(fp.read(), dtype=np.uint8)
        return RleVoxels(rle_data, dims, translate, scale)

    @staticmethod
    def from_binvox_path(path):
        with open(path, 'r') as fp:
            return Voxels.from_binvox_path(fp)

    @staticmethod
    def from_binvox(path_or_file):
        if hasattr(path_or_file, 'read'):
            return Voxels.from_binvox_file(path_or_file)
        elif isinstance(path_or_file, (str, unicode)):
            return Voxels.from_binvox_path(path_or_file)
        else:
            raise TypeError(
                'path_or_file must file-like (have a `read` attribute) '
                'or be a string/uincode.')

    @staticmethod
    def from_path(path):
        ext = os.path.splitext(path)[1]
        if ext == '.binvox':
            return Voxels.from_binvox(path)
        elif ext == '.npy':
            return Voxels.from_numpy(path)
        else:
            raise NotImplementedError('Unrecognized extension "%s"' % ext)

    @staticmethod
    def from_numpy(path_or_file):
        return DenseVoxels(np.load(path_or_file))

    @staticmethod
    def load(path_or_file):
        if hasattr(path_or_file, 'read'):
            return Voxels.from_file(path_or_file)
        elif isinstance(path_or_file, (str, unicode)):
            return Voxels.from_path(path_or_file)
        else:
            raise TypeError(
                'path_or_file must file-like (have a `read` attribute) '
                'or be a string/uincode.')

    def save(self, path):
        with open(path, 'w') as fp:
            self.save_to_file(fp)

    def save_to_file(self, fp):
        dims = self.dims
        translate = self.translate
        scale = self.scale
        fp.write('#binvox 1\n')
        fp.write('dim ' + ' '.join(map(str, dims)) + '\n')
        fp.write('translate ' + ' '.join(map(str, translate)) + '\n')
        fp.write('scale ' + str(scale) + '\n')
        fp.write('data\n')
        # fp.write((chr(d) for d in self.rle_data()))
        fp.write(self.rle_data().tostring())

    @property
    def dims(self):
        return self._dims

    def to_dense(self):
        return DenseVoxels(
            self.dense_data(), self.dims, self.translate, self.scale)

    def to_sparse(self):
        return SparseVoxels(
            self.sparse_data(), self.dims, self.translate, self.scale)

    def to_rle(self):
        return RleVoxels(self.rle_data(), self.dims)

    def rle_data(self):
        raise NotImplementedError('Abstract method')

    def dense_data(self, fix_coords=False):
        raise NotImplementedError('Abstract method')

    def sparse_data(self, fix_coords=False):
        raise NotImplementedError('Abstract method')

    def gather(self, indices, fix_coords=False):
        raise NotImplementedError('Abstract method')


class RleVoxels(Voxels):
    def __init__(self, rle_data, dims, translate=(0, 0, 0), scale=1):
        self._rle_data = rle_data
        super(RleVoxels, self).__init__(dims, translate, scale)

    def rle_data(self):
        return self._rle_data

    def dense_data(self, fix_coords=False):
        rle_data = self._rle_data
        data = rle.rle_to_dense(rle_data)
        assert(data.dtype == np.bool)
        data = data.reshape(self.dims)
        if fix_coords:
            data = np.transpose(data, (0, 2, 1))
        return data

    def sparse_data(self, fix_coords=False):
        indices = rle.rle_to_sparse(self._rle_data)
        dims = self.dims
        d2 = dims[2]
        d1 = dims[1]*d2
        i = indices // d1
        kj = indices % d1
        k = kj // d2
        j = kj % d2
        if fix_coords:
            return i, k, j
        else:
            return i, j, k

    def gather(self, indices, fix_coords=False):
        if fix_coords:
            x, y, z = indices
            indices = x, z, y
        indices = np.ravel_multi_index(indices, self.dims)
        order = np.argsort(indices)
        ordered_indices = indices[order]
        ans = np.empty(len(order), dtype=np.bool)
        ans[order] = tuple(self._sorted_gather(ordered_indices))
        return ans

    def _sorted_gather(self, ordered_indices):
        return rle.sorted_gather_1d(self._rle_data, ordered_indices)


class DenseVoxels(Voxels):
    def __init__(self, dense_data, translate=(0, 0, 0), scale=1):
        self._dense_data = dense_data
        super(DenseVoxels, self).__init__(dense_data.shape, translate, scale)

    def rle_data(self):
        return np.array(tuple(
            rle.dense_to_rle(self._dense_data.flatten())), dtype=np.uint8)

    def dense_data(self, fix_coords=False):
        return self._dense_data

    def sparse_data(self, fix_coords=False):
        i, k, j = np.where(self._dense_data)
        if fix_coords:
            return i, j, k
        else:
            return i, k, j

    def gather(self, indices, fix_coords=False):
        if fix_coords:
            i, j, k = indices
        else:
            i, k, j = indices
        return self._dense_data[i, k, j]


class SparseVoxels(Voxels):
    def __init__(self, sparse_data, dims, translate=(0, 0, 0), scale=1):
        self._sparse_data = sparse_data
        super(SparseVoxels, self).__init__(dims, translate, scale)

    def rle_data(self):
        i, k, j = self._sparse_data
        indices = np.ravel_multi_index((i, k, j), self.dims)
        return rle.sparse_to_rle(indices, np.prod(self.dims))

    def dense_data(self, fix_coords=False):
        dims = self.dims
        if fix_coords:
            dims = dims[0], dims[2], dims[1]
            i, k, j = self._sparse_data
        else:
            i, j, k = self._sparse_data
        data = np.zeros(dims, dtype=np.bool)
        data[i, j, k] = True
        return data

    def sparse_data(self, fix_coords=False):
        i, k, j = self._sparse_data
        if fix_coords:
            return i, j, k
        else:
            return i, k, j

    def gather(self, indices, fix_coords=False):
        if fix_coords:
            i, j, k = indices
        else:
            i, k, j = indices
        dims = self.dims
        indices_1d = np.ravel_multi_index((i, k, j), dims)
        sparse_1d = set(np.ravel_multi_index(self._sparse_data, dims))
        return np.array([i1d in sparse_1d for i1d in indices_1d], np.bool)


if __name__ == '__main__':
    dims = (32,) * 3
    dense_data = np.random.randn(*dims) > 0
    dense = DenseVoxels(dense_data)
    rl = dense.to_rle()
    sparse = dense.to_sparse()
    ds_data = dense.sparse_data()
    print(np.all(dense_data == sparse.dense_data()))
    print(np.all(dense_data == rl.dense_data()))
    for expected, actual in zip(sparse.sparse_data(), ds_data):
        print(np.all(expected == actual))
