import numpy as np
import _binvox_rw as binvox_rw


class Voxels(object):
    def __init__(self, translate=[0, 0, 0], scale=1, axis_order='xzy'):
        self.translate = translate
        self.scale = scale
        self.axis_order = axis_order

    def _kwargs(self):
        return dict(
            translate=self.translate,
            scale=self.scale,
            axis_order=self.axis_oder)

    @property
    def dims(self):
        raise NotImplementedError('Abstract property')

    def dense_data(self):
        raise NotImplementedError('Abstract method')

    def sparse_data(self):
        raise NotImplementedError('Abstract method')

    def to_dense(self):
        return DenseVoxels(self.dense_data(), **self._kwargs())

    def to_sparse(self):
        return SparseVoxels(self.sparse_data(), self.dims, **self._kwargs())

    @staticmethod
    def dense(dense_array, *args, **kwargs):
        return DenseVoxels(dense_array, *args, **kwargs)

    @staticmethod
    def sparse(sparse_array, dims, *args, **kwargs):
        return SparseVoxels(sparse_array, dims, *args, **kwargs)

    def write_to_file(self, fp):
        if self.axis_order not in ('xzy', 'xyz'):
            raise ValueError('Unsupported voxel model axis order')
        dense_data = self.dense_data()

        if self.axis_order == 'xyz':
            dense_data = np.transpose(dense_data, (0, 2, 1))
        flat_data = dense_data.flatten()
        _write_voxel_data(
            fp, flat_data, self.dims, translate=self.translate,
            scale=self.scale)

    def write(self, path_or_file):
        if isinstance(path_or_file, (str, unicode)):
            with open(path_or_file, 'w') as fp:
                self.write_to_file(fp)


class SparseVoxels(Voxels):
    def __init__(self, sparse_data, dims, *args, **kwargs):
        if len(sparse_data.shape) != 2:
            raise ValueError(
                'sparse_data must be a 2d array, got shape %s' %
                str(sparse_data.shape))
        if sparse_data.shape[1] != 3:
            raise ValueError(
                'sparse_data.shape[1] must be 3, got %s'
                % str(sparse_data.shape))
        self._sparse_data = sparse_data
        self._dims = dims
        super(SparseVoxels, self).__init__(*args, **kwargs)

    @property
    def dims(self):
        return self._dims

    def dense_data(self):
        return binvox_rw.sparse_to_dense(self._sparse_data, self.dims)

    def sparse_data(self):
        return self._sparse_data

    @staticmethod
    def from_file(fp, fix_coords=False):
        base = binvox_rw.read_as_coord_array(fp, fix_coords=fix_coords)
        return SparseVoxels(
            base.data, base.dims, base.translate, base.scale, base.axis_order)

    @staticmethod
    def from_path(path, fix_coords=False):
        with open(path, 'r') as fp:
            return SparseVoxels.from_file(fp, fix_coords)


class DenseVoxels(Voxels):
    def __init__(self, dense_data, *args, **kwargs):
        if len(dense_data.shape) != 3:
            raise ValueError(
                'dense_data must be a 3d array, got shape %s' %
                str(dense_data.shape))
        self._dense_data = dense_data
        super(DenseVoxels, self).__init__(*args, **kwargs)

    @property
    def dims(self):
        return self._dense_data.shape

    def dense_data(self):
        return self._dense_data

    def sparse_data(self):
        return binvox_rw.dense_to_sparse(self._dense_data)

    @staticmethod
    def from_file(fp, fix_coords=False):
        base = binvox_rw.read_as_3d_array(fp, fix_coords=fix_coords)
        return DenseVoxels(
            base.data, base.translate, base.scale, base.axis_order)

    @staticmethod
    def from_path(path, fix_coords=False):
        with open(path, 'r') as fp:
            return DenseVoxels.from_file(fp, fix_coords)


def _write_voxel_data(fp, flat_data, dims, translate=[0, 0, 0], scale=1):
    """
    Write voxel data to file.

    flat_data must be flattened and in xzy order.

    Refactoring of on binvox_rw.write_voxel_data
    """
    fp.write('#binvox 1\n')
    fp.write('dim ' + ' '.join(map(str, dims)) + '\n')
    fp.write('translate ' + ' '.join(map(str, translate)) + '\n')
    fp.write('scale ' + str(scale) + '\n')
    fp.write('data\n')

    # keep a sort of state machine for writing run length encoding
    state = flat_data[0]
    ctr = 0
    for c in flat_data:
        if c == state:
            ctr += 1
            # if ctr hits max, dump
            if ctr == 255:
                fp.write(chr(state))
                fp.write(chr(ctr))
                ctr = 0
        else:
            # if switch state, dump
            fp.write(chr(state))
            fp.write(chr(ctr))
            state = c
            ctr = 1
    # flush out remainders
    if ctr > 0:
        fp.write(chr(state))
        fp.write(chr(ctr))
