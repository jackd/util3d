import os
import binvox as bv
import dids
import dids.auto_save as auto_save


class BinvoxDataset(dids.Dataset):
    def __init__(self, root_dir, mode='r'):
        self._root_dir = root_dir
        self._mode = mode

    @property
    def root_dir(self):
        return self._root_dir

    def path(self, key):
        if not isinstance(key, (str, unicode)):
            raise KeyError('key should be string/unicode, got %s' % str(key))
        path = os.path.join(self._root_dir, '%s.binvox' % key)
        folder = os.path.dirname(path)
        if not os.path.isdir(folder):
            os.makedirs(folder)
        return path

    def __getitem__(self, key):
        with open(self.path(key), 'r') as fp:
            return bv.Voxels.from_file(fp)

    def is_writable(self):
        return self._mode in ('a', 'w')

    def __setitem__(self, key, value):
        if self.is_writable():
            with open(self.path(key), 'w') as fp:
                value.save_to_file(fp)
        else:
            raise RuntimeError('Dataset not writable')

    def __delitem__(self, key):
        if self.is_writable():
            os.remove(self.path(key))
        else:
            raise RuntimeError('Dataset not writable')

    def keys(self):
        root_len = len(self.root_dir) + 1
        for root, dirs, files in os.walk(self._root_dir):
            for f in files:
                if f[-7:] == '.binvox':
                    yield os.path.join(root[root_len:], f[:-7])

    def __len__(self):
        return len(tuple(self.keys()))

    def __contains__(self, key):
        return os.path.isfile(self.path(key))


class BinvoxSavingManager(auto_save.AutoSavingManager):
    def __init__(self, save_dir, dense=True, saving_message=None):
        self._save_dir = save_dir
        self._saving_message = saving_message

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def saving_message(self):
        return self._saving_message

    def get_saving_dataset(self, mode='a'):
        return BinvoxDataset(self.save_dir, dense=self.dense, mode=mode)
