"""
Basic algorithms for byte run length encoding.

This is the main part of the binvox file format.
"""
import numpy as np


def rle_to_dense(rle_data, dtype=None):
    if not isinstance(rle_data, np.ndarray):
        rle_data = np.array(rle_data, dtype=np.uint8)
    values, counts = rle_data[::2], rle_data[1::2]
    if dtype is not None and values.dtype != dtype:
        values = values.astype(dtype)
    return np.repeat(values, counts)


def rle_to_sparse(rle_data):
    indices = []
    it = iter(rle_data)
    index = 0
    try:
        while True:
            value = next(it)
            counts = next(it)
            end = index + counts
            if value == 1:
                indices.append(np.arange(index, end, dtype=np.int32))
            index = end
    except StopIteration:
        pass
    indices = np.concatenate(indices)
    return indices


def dense_to_rle(dense_data):
    data_iter = iter(dense_data)
    try:
        count = 0
        value = next(data_iter)
        while True:
            count += 1
            next_val = next(data_iter)
            if next_val != value or count == 255:
                yield value
                yield count
                count = 0
                value = next_val
    except StopIteration:
        if count > 0:
            yield value
            yield count


def dense_to_rle2(dense_data):
    """
    Based on:
    https://gist.github.com/nvictus/66627b580c13068589957d6ab0919e66
    """
    n = len(dense_data)
    starts = np.r_[0, np.flatnonzero(dense_data[1:] != dense_data[:-1]) + 1]
    lengths = np.diff(np.r_[starts, n])
    values = dense_data[starts]
    nl = len(lengths)
    bad_length_indices = np.where(lengths > 255)
    bad_lengths = lengths[bad_length_indices]
    nl += np.sum(bad_lengths // 255)
    out = np.empty((nl, 2), dtype=np.uint8)
    i = 0
    for (val, length) in zip(values, lengths):
        if length > 255:
            n = length // 255
            out[i:i+n] = val, 255
            length = length % 255
            i += n
        out[i] = val, length
        i += 1
    if i != nl:
        print(i, nl)
    assert(i == nl)
    return out.flatten()


def dense_to_rle_fp(dense_data, fp):
    ctr = 0
    state = dense_data[0]
    for c in dense_data:
        if c == state:
            ctr += 1
            if ctr == 255:
                fp.write(chr(state))
                fp.write(chr(ctr))
                ctr = 0
        else:
            fp.write(chr(state))
            fp.write(chr(ctr))
            state = c
            ctr = 1
    if ctr > 0:
        fp.write(chr(state))
        fp.write(chr(ctr))


def dense_to_rle_with_buffer(dense_data):
    import io
    fp = io.BytesIO()
    dense_to_rle_fp(dense_data, fp)
    fp.seek(0)
    return np.fromstring(fp.read(), dtype=np.uint8)


def _repeated(count):
    while count > 255:
        yield 255
        count -= 255
    if count > 0:
        yield count


def sparse_to_rle(indices, length):
    index_iter = iter(indices)
    try:
        last = next(index_iter)
        if last != 0:
            yield 0
            yield last
        count = 1
        while True:
            n = next(index_iter)
            if n == last + count:
                count += 1
            else:
                for c in _repeated(count):
                    yield 1
                    yield c
                # 0 block
                for c in _repeated(n - last - count):
                    yield 0
                    yield c
                last = n
                count = 1

    except StopIteration:
        for c in _repeated(count):
            yield 1
            yield c
        for c in _repeated(length - n - 1):
            yield 0
            yield c


def sorted_gather_1d(raw_data, ordered_indices):
    data_iter = iter(raw_data)
    index_iter = iter(ordered_indices)
    index = next(index_iter)
    start = 0
    while True:
        while start <= index:
            try:
                value = next(data_iter)
                start += next(data_iter)
            except StopIteration:
                raise IndexError(
                    'Index %d out of range of raw_values length %d'
                    % (index, start))
        try:
            while index < start:
                yield value
                index = next(index_iter)
        except StopIteration:
            break


def gatherer_1d(indices):
    if not isinstance(indices, np.ndarray):
        indices = np.array(indices, copy=False)
    order = np.argsort(indices)
    ordered_indices = indices[order]

    def f(data):
        ans = np.empty(len(order), dtype=np.bool)
        ans[order] = tuple(sorted_gather_1d(data, ordered_indices))
        return ans

    return f


def gather_1d(rle_data, indices):
    return gatherer_1d(indices)(rle_data)


def reverse(rle_data):
    if not isinstance(rle_data, np.ndarray):
        rle_data = np.array(rle_data, copy=False)
    rle_data = np.reshape(rle_data, (-1, 2))
    rle_data = rle_data[-1::-1]
    return np.reshape(rle_data, (-1,))


def _get_contiguous_regions_1d(
        data_iter, max_index, i=0, start_val=0):
    start_index = 0
    vals = []
    try:
        while i < max_index:
            val = next(data_iter)
            n = next(data_iter)
            # print(val, start_val, start_index, i)
            if val != start_val:
                if val == 1:
                    start_index = i
                else:
                    if start_index != i:
                        vals.append((start_index, i))
                    start_index = i
                start_val = val
            i += n
        if start_val == 1:
            vals.append((start_index, max_index))
        done = False
    except StopIteration:
        done = True
    return vals, i, start_val, done


def get_contiguous_regions_1d(rle_data, max_index=np.inf):
    return _get_contiguous_regions_1d(iter(rle_data), max_index)


def get_contiguous_regions_2d(rle_data, dim):
    """
    Get the start/end position of contiguous occupied regions from 2D rle_data.

    Args:
        rel_data: iterable of run-length-encoded values
        dim: last dimension
    Returns:
        Iterable of list of shape (ni, 2), where ni is the number of contiguous
            regions for the ith 1st dimension in rle_data.
    """
    start_val = 0
    start_index = 0
    data_iter = iter(rle_data)
    done = False
    while not done:
        vals, i, start_val, done = _get_contiguous_regions_1d(
            data_iter, dim, start_index, start_val)
        start_index = i - dim
        yield vals


def get_contiguous_regions(rle_data, dims):
    """
    Get the start/end position of contiguous occupied regions from rle_data.

    Args:
        rle_data: iterable of run-length-encoded values
        dims: N-tuple of dimensions
    Returns:
        dims[:-1] + (n, 2) jagged array, where n is a jagged index.
    """
    if len(dims) != 3:
        raise NotImplementedError('Could be generalized, but currently not.')
    u, v, w = dims
    uv = u*v
    ret = np.empty((uv,), dtype=np.object)

    for i, vals in enumerate(get_contiguous_regions_2d(rle_data, w)):
        if len(vals) > 0:
            ret[i] = np.array(vals)
        else:
            ret[i] = np.zeros((0, 2), dtype=np.int32)
    for j in range(i, uv):
        ret[j] = np.zeros((0, 2), dtype=np.int32)
    return ret.reshape((u, v))


def sample_occupied_indices(rle_data, n_samples):
    import random
    if n_samples > 0:
        s = reduce_rle_sum(rle_data)
        ns = random.sample(range(s), n_samples)
        ns.sort()
        rle_index = 0
        count = 0
        rle_iter = iter(rle_data)
        for n in ns:
            while n >= count:
                value = next(rle_iter)
                c = next(rle_iter)
                if value:
                    count += c
                rle_index += c
            diff = count - n
            yield rle_index - diff


def value_length_pairs(rle_data):
    """Get an iterable of (value, length) pairs."""
    it = iter(rle_data)
    try:
        while True:
            yield next(it), next(it)
    except StopIteration:
        pass


def length(rle_data):
    return sum(l for v, l in value_length_pairs(rle_data))


def reduce_rle_sum(rle_data):
    return sum(v*l for v, l in value_length_pairs(rle_data))


if __name__ == '__main__':
    # rle_data = np.array(
    #     [0, 5, 1, 3, 0, 255, 0, 2, 1, 255, 1, 3, 0, 2], dtype=np.uint8)
    # indices = [0, 2, 5, 6, 10, 17]
    # s = np.array(tuple(sorted_gather_1d(rle_data, indices)), dtype=np.bool)
    # print(s)
    # dense = rle_to_dense(rle_data)
    # r2 = np.array(tuple(dense_to_rle(dense)), dtype=np.uint8)
    # print(dense[indices])
    #
    # print('---')
    # print(rle_data)
    # print(r2)
    #
    # print('---')
    # sparse = rle_to_sparse(rle_data)
    # print(sparse)
    # print(np.where(dense)[0])
    # print('***')
    # print(sparse)
    # r3 = np.array(tuple(sparse_to_rle(sparse, len(dense))), dtype=np.int32)
    # print(r2)
    # print(r3)

    # data = [0, 2, 1, 3, 0, 4, 1, 5, 0, 10]
    # dim = 20
    # vals, s, v, done = _get_contiguous_regions_1d(iter(data), dim)
    # print(vals, s, v)
    # dims = (4, 4, 4)
    # rle_data = [0, 2, 1, 1, 0, 2, 1, 3, 0, 10, 1, 5]
    # print(rle_data)
    # r = tuple(get_contiguous_regions_2d(rle_data, 6))
    # print(r)
    # r = get_contiguous_regions(rle_data, dims)
    # print(r)
    # print(r[0][0])

    rle_data = np.array([0, 5, 1, 3, 0, 2, 1, 4, 0, 2], dtype=np.uint8)
    dense = rle_to_dense(rle_data)
    print(reduce_rle_sum(rle_data))
    print(np.sum(dense))
    print(dense)
    print(np.array(tuple(sample_occupied_indices(rle_data, 5))))
