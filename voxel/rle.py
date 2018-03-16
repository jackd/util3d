"""
Basic algorithms for byte run length encoding.

This is the main part of the binvox file format.
"""
import numpy as np


def rle_to_dense(rle_data):
    values, counts = rle_data[::2], rle_data[1::2]
    return np.repeat(values.astype(np.bool), counts)


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
                    print(c, count)
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


if __name__ == '__main__':
    rle_data = np.array(
        [0, 5, 1, 3, 0, 255, 0, 2, 1, 255, 1, 3, 0, 2], dtype=np.uint8)
    indices = [0, 2, 5, 6, 10, 17]
    s = np.array(tuple(sorted_gather_1d(rle_data, indices)), dtype=np.bool)
    print(s)
    dense = rle_to_dense(rle_data)
    r2 = np.array(tuple(dense_to_rle(dense)), dtype=np.uint8)
    print(dense[indices])

    print('---')
    print(rle_data)
    print(r2)

    print('---')
    sparse = rle_to_sparse(rle_data)
    print(sparse)
    print(np.where(dense)[0])
    print('***')
    print(sparse)
    r3 = np.array(tuple(sparse_to_rle(sparse, len(dense))), dtype=np.int32)
    print(r2)
    print(r3)
