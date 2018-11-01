"""
Binary run-length encoding (BRLE) utilities.

Binary run length encodings are similar to run-length encodings (RLE) but
encode lists of values that can only be true or false. As such, the RLE can be
effectively halved by not saving the value of each run.

e.g.
[0, 0, 1, 1, 1, 0, 0, 1, 0]
would have a BRLE of
[2, 3, 2, 1, 1]

For simplicity, we require all BRLEs to be of even length. If a sequence ends
in a False, it must be padded with an additional zero.

Like RLEs the encoding is stored as uint8s. This means runs of length greater
than 255 must be stored with intermediate zeros, e.g. 300 zeros would be
represented by
[255, 0, 45]

First value is START_VALUE == False
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
START_VALUE = False
_ft = np.array([False, True], dtype=np.bool)
_2550 = np.array([255, 0], dtype=np.uint8)


def brle_to_dense(brle_data):
    if not isinstance(brle_data, np.ndarray):
        brle_data = np.array(brle_data, dtype=np.uint8)
    ft = np.repeat(_ft[np.newaxis, :], len(brle_data) // 2, axis=0).flatten()
    return np.repeat(ft, brle_data).flatten()


def zeros(length):
    data = np.empty((length // 255 + 1, 2), dtype=np.uint8)
    data[:-1, 0] = 255
    data[-1, 0] = length % 255
    data[:, 1] = 0
    return data.flatten()


def ones(length):
    data = np.empty((length // 255 + 1, 2), dtype=np.uint8)
    data[:-1, 1] = 255
    data[-1, 1] = length % 255
    data[:, 0] = 0
    return data.flatten()


def _empty_padded(n):
    if n % 2 == 0:
        # return np.zeros((n,), dtype=np.uint8)
        return np.empty((n,), dtype=np.uint8)
    else:
        # x = np.zeros((n+1,), dtype=np.uint8)
        x = np.empty((n+1,), dtype=np.uint8)
        x[-1] = 0
        return x


def dense_to_brle(dense_data):
    if dense_data.dtype != np.bool:
        raise ValueError('dense_data must be bool')
    n = len(dense_data)
    starts = np.r_[0, np.flatnonzero(dense_data[1:] != dense_data[:-1]) + 1]
    lengths = np.diff(np.r_[starts, n])
    return start_and_lengths_to_brle(dense_data[0], lengths)


def start_and_lengths_to_brle(start_value, lengths):
    nl = len(lengths)
    bad_length_indices = np.where(lengths > 255)
    bad_lengths = lengths[bad_length_indices]
    nl += 2*np.sum(bad_lengths // 255)
    if start_value:
        i = 1
        out = _empty_padded(nl+1)
        out[0] = 0
    else:
        i = 0
        out = _empty_padded(nl)

    for length in lengths:
        nf = length // 255
        nr = length % 255
        out[i:i + 2*nf] = np.repeat(_2550, nf)
        i += 2*nf
        out[i] = nr
        i += 1
    return out


def rle_to_brle(rle_data):
    from .rle import split_rle_data
    values, counts = split_rle_data(rle_data)
    repeated = values[1:] == values[:-1]
    out_length = len(values) + np.count_nonzero(repeated)
    v0 = values[0]
    if v0 == 1:
        out_length += 1
        out = _empty_padded(out_length)
        out[0] = 0
        i = 1
    else:
        i = 0
        out = _empty_padded(out_length)
    out[i] = counts[0]
    i += 1
    for count, rep in zip(counts[1:], repeated):
        if rep:
            out[i:i+2] = (0, count)
            i += 2
        else:
            out[i] = count
            i += 1
    assert(i == out_length)
    # start = 0
    # end = 10
    # print(rle_data[start:end])
    # print(out[start:end])
    # exit()
    return out


def brle_to_rle(brle_data):
    data = np.repeat(_ft, np.reshape(brle_data, (-1, 2)))
    # remove empty runs
    data = data[data[:, 1] > 0]
    return data.flatten()


def length(brle_data):
    return np.sum(brle_data)


def reduce_brle_sum(brle_data):
    return np.sum(brle_data[1::2])


def brle_to_sparse(brle_data, dtype=np.int32):
    cs = np.cumsum(brle_data)
    starts = cs[::2]
    ends = cs[1::2]
    return np.concatenate(
        [np.arange(s, e, dtype=dtype) for s, e in zip(starts, ends)])


def reverse(brle_data):
    if brle_data[-1] == 0:
        brle_data = brle_data[-1::-1]
    else:
        brle_data = np.r_[0, brle_data[-1::-1]]
    if len(brle_data) % 2 != 0:
        brle_data = np.r_[brle_data, 0]
    return brle_data


def sorted_gather_1d(raw_data, ordered_indices):
    data_iter = iter(raw_data)
    index_iter = iter(ordered_indices)
    index = next(index_iter)
    start = 0
    value = START_VALUE
    while True:
        while start <= index:
            try:
                value = not value
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


def sparse_to_brle(indices, length):
    from . import rle
    return rle_to_brle(rle.rle_to_sparse(indices, length))


def pad_to_length(brle_data, length):
    from . import rle
    return rle.pad_to_length(brle_data, length)


def remove_length_padding(rle_data):
    data = np.reshape(rle_data, (-1, 2))
    data = data[data != [0, 0]]
    return data.flatten()
