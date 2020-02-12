"""File IO utilities using HDF5.

The data element saved in HDF5 should be a dictionary. The types of each value
should be in HDF5_DATA_TYPES.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import h5py
import numpy as np
import uuid
import traceback


def write_data_to_hdf5(f, data, compress_size_thresh=100):
    """Wrte data to HDF5 group.

    Args:
        f: The HDF5 group to write the data to.
        data: The data to be writtent, which can be a group, a dict or a list.
        compress_size_thresh: Data larger or equal to this size will be
            compressed by the gzip format.
    """
    for key, value in data.items():
        if isinstance(value, dict):
            group = f.create_group(key)
            write_data_to_hdf5(group, value)
        elif isinstance(value, list):
            group_list = f.create_group(key + '[]')
            for i, value_i in enumerate(value):
                assert isinstance(value_i, (dict, np.ndarray)), (
                        'List \'%s\' has type %s value %s, which is forbidden.'
                        ' Lists should only have dict or numpy.ndarray values.'
                        % (key, type(value_i), value_i))
                group = group_list.create_group(str(i))
                write_data_to_hdf5(group, value_i)
        else:
            try:
                if value is None:
                    f[key] = 'None'
                else:
                    value = np.array(value)
                    if np.prod(value.shape) >= compress_size_thresh:
                        f.create_dataset(
                            key,
                            data=value,
                            compression="gzip", compression_opts=9)
                    else:
                        f.create_dataset(key, data=value)
            except Exception:
                traceback.print_exc()
                raise ValueError('Unsupported data \'%s\' of type %s.'
                                 % (key, type(value)))


def read_data_from_hdf5(f):
    """Read data from HDF5 group.

    Args:
        f: The HDF5 group to read the data from.

    Returns:
        The data read from the group.
    """
    data = dict()

    for key, value in f.items():
        if isinstance(value, h5py._hl.group.Group):
            if key[-2:] != '[]':
                # Read dictionary.
                data[key] = read_data_from_hdf5(value)
            else:
                # Read list.
                list_var = [None] * len(value)
                for ind, element in value.items():
                    list_var[int(ind)] = read_data_from_hdf5(element)
                data[key[:-2]] = list_var
        else:
            # Read numpy array or scalar.
            value = value.value
            if value == 'None':
                data[key] = None
            else:
                value = np.array(value)
                if value.shape == ():
                    data[key] = np.asscalar(value)
                else:
                    data[key] = value
        # assert isinstance(value, h5py._hl.dataset.Dataset), (
        #         'Item \'%s\' has type %s.' % (key, type(value)))
        # data[key] = np.array(value.value)

    return data


class HDF5Writer(object):
    """A class to dump pickle to file.
    """

    def __init__(self, filename):
        """Initialize.

        Args:
            filename: The filename of the HDF5 file.
        """
        self._file = h5py.File(filename, 'w')

    def write(self, data):
        """Write data to a pickle file.

        Args:
            data: An element of the data.
        """
        assert isinstance(data, dict)

        name = str(uuid.uuid4())
        group = self._file.create_group(name)
        write_data_to_hdf5(group, data)

    def close(self):
        """Close the HDF5 file.
        """
        self._file.close()


def read(filename):
    """Read data from an HDF5 file.

    Args:
        filename: The path to the HDf5 file.

    Yields:
        data: An element of the data.
    """
    with h5py.File(filename, 'r') as f:
        for name, group in f.items():
            try:
                data = read_data_from_hdf5(group)
                yield data
            except Exception:
                raise ValueError('Errors in reading data from the HDF5 file.')
