"""File IO utilities using pickle.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from robovat.utils.logging import logger

from robovat.utils import time_utils

try:
    import cPickle as pickle
except Exception as e:
    logger.warn(str(e))
    import pickle


class PickleWriter(object):
    """Write trajectories as TFRecord format."""

    def __init__(self,
                 output_dir,
                 num_entries_per_file,
                 use_random_name=True):
        """Initialize.

        Args:
            output_dir: Output directory.
            num_entries_per_file: Number of entries in each file.
            use_random_name: Use randomly generated name if True.
        """
        self._output_dir = output_dir
        self._num_entries_per_file = num_entries_per_file
        self._use_random_name = use_random_name

        self._file = None
        self._output_path = None
        self._num_files = 0
        self._num_entries_this_file = 0

        if not os.path.isdir(output_dir):
            logger.info('Making output directory %s...', output_dir)
            os.makedirs(output_dir)

    def __call__(self, data):
        """Call function.

        Args:
            data: The input data.
        """
        # Create a file for saving the episode data.
        if self._num_entries_this_file == 0:
            if self._use_random_name:
                timestamp = time_utils.get_timestamp_as_string()
            else:
                timestamp = '%06d' % (self._num_files)
            filename = 'data_%s.pickle' % (timestamp)
            self._output_path = os.path.join(self._output_dir, filename)
            self._num_files += 1
            if self._file:
                self._file.close()

            self._file = open(self._output_path, 'wb')

        # Append the episode to the file.
        logger.info('Saving data to file %s (%d / %d)...',
                    self._output_path,
                    self._num_entries_this_file,
                    self._num_entries_per_file)
        num_entries = self.write(data)

        # Update the cursor.
        self._num_entries_this_file += num_entries
        self._num_entries_this_file %= self._num_entries_per_file

    def write(self, data):
        """Write a string record to the file.

        Args:
            data: The input data.
        """
        pickle.dump(data, self._file, protocol=pickle.HIGHEST_PROTOCOL)
        return 1

    def close(self):
        """Close the pickle file.
        """
        if self._file is not None:
            self._file.close()


def read(filename):
    """Read data from a pickle file.

    Args:
        filename: The filename to the pickle file.

    Yields:
        data: An element of the data.
    """
    with open(filename, 'rb') as f:
        while True:
            try:
                data = pickle.load(f)
                yield data
            except EOFError:
                break


def read_all(filename):
    """Read all data from a pickle file.

    Args:
        filename: The filename to the pickle file.

    Return:
        data_list: List of elements.
    """
    data_list = []

    with open(filename, 'rb') as f:
        while True:
            try:
                data = pickle.load(f)
                data_list.append(data)
            except EOFError:
                break

    return data_list
