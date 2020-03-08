"""The class of 3D point.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class Point(np.ndarray):
    """3D point.
    """

    def __new__(cls, value):
        """Initialize the orientation.

        Args:
            value: A 3-dimensional float32 numpy array of [x, y, z].

        Returns:
            A new instance.
        """
        if isinstance(value, (list, tuple)):
            assert len(value) == 3
        elif isinstance(value, np.ndarray):
            assert value.size == 3
        else:
            raise ValueError('Data %r has unrecognized type: %r',
                             value, type(value))

        obj = np.asarray(value).view(cls)

        return obj

    def __str__(self):
        return '[%g, %g, %g]' % (self.x, self.y, self.z)

    @property
    def x(self):
        return self[0]

    @property
    def y(self):
        return self[1]

    @property
    def z(self):
        return self[2]

    @x.setter
    def x(self, value):
        self[0] = value

    @y.setter
    def y(self, value):
        self[1] = value

    @z.setter
    def z(self, value):
        self[2] = value
