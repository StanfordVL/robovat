"""3D orientation as Euler angles.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from third_party.transformations import quaternion_from_euler
from third_party.transformations import matrix3_from_euler


class Euler(np.ndarray):
    """Euler angles.
    """

    def __new__(cls, value):
        """Initialize the orientation.

        Args:
            value: A 3-dimensional float32 numpy array of [roll, pitch, yaw].

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

        obj = np.asarray(value).reshape(3,).view(cls)

        return obj

    def __str__(self):
        return '[%g, %g, %g]' % (self.roll, self.pitch, self.yaw)

    @property
    def euler(self):
        return self

    @property
    def quaternion(self):
        return quaternion_from_euler(self.roll, self.pitch, self.yaw)

    @property
    def matrix3(self):
        return matrix3_from_euler(self.roll, self.pitch, self.yaw)

    @property
    def roll(self):
        return self[0]

    @property
    def pitch(self):
        return self[1]

    @property
    def yaw(self):
        return self[2]

    @roll.setter
    def roll(self, value):
        self[0] = value

    @pitch.setter
    def pitch(self, value):
        self[1] = value

    @yaw.setter
    def yaw(self, value):
        self[2] = value
