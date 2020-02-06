"""Define the class of 3D orientation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from copy import deepcopy

import numpy as np

from third_party.transformations import euler_from_matrix3
from third_party.transformations import quaternion_from_matrix3
from robovat.math.euler import Euler
from robovat.math.quaternion import Quaternion


class Orientation(object):
    """3D orientation."""

    def __init__(self, value):
        """Initialize the orientation.

        Args:
            value: The input orientation can be either an instance of
                Orientation, an Euler angle(roll, pitch, yall), a quaternion(x,
                y, z, w), or a 3x3 rotation matrix.
        """
        self._euler = None
        self._quaternion = None
        self._matrix3 = None

        if value is None:
            self._euler = None
            self._quaternion = None
            self._matrix3 = None
        elif isinstance(value, Orientation):
            if value.euler is not None:
                self._euler = value.euler.copy()
            if value.quaternion is not None:
                self._quaternion = value.quaternion.copy()
            if value.matrix3 is not None:
                self._matrix3 = value.matrix3.copy()
        elif isinstance(value, Euler):
            self._euler = value.copy()
        elif isinstance(value, Quaternion):
            self._quaternion = value.copy()
        else:
            value = np.array(value, dtype=np.float32)
            if value.size == 3:
                self._euler = Euler(value)
            elif value.size == 4:
                self._quaternion = Quaternion(value)
            elif value.size == 9:
                self._matrix3 = value.reshape([3, 3])
            else:
                raise ValueError

    def __str__(self):
        return str(self.euler)

    def copy(self):
        """Copy the orientation.

        Returns:
            A deep copy of the orientation.
        """
        return deepcopy(self)

    @property
    def euler(self):
        if self._euler is not None:
            pass
        elif self._quaternion is not None:
            self._euler = self.quaternion.euler
        elif self._matrix3 is not None:
            self._euler = Euler(euler_from_matrix3(self._matrix3))
        else:
            pass
        return self._euler

    @property
    def quaternion(self):
        if self._quaternion is not None:
            pass
        elif self._euler is not None:
            self._quaternion = self.euler.quaternion
        elif self._matrix3 is not None:
            self._quaternion = quaternion_from_matrix3(self._matrix3)
        else:
            pass
        return self._quaternion

    @property
    def matrix3(self):
        if self._matrix3 is not None:
            pass
        elif self._euler is not None:
            self._matrix3 = self.euler.matrix3
        elif self._quaternion is not None:
            self._matrix3 = self.quaternion.matrix3
        else:
            pass
        return self._matrix3

    @euler.setter
    def euler(self, value):
        self._euler = value
        self._quaternion = None
        self._matrix3 = None

    @quaternion.setter
    def quaternion(self, value):
        self._euler = None
        self._quaternion = value
        self._matrix3 = None

    @matrix3.setter
    def matrix3(self, value):
        self._euler = None
        self._quaternion = None
        self._matrix3 = value
