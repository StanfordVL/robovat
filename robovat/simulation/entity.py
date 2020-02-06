"""The Entity class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

from robovat.simulation.base import Base


class Entity(Base):
    """Entity."""

    __metaclass__ = abc.ABCMeta

    def __init__(self, simulator, name=None):
        """Initialize.

        Args:
            simulator: The simulator of the entity.
            name: The name of the entity.
        """
        Base.__init__(self, simulator, name)

    @property
    def pose(self):
        raise NotImplementedError

    @property
    def position(self):
        return self.pose.position

    @property
    def orientation(self):
        return self.pose.orientation

    @property
    def euler(self):
        return self.orientation.euler

    @property
    def quaternion(self):
        return self.orientation.quaternion

    @property
    def matrix3(self):
        return self.orientation.matrix3

    @property
    def mass(self):
        raise NotImplementedError

    @pose.setter
    def pose(self, value):
        raise NotImplementedError

    @position.setter
    def position(self, value):
        raise NotImplementedError

    @orientation.setter
    def orientation(self, value):
        raise NotImplementedError

    @euler.setter
    def euler(self, value):
        self.orientation = value

    @quaternion.setter
    def quaternion(self, value):
        self.orientation = value

    @matrix3.setter
    def matrix3(self, value):
        self.orientation = value

    @mass.setter
    def mass(self, value):
        raise NotImplementedError
