"""Define the class of 3D Pose.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from copy import deepcopy

import numpy as np

from robovat.math.point import Point
from robovat.math.orientation import Orientation


class Pose(object):
    """3D Pose."""

    def __init__(self, value=[[0, 0, 0], [0, 0, 0]]):
        """Initialize the pose.

        Args:
            value: The input pose can be an instance of Pose, a tuple/list of
                position and orientation, or a transform matrix as a [4, 4]
                numpy array.
        """
        if isinstance(value, np.ndarray) and value.size == 16:
            self.matrix4 = value.reshape((4, 4))
        else:
            self.position = value[0]
            self.orientation = value[1]

    def __str__(self):
        return '[position: %g, %g, %g, euler: %g, %g, %g]' % (
                self.x, self.y, self.z,
                (self.euler[0] + np.pi) % (2 * np.pi) - np.pi,
                (self.euler[1] + 0.5 * np.pi) % np.pi - 0.5 * np.pi,
                (self.euler[2] + np.pi) % (2 * np.pi) - np.pi)

    def __getitem__(self, index):
        if index == 0:
            return self.position
        elif index == 1:
            return self.orientation
        else:
            raise ValueError(
                'The index of a Pose instance can only be 0 or 1.')

    @property
    def pose(self):
        return self

    @property
    def position(self):
        return self._position

    @property
    def x(self):
        return self._position.x

    @property
    def y(self):
        return self._position.y

    @property
    def z(self):
        return self._position.z

    @property
    def orientation(self):
        return self._orientation

    @property
    def euler(self):
        return self._orientation.euler

    @property
    def roll(self):
        return self.euler[0]

    @property
    def pitch(self):
        return self.euler[1]

    @property
    def yaw(self):
        return self.euler[2]

    @property
    def quaternion(self):
        return self._orientation.quaternion

    @property
    def matrix3(self):
        return self._orientation.matrix3

    @property
    def matrix4(self):
        matrix4 = np.eye(4)
        matrix4[:3, :3] = self.matrix3
        matrix4[:3, 3] = self.position
        return matrix4

    @position.setter
    def position(self, value):
        self._position = Point(value)

    @x.setter
    def x(self, value):
        self._position.x = value

    @y.setter
    def y(self, value):
        self._position.y = value

    @z.setter
    def z(self, value):
        self._position.z = value

    @orientation.setter
    def orientation(self, value):
        self._orientation = Orientation(value)

    @euler.setter
    def euler(self, value):
        # When setting euler, orientation should be updated.
        self._orientation = Orientation(value)

    @roll.setter
    def roll(self, value):
        # When setting euler, orientation should be updated.
        self._orientation = Orientation([value, self.pitch, self.yaw])

    @pitch.setter
    def pitch(self, value):
        # When setting euler, orientation should be updated.
        self._orientation = Orientation([self.roll, value, self.yaw])

    @yaw.setter
    def yaw(self, value):
        # When setting euler, orientation should be updated.
        self._orientation = Orientation([self.roll, self.pitch, value])

    @quaternion.setter
    def quaternion(self, value):
        # When setting quaternion, orientation should be updated.
        self._orientation = Orientation(value)

    @matrix3.setter
    def matrix3(self, value):
        # When setting matrix3, orientation should be updated.
        self._orientation = Orientation(value)

    @matrix4.setter
    def matrix4(self, value):
        self._position = Point(value[:3, 3])
        self._euler = Orientation(value[:3, :3]).euler
        self._quaternion = Orientation(value[:3, :3]).quaternion
        self._matrix3 = Orientation(value[:3, :3]).matrix3

    def inverse(self):
        """Get the inverse rigid transformation of the pose.

        If the pose is defined in the world frame, then the return value s the
        world origin in the frame of this pose.

        Returns:
            An instance of Pose.
        """
        position = np.dot(-self.position, self.matrix3)
        orientation = self.matrix3.T
        return Pose([position, orientation])

    def transform(self, pose):
        """Rigid transformation from one from to another.

        The rigid transformation from the source frame to the target frame is
        defined by self.position and self.orientation.

        Args:
            pose: The pose defined in the source frame

        Returns:
            The corresponding pose defined in the target frame.
        """
        pose = Pose(pose)
        position = self.position + np.dot(pose.position, self.matrix3.T)
        orientation = np.dot(self.matrix3, pose.matrix3)
        return Pose([position, orientation])

    def copy(self):
        """Copy the pose.

        Returns:
            A deep copy of the pose.
        """
        return deepcopy(self)

    def to_array(self):
        """Convert to numpy array.

        Returns:
            A numpy array of [2, 3].
        """
        pose = np.r_[self.position, self.euler]
        return np.array(pose, dtype=np.float)

    @staticmethod
    def uniform(x,
                y,
                z,
                roll=0.0,
                pitch=0.0,
                yaw=0.0):
        """Draw pose samples from a uniform distribution.

        Args:
            x: Value/range of the x position.
            y: Value/range of the y position.
            z: Value/range of the z position.
            roll: Value/range of the roll position.
            pitch: Value/range of the pitch position.
            yaw: Value/range of the yaw position.

        Returns:
            A Pose instance.
        """
        if isinstance(x, (list, tuple)):
            x = np.random.uniform(x[0], x[1])

        if isinstance(y, (list, tuple)):
            y = np.random.uniform(y[0], y[1])

        if isinstance(z, (list, tuple)):
            z = np.random.uniform(z[0], z[1])

        if isinstance(roll, (list, tuple)):
            roll = np.random.uniform(roll[0], roll[1])

        if isinstance(pitch, (list, tuple)):
            pitch = np.random.uniform(pitch[0], pitch[1])

        if isinstance(yaw, (list, tuple)):
            yaw = np.random.uniform(yaw[0], yaw[1])

        return Pose([[x, y, z], [roll, pitch, yaw]])


def get_transform(source=None, target=None):
    """
    Get rigid transformation from one frame to another.

    Args:
        source: The source frame. Set to None if it is the world frame.
        target: The target frame. Set to None if it is the world frame.

    Returns:
        An instance of Pose.
    """
    if source is not None and not isinstance(source, Pose):
        source = Pose(source)

    if target is not None and not isinstance(target, Pose):
        target = Pose(target)

    if source is not None and target is not None:
        orientation = np.dot(target.matrix3.T, source.matrix3)
        position = np.dot(source.position - target.position, target.matrix3)
    elif source is not None:
        orientation = source.matrix3
        position = source.position
    elif target is not None:
        orientation = target.matrix3.T
        position = np.dot(-target.position, target.matrix3)
    else:
        orientation = np.eye(3, dtype=np.float32)
        position = np.ones(3, dtype=np.float32)

    return Pose([position, orientation])
