"""The body class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

from robovat.simulation.entity import Entity
from robovat.simulation.joint import Joint
from robovat.simulation.link import Link


class Body(Entity):
    """Body."""

    def __init__(self,
                 simulator,
                 filename,
                 pose,
                 scale=1.0,
                 is_static=False,
                 name=None):
        """Initialize.

        Args:
            simulator: The simulator of the entity.
            filename: Path to the URDF file.
            pose: The initial pose of the body.
            scale: The scaling factor of the body.
            is_static: If the body is static.
            name: The name of the entity.
        """
        Entity.__init__(self, simulator, name)

        self._uid = self.physics.add_body(filename,
                                          pose,
                                          scale=scale,
                                          is_static=is_static)
        self._initial_relative_pose = pose
        self._is_static = is_static

        self._links = [
            Link(self, link_ind)
            for link_ind in self.physics.get_body_link_indices(self.uid)
        ]

        self._joints = [
            Joint(self, joint_ind)
            for joint_ind in self.physics.get_body_joint_indices(self.uid)
        ]

        self._constraints = []

        if name is None:
            model_name, _ = os.path.splitext(os.path.basename(filename))
            self._name = '%s_%s' % (model_name, self.uid)

    @property
    def uid(self):
        return self._uid

    @property
    def links(self):
        return self._links

    @property
    def joints(self):
        return self._joints

    @property
    def pose(self):
        return self.physics.get_body_pose(self.uid)

    @property
    def position(self):
        return self.pose.position

    @property
    def orientation(self):
        return self.pose.orientation

    @property
    def joint_positions(self):
        return [joint.position for joint in self.joints]

    @property
    def joint_velocities(self):
        return [joint.velocity for joint in self.joints]

    @property
    def joint_lower_limits(self):
        return [joint.lower_limit for joint in self.joints]

    @property
    def joint_upper_limits(self):
        return [joint.upper_limit for joint in self.joints]

    @property
    def joint_max_efforts(self):
        return [joint.max_effort for joint in self.joints]

    @property
    def joint_max_velocities(self):
        return [joint.max_velocity for joint in self.joints]

    @property
    def joint_dampings(self):
        return [joint.damping for joint in self.joints]

    @property
    def joint_frictions(self):
        return [joint.friction for joint in self.joints]

    @property
    def joint_ranges(self):
        return [joint.range for joint in self.joints]

    @property
    def linear_velocity(self):
        return self.physics.get_body_linear_velocity(self.uid)

    @property
    def angular_velocity(self):
        return self.physics.get_body_angular_velocity(self.uid)

    @property
    def mass(self):
        if self._mass is None:
            self._mass = self.physics.get_body_mass(self.uid)
        return self._mass

    @property
    def dynamics(self):
        return self.physics.get_body_dynamics(self.uid)

    @property
    def is_static(self):
        return self._is_static

    @property
    def contacts(self):
        return self.physics.get_body_contacts(self.uid)

    @pose.setter
    def pose(self, value):
        self.physics.set_body_pose(self.uid, value)

    @position.setter
    def position(self, value):
        self.physics.set_body_position(self.uid, value)

    @orientation.setter
    def orientation(self, value):
        self.physics.set_body_orientation(self.uid, value)

    @joint_positions.setter
    def joint_positions(self, value):
        for joint, joint_position in zip(self.joints, value):
            joint.position = joint_position

    @linear_velocity.setter
    def linear_velocity(self, value):
        self.physics.set_body_linear_velocity(self.uid, value)

    @angular_velocity.setter
    def angular_velocity(self, value):
        self.physics.set_body_angular_velocity(self.uid, value)

    @mass.setter
    def mass(self, value):
        return self.physics.set_body_mass(self.uid, value)

    def get_joint_by_name(self, name):
        """Get the joint by the joint name.,

        Args:
            name: The joint name.

        Returns:
            An instance of Joint. Return None if the joint is not found.
        """
        for joint in self.joints:
            if joint.name == name:
                return joint

        raise ValueError('The joint %s is not found in body %s.'
                         % (name, self.name))

    def get_link_by_name(self, name):
        """Get the link by the link name.,

        Args:
            name: The link name.

        Returns:
            An instance of Link. Return None if the link is not found.
        """
        for link in self.links:
            if link.name == name:
                return link

        raise ValueError('The link %s is not found in body %s.'
                         % (name, self.name))

    def update(self):
        """Update disturbances."""
        pass

    def set_dynamics(self,
                     mass=None,
                     lateral_friction=None,
                     rolling_friction=None,
                     spinning_friction=None,
                     ):
        """Set dynmamics.

        Args:
            mass: The mass of the body.
            lateral_friction: The lateral friction coefficient.
            rolling_friction: The rolling friction coefficient.
            spinning_friction: The spinning friction coefficient.
        """
        return self.physics.set_body_dynamics(
            self.uid,
            mass=mass,
            lateral_friction=lateral_friction,
            rolling_friction=rolling_friction,
            spinning_friction=rolling_friction,
        )

    def set_color(self, rgba=None, specular=None):
        """Set color.

        Args:
            rgba: The color in RGBA.
            specular: The specular of the object.
        """
        return self.physics.set_body_color(
            self.uid, rgba=rgba, specular=specular)
