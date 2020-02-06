"""The Joint class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from robovat.simulation.base import Base


class Joint(Base):
    """Joint of the body."""

    def __init__(self, body, joint_ind):
        """Initialize.

        Args:
            body: The body of the joint.
            joint_ind: The joint index.
        """
        uid = (body.uid, joint_ind)
        name = body.physics.get_joint_name(uid)

        if isinstance(name, bytes):
            name = name.decode('utf-8')

        Base.__init__(self, simulator=body.simulator, name=name)
        self._uid = uid
        self._parent = body
        self._index = joint_ind
        self._has_sensor = False

    @property
    def parent(self):
        return self._parent

    @property
    def index(self):
        return self._index

    @property
    def limit(self):
        return self.physics.get_joint_limit(self.uid)

    @property
    def lower_limit(self):
        return self.physics.get_joint_limit(self.uid)['lower']

    @property
    def upper_limit(self):
        return self.physics.get_joint_limit(self.uid)['upper']

    @property
    def max_effort(self):
        return self.physics.get_joint_limit(self.uid)['effort']

    @property
    def max_velocity(self):
        return self.physics.get_joint_limit(self.uid)['velocity']

    @property
    def range(self):
        return self.upper_limit - self.lower_limit

    @property
    def dynamics(self):
        return self.physics.get_joint_dynamics(self.uid)

    @property
    def damping(self):
        return self.physics.get_joint_dynamics(self.uid)['damping']

    @property
    def friction(self):
        return self.physics.get_joint_dynamics(self.uid)['friction']

    @property
    def position(self):
        return self.physics.get_joint_position(self.uid)

    @property
    def velocity(self):
        return self.physics.get_joint_velocity(self.uid)

    @property
    def reaction_force(self):
        if not self._has_sensor:
            raise ValueError('Joint %s has no sensor enabled.' % (self.name))
        return self.physics.get_joint_reaction_force(self.uid)

    @position.setter
    def position(self, value):
        self.physics.set_joint_position(self.uid, position=value)

    def position_control(self, position, velocity=None, max_force=None,
                         position_gain=None, velocity_gain=None):
        """Position control."""
        if max_force is None:
            max_force = self.limit['effort']

        self.physics.pos_control(
            self.uid,
            position=position,
            velocity=velocity,
            max_force=max_force,
            position_gain=position_gain,
            velocity_gain=velocity_gain)

    def velocity_control(self, velocity, max_force=None,
                         position_gain=None, velocity_gain=None):
        """Velocity control."""
        if max_force is None:
            max_force = self.limit['effort']

        self.physics.vel_control(
            self.uid,
            velocity=velocity,
            max_force=max_force,
            position_gain=position_gain,
            velocity_gain=velocity_gain)

    def torque_control(self, torque):
        """Torque control."""
        self.physics.torque_control(
            self.uid,
            torque=torque)

    def enable_sensor(self):
        """Enable the force/torque sensor of this joint."""
        self._has_sensor = True
        self.physics.enable_joint_sensor(self.uid)
