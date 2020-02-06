"""The Constraint class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from robovat.simulation.base import Base


class Constraint(Base):
    """The constraint between two entities."""

    def __init__(self,
                 parent,
                 child=None,
                 joint_type='fixed',
                 joint_axis=[0, 0, 0],
                 parent_frame_pose=None,
                 child_frame_pose=None,
                 max_force=None,
                 name=None):
        """Initialize.

        Args:
            parent: The parent entity as an instance of Entity.
            child: The child entity as an instance of Entity.
            joint_type: The type of the joint.
            joint_axis: The axis of the joint.
            parent_frame_pose: The pose of the joint in the parent frame.
            child_frame_pose: The pose of the joint in the child frame.
            max_force: Max force the constraint can apply.
        """
        Base.__init__(self, simulator=parent.simulator, name=name)

        parent_uid = parent.uid
        child_uid = child.uid if child is not None else None

        self._uid = self.physics.add_constraint(
                parent_uid,
                child_uid,
                joint_type,
                joint_axis,
                parent_frame_pose,
                child_frame_pose)

        self._parent = parent
        self._child = child
        self._joint_type = joint_type

        if max_force is not None:
            self.max_force = max_force

        if name is None:
            name = '%s_constraint_(%s)_(%s)_%s' % (
                joint_type, parent.name, child.name, self.uid)

    @property
    def parent(self):
        return self._parent

    @property
    def child(self):
        return self._child

    @property
    def joint_type(self):
        return self._joint_type

    @property
    def pose(self):
        return self.physics.get_constraint_pose(self.uid)

    @property
    def position(self):
        return self.physics.get_constraint_position(self.uid)

    @property
    def orientation(self):
        return self.physics.get_constraint_orientation(self.uid)

    @property
    def max_force(self):
        return self.physics.get_constraint_max_force(self.uid)

    @pose.setter
    def pose(self, value):
        self.physics.set_constraint_pose(self.uid, value)

    @position.setter
    def position(self, value):
        self.physics.set_constraint_position(self.uid, value)

    @orientation.setter
    def orientation(self, value):
        self.physics.set_constraint_orientation(self.uid, value)

    @max_force.setter
    def max_force(self, value):
        self.physics.set_constraint_max_force(self.uid, value)

    def update(self):
        """Update disturbances."""
        pass
