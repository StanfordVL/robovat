"""The controllable constraint class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from robovat.math.pose import Pose
from robovat.simulation.constraint import Constraint
from robovat.utils.logging import logger


# These default constants.
NUM_STEPS_CHECK = 100
TIMEOUT = 1.0
POSITION_THRESHOLD = 0.01
EULER_THRESHOLD = np.pi / 36


class ControllableConstraint(Constraint):
    """Controllable constraint."""

    def __init__(self,
                 parent,
                 child,
                 joint_type='fixed',
                 joint_axis=[0, 0, 0],
                 parent_frame_pose=None,
                 child_frame_pose=None,
                 max_linear_velocity=None, max_angular_velocity=None,
                 max_force=None,
                 name=None):
        """Initialize.

        parent: The parent of the constraint.
        child: The child of the constraint.
        joint_type: Type of the joint.
        joint_axis: The rotation axis of the joint.
        parent_frame_pose: The joint pose in the parent frame.
        child_frame_pose: The joint pose in the child frame.
        max_linear_velocity: The maximum linear velocity of the joint.
        max_augular_velocity: The maximum angular velocity of the joint.
        max_force: The maximum force of the joint.
        name: Name of the joint.
        """
        super(ControllableConstraint, self).__init__(
            parent=parent,
            child=child,
            joint_type=joint_type,
            joint_axis=joint_axis,
            parent_frame_pose=parent_frame_pose,
            child_frame_pose=child_frame_pose,
            max_force=max_force,
            name=name)

        self._max_linear_velocity = max_linear_velocity
        self._max_angular_velocity = max_angular_velocity

        self.reset_targets()

    def reset_targets(self):
        """Reset the control variables.
        """
        self._target_pose = None
        self._target_linear_velocity = None
        self._target_angular_velocity = None
        self._start_time = None
        self._stop_time = None

    def is_ready(self):
        """Check if the constraint is ready.

        Returns:
            True if all control commands are done, False otherwise.
        """
        return self._target_pose is None

    def set_target_pose(self,
                        pose,
                        linear_velocity=None,
                        angular_velocity=None,
                        timeout=TIMEOUT):
        """Set target pose.
        """
        self._target_pose = pose
        self._target_linear_velocity = self.physics.time_step * (
            linear_velocity or self._max_linear_velocity)
        self._target_angular_velocity = self.physics.time_step * (
            angular_velocity or self._max_angular_velocity)
        self._start_time = self.physics.time()
        self._stop_time = self._start_time + timeout

        self._position_threshold = POSITION_THRESHOLD
        self._euler_threshold = EULER_THRESHOLD

    def update(self):
        """Update control and disturbances."""
        # Call the update function of the super class.
        super(ControllableConstraint, self).update()

        if self._target_pose is not None:
            self._update_pose_control()

            if self.physics.num_steps % NUM_STEPS_CHECK == 0:
                if self.check_reached() or self.check_timeout():
                    self.reset_targets()

    def _update_pose_control(self):
        """Update the pose control."""
        delta_position = self._target_pose.position - self.pose.position
        delta_position /= np.linalg.norm(delta_position)
        delta_position *= self._target_linear_velocity
        new_position = self.pose.position + delta_position

        delta_euler = (self._target_pose.euler - self.pose.euler + np.pi
                       ) % (2 * np.pi) - np.pi
        delta_euler = np.minimum(np.maximum(
            delta_euler,
            -self._target_angular_velocity),
            self._target_angular_velocity)
        new_euler = self.pose.euler + delta_euler
        new_euler[0] = (new_euler[0] + np.pi) % (2 * np.pi) - np.pi
        new_euler[1] = (new_euler[1] + 0.5 * np.pi) % np.pi - 0.5 * np.pi
        new_euler[2] = (new_euler[2] + np.pi) % (2 * np.pi) - np.pi

        new_pose = Pose((new_position, new_euler))

        self.pose = new_pose

    def check_reached(self):
        """Check if the specified joint positions are reached.

        Returns:
            True if the target has been reached, False otherwise.
        """
        delta_position = self._target_pose.position - self.pose.position
        position_reached = (
            abs(delta_position)[0] < self._position_threshold and
            abs(delta_position)[1] < self._position_threshold and
            abs(delta_position)[2] < self._position_threshold)

        delta_euler = self._target_pose.euler - self.pose.euler
        euler_reached = (
            abs(delta_euler)[0] % (2 * np.pi) < self._euler_threshold and
            abs(delta_euler)[1] % (2 * np.pi) < self._euler_threshold and
            abs(delta_euler)[2] % (2 * np.pi) < self._euler_threshold)

        if not (position_reached and euler_reached):
            return False
        else:
            return True

    def check_timeout(self):
        """Check if the joint is timeout.

        Returns:
            True if it is timeout, False otherwise.
        """
        is_timeout = self.physics.time() >= self._stop_time

        if is_timeout:
            logger.warning(
                'Time out (%.2f) with target_pose = %s, current_pose = %s.'
                % (self._stop_time - self._start_time,
                    self._target_pose,
                    self.pose)
            )

        return is_timeout
