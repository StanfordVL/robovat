"""The body class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from robovat.simulation.body import Body
from robovat.utils.logging import logger


# Default simulation parameters for the Sawyer robot.
TIMEOUT = 15.0
POSITION_THRESHOLD = 0.008726640
VELOCITY_THRESHOLD = 0.05
POSITION_GAIN = 0.05
VELOCITY_GAIN = 1.0
JOINT_DAMPING = 0.7

# Number of steps to check if the target has been reached.
STEPS_TO_CHECK_DONE = 100

# Number of steps to recompute IK during simulation.
STEPS_TO_UPDATE_IK = 10


class JointTarget(object):
    """Specify the target in terms of joint positions and velocities."""

    def __init__(self, body):
        self._body = body
        self.reset()

    @property
    def indices(self):
        return self._indices

    @property
    def positions(self):
        return self._positions

    @property
    def velocities(self):
        return self._velocities

    @property
    def start_time(self):
        return self._start_time

    @property
    def stop_time(self):
        return self._stop_time

    @property
    def position_gains(self):
        return self._position_gains

    @property
    def velocity_gains(self):
        return self._velocity_gains

    @property
    def position_threshold(self):
        return self._position_threshold

    @property
    def velocity_threshold(self):
        return self._velocity_threshold

    def is_ready(self):
        """Check if the target has been reached.

        Returns:
            True if the target has been reached, False otherwise.
        """
        return self._indices is None

    def reset(self):
        """Reset the target."""
        self._indices = None
        self._positions = None
        self._velocities = None
        self._start_time = None
        self._stop_time = None

        self._position_gains = None
        self._velocity_gains = None

        self._position_threshold = None
        self._velocity_threshold = None

    def set(self,
            indices,
            positions,
            velocities,
            start_time=None,
            stop_time=None,
            position_gains=None,
            velocity_gains=None,
            position_threshold=None,
            velocity_threshold=None,
            timeout=TIMEOUT):
        """Set the target.

        Args:
            indices: Joint indices.
            positions: Target joint positions.
            velocities: Target joint velocities.
            start_time: Starting time of the target.
            stop_time: Stopping time of the target.
            position_gains: Position gain of position control.
            velocity_gains: Velocity gain of position control.
            position_threshold: Threshold of reaching the target positions.
            velocity_threshold: Threshold of reaching the target velocities.
            timeout: Maximum execution time in seconds.
        """
        self._indices = indices
        self._positions = positions
        self._velocities = velocities

        self._start_time = start_time or self._body.physics.time()
        self._stop_time = stop_time or self._start_time + timeout

        num_joints = len(indices)
        self._position_gains = position_gains or [POSITION_GAIN] * num_joints
        self._velocity_gains = velocity_gains or [VELOCITY_GAIN] * num_joints

        self._position_threshold = position_threshold or POSITION_THRESHOLD
        self._velocity_threshold = velocity_threshold or VELOCITY_THRESHOLD


class LinkTarget(object):
    """Specify the target in terms of pose of a link."""

    def __init__(self, body):
        self._body = body
        self.reset()

    @property
    def index(self):
        return self._index

    @property
    def link(self):
        return self._link

    @property
    def pose(self):
        return self._pose

    @property
    def pose_queue(self):
        return self._pose_queue

    @property
    def start_time(self):
        return self._start_time

    @property
    def position_threshold(self):
        return self._position_threshold

    @property
    def velocity_threshold(self):
        return self._velocity_threshold

    @property
    def stop_time(self):
        return self._stop_time

    def is_ready(self):
        """Check if the target has been reached.

        Returns:
            True if the target has been reached, False otherwise.
        """
        return self._index is None

    def reset(self):
        """Reset the target."""
        self._index = None
        self._link = None
        self._pose = None
        self._pose_queue = []

        self._start_time = None
        self._stop_time = None

    def set(self,
            index,
            pose,
            start_time=None,
            stop_time=None,
            position_threshold=None,
            velocity_threshold=None,
            timeout=TIMEOUT):
        """Set the target.

        Args:
            index: Link index.
            pose: Target link pose.
            start_time: Starting time of the target.
            stop_time: Stopping time of the target.
            position_threshold: Threshold of reaching the target positions.
            velocity_threshold: Threshold of reaching the target velocities.
            timeout: Maximum execution time in seconds.
        """
        self._index = index
        self._link = self._body.links[index]

        if isinstance(pose, list):
            self._pose = None
            self._pose_queue = pose
        else:
            self._pose = pose
            self._pose_queue = []

        self._start_time = start_time or self._body.physics.time()
        self._stop_time = stop_time or self._start_time + timeout

        self._position_threshold = position_threshold or POSITION_THRESHOLD
        self._velocity_threshold = velocity_threshold or VELOCITY_THRESHOLD

    def pop(self):
        if len(self._pose_queue) == 0:
            self.reset()
        else:
            self._pose = self._pose_queue[0]
            self._pose_queue = self._pose_queue[1:]

        return self._pose


class ControllableBody(Body):
    """Body."""

    def __init__(self,
                 simulator,
                 filename,
                 pose,
                 scale=1.0,
                 is_static=False,
                 name=None):
        """Initialize."""
        Body.__init__(self,
                      simulator=simulator,
                      filename=filename,
                      pose=pose,
                      scale=scale,
                      is_static=is_static,
                      name=name)

        self._link_target = LinkTarget(self)
        self._joint_target = JointTarget(self)

        self._neutral_joint_positions = None
        self._max_joint_velocities = [
            joint.max_velocity for joint in self.joints
        ]
        self._max_reaction_forces = None

    def set_target_joint_positions(self,
                                   joint_positions,
                                   timeout=TIMEOUT,
                                   threshold=POSITION_THRESHOLD):
        """Set target joint positions for position control.

        Args:
            joint_positions: The joint positions for each specified joint.
            timeout: Seconds to wait for move to finish.
            threshold: Joint position threshold in radians across each joint
                when move is considered successful.
        """
        indices = []
        positions = []

        if isinstance(joint_positions, (list, tuple)):
            for ind, position in enumerate(joint_positions):
                if position is not None:
                    indices.append(ind)
                    positions.append(position)
        elif isinstance(joint_positions, dict):
            for key, position in joint_positions.items():
                joint = self.get_joint_by_name(key)
                indices.append(joint.index)
                positions.append(position)
        else:
            raise ValueError

        num_joints = len(joint_positions)
        velocities = [0.0] * num_joints

        self._joint_target.set(
            indices,
            positions,
            velocities,
            timeout=timeout,
            position_threshold=threshold)

    def set_target_link_pose(self,
                             link_ind,
                             link_pose,
                             timeout=TIMEOUT,
                             threshold=POSITION_THRESHOLD):
        """Set the target pose of the link for position control.

        Args:
            link_ind: The index of the end effector link.
            link_pose: The pose of the end effector link.
            timeout: Seconds to wait for move to finish.
            threshold: Joint position threshold in radians across each joint
                when move is considered successful.
        """
        self._link_target.set(
            index=link_ind,
            pose=link_pose,
            timeout=timeout,
            position_threshold=threshold)

    def set_target_link_poses(self,
                              link_ind,
                              link_poses,
                              timeout=TIMEOUT,
                              threshold=POSITION_THRESHOLD):
        """Set the target poses of the link for position control.

        The list of poses should be reached sequentially.

        Args:
            link_ind: The index of the end effector link.
            link_pose: The pose of the end effector link.
            timeout: Seconds to wait for move to finish.
            threshold: Joint position threshold in radians across each joint
                when move is considered successful.
        """
        assert isinstance(link_poses, list)

        self._link_target.set(
            index=link_ind,
            pose=link_poses,
            timeout=timeout,
            position_threshold=threshold)

        self._link_target.pop()

    def set_neutral_joint_positions(self, joint_positions):
        """Set the neutral joint positions.

        This is used for computing the IK solution.

        Args:
            joint_positions: A list of joint positions.
        """
        self._neutral_joint_positions = joint_positions

    def set_max_joint_velocities(self, joint_velocities):
        """Set the maximal joint velocities for position control.

        Args:
            joint_velocities: The maximal joint velocities.
        """
        # TODO(kuanfang): This functionality is not supported by pybulelt yet.
        if isinstance(joint_velocities, (list, tuple)):
            for joint_ind, joint_velocity in enumerate(joint_velocities):
                if joint_velocity is not None:
                    self._max_joint_velocities[joint_ind] = joint_velocity
        elif isinstance(joint_velocities, dict):
            for key, joint_velocity in joint_velocities.items():
                joint = self.get_joint_by_name(key)
                joint_ind = joint.index
                self._max_joint_velocities[joint_ind] = joint_velocity
        else:
            raise ValueError

    def set_max_reaction_force(self, joint_ind, force):
        """Set the maximum reaction force for a joint.
        """
        self._max_reaction_forces[joint_ind] = force
        self.joints[joint_ind].enable_sensor()

    def reset_targets(self):
        """Reset the joint and link targets."""
        self._link_target.reset()
        self._joint_target.reset()

    def update(self):
        """Update control and disturbances."""
        ik_updated = False

        if not self._link_target.is_ready():
            if self.simulator.num_steps % STEPS_TO_CHECK_DONE == 0:
                if self._check_link_target_done():
                    self._link_target.reset()

        if not self._link_target.is_ready():
            if (self.simulator.num_steps % STEPS_TO_UPDATE_IK == 0 or
                    self._joint_target.is_ready()):
                self._update_ik()
                ik_updated = True

                # Check if the IK computation has converged.
                if self.check_joints_reached():
                    self._link_target.pop()

        if not self._joint_target.is_ready():
            if (self.simulator.num_steps % STEPS_TO_CHECK_DONE == 0
                    or ik_updated):
                if self._check_joint_target_done():
                    self._joint_target.reset()

        if not self._joint_target.is_ready():
            self._update_position_control()

    def _check_link_target_done(self):
        """Check if the link target has been reached.

        Returns:
            True if the target has reached, False otherwise.
        """
        # Check timeout.
        if self._link_target.stop_time is None:
            return True
        elif self.physics.time() >= self._link_target.stop_time:
            return True

        # Check queue clear.
        if (self._link_target.pose is None and
                len(self._link_target.pose_queue) == 0):
            return True

        return False

    def _check_joint_target_done(self):
        """Check if the joint target has been reached.

        Returns:
            True if the target has reached, False otherwise.
        """
        # Check timeout.
        if self._joint_target.stop_time is None:
            return True
        elif self.physics.time() >= self._joint_target.stop_time:
            logger.debug('Time out for the position control.')
            return True

        # Check if target joints have been reached.
        if self.check_joints_reached():
            return True

        # Check if joints are safe.
        if self._max_reaction_forces is not None:
            if not self.check_joints_safe():
                return True

        return False

    def _update_position_control(self):
        """Update the position control."""
        self.physics.position_control_array(
            self.uid,
            self._joint_target.indices,
            target_positions=self._joint_target.positions,
            target_velocities=self._joint_target.velocities,
            position_gains=self._joint_target.position_gains,
            velocity_gains=self._joint_target.velocity_gains)

    def _update_ik(self):
        """Update the inverse kinematics results.
        """
        num_joints = self._link_target.index
        joint_inds = range(num_joints)

        # Computer IK.
        positions = self.physics.compute_inverse_kinematics(
            self._link_target.link.uid,
            self._link_target.pose,
            neutral_positions=self._neutral_joint_positions)

        # TODO: Bullet IK results include irrelevant joints, which can cause
        # problems for some URDF models.
        positions = positions[:num_joints]

        # If there is no more target lin poses waiting in the queue, the target
        # velocities are zeros; otherwise velocities do not matter.
        if len(self._link_target.pose_queue) == 0:
            velocities = [0.0] * num_joints
        else:
            velocities = None

        # Set the position control.
        self._joint_target.set(
            joint_inds,
            positions,
            velocities,
            start_time=self._link_target.start_time,
            stop_time=self._link_target.stop_time,
            position_threshold=self._link_target.position_threshold,
            velocity_threshold=self._link_target.velocity_threshold)

    def check_joints_reached(self):
        """Check if the specified joint positions are reached.

        Returns:
            True is all specified joint positions are reached and corresponding
                joint velocities are close to zero, False otherwise.
        """
        if self._joint_target.is_ready():
            return True

        inds = self._joint_target.indices
        positions = self._joint_target.positions
        velocities = self._joint_target.velocities

        if velocities is None:
            velocities = [None] * len(inds)

        for ind, target_position, target_velocity in zip(
                inds, positions, velocities):
            current_position = self.joints[ind].position
            delta_position = target_position - current_position
            position_reached = (
                abs(delta_position) < self._joint_target.position_threshold)

            if target_velocity is None:
                velocity_reached = True
            else:
                current_velocity = self.joint_velocities[ind]
                delta_velocity = target_velocity - current_velocity
                velocity_reached = (
                    abs(delta_velocity) < self._joint_target.velocity_threshold
                )

            if not (position_reached and velocity_reached):
                return False

        return True

    def check_joints_safe(self):
        """Check if the joint is safe.

        Returns:
            True for safe, False otherwise.
        """
        for joint_ind in self._joint_target.indices:
            max_force = self._max_reaction_forces[joint_ind]

            if max_force is None:
                continue
            else:
                joint = self.joints[joint_ind]
                reaction_force = joint.reaction_force
                reaction_force_norm = np.linalg.norm(reaction_force[:3])
                is_safe = reaction_force_norm < max_force

                if not is_safe:
                    logger.warning(
                        'Joint %s has a reaction force %s, '
                        'which is larger than the threshold %.2f.'
                        % (joint_ind, reaction_force_norm, max_force))
                    return False

        return True

    def is_ready(self, joint_inds=None):
        """Check if the body is ready.

        Args:
            joint_inds: The joints to be checked.

        Returns:
            True if all control commands are done, False otherwise.
        """
        if self._check_link_target_done():
            self._link_target.reset()

        if self._check_joint_target_done():
            self._joint_target.reset()

        if joint_inds is None:
            return (self._link_target.is_ready() and
                    self._joint_target.is_ready())

        else:
            if not self._link_target.is_ready():
                for joint_ind in joint_inds:
                    if joint_ind < self._link_target.index:
                        return False

            if not self._joint_target.is_ready():
                for joint_ind in joint_inds:
                    if joint_ind in self._joint_target.indices:
                        return False

            return True
