"""The class of the Sawyer robot."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from robovat.math import Pose
from robovat.robots.robot_command import RobotCommand
from robovat.robots.sawyer import sawyer
from robovat.utils.logging import logger


class SawyerSim(sawyer.Sawyer):
    """Sawyer wrapper in simulation."""

    ARM_NAME = 'sawyer_arm'
    BASE_NAME = 'sawyer_base'
    HEAD_NAME = 'sawyer_head'

    def __init__(self,
                 simulator,
                 pose=[[0, 0, 0], [0, 0, 0]],
                 joint_positions=None,
                 config=None):
        """Initialize.

        Args:
            simulator: The simulated simulator for the robot.
            pose: The initial pose of the robot base.
            joint_positions: The list of initial joint positions.
            config: The configuartion as a dictionary.
        """
        super(SawyerSim, self).__init__(config=config)

        self._simulator = simulator

        self._arm_pose = Pose(pose)

        self._arm = None
        self._base = None
        self._head = None

        if joint_positions is None:
            self._initial_joint_positions = self.config.LIMB_NEUTRAL_POSITIONS
        else:
            self._initial_joint_positions = joint_positions

        self.reboot()

    @property
    def arm(self):
        return self._arm

    @property
    def end_effector(self):
        return self._end_effector

    @property
    def base(self):
        return self._base

    @property
    def head(self):
        return self._head

    @property
    def l_finger_tip(self):
        return self._l_finger_tip

    @property
    def r_finger_tip(self):
        return self._r_finger_tip

    @property
    def pose(self):
        return self._arm.pose

    @property
    def joint_positions(self):
        return dict(
            (joint.name, joint.position) for joint in self._limb_joints
        )

    def reboot(self):
        """Reboot the robot.
        """
        # Remove the bodies if it already exists.
        if self._arm is not None:
            self._simulator.remove_body(self._arm)

        if self._base is not None:
            self._simulator.remove_body(self._base)

        if self._head is not None:
            self._simulator.remove_body(self._head)

        # Add the arm.
        self._arm = self._simulator.add_body(
            filename=self.config.ARM_URDF,
            pose=self._arm_pose,
            is_static=True,
            is_controllable=True,
            name=self.ARM_NAME)

        # Add the base.
        self._base = self._simulator.add_body(
            filename=self.config.BASE_URDF,
            pose=self._arm_pose,
            is_static=True,
            is_controllable=False,
            name=self.BASE_NAME)

        # Add the head.
        self._head = self._simulator.add_body(
            filename=self.config.HEAD_URDF,
            pose=self._arm_pose,
            is_static=True,
            is_controllable=False,
            name=self.HEAD_NAME)

        self._arm.set_neutral_joint_positions(
                self.config.LIMB_NEUTRAL_POSITIONS)

        # Find the robot parts.
        self._limb_joints = [
            self._arm.get_joint_by_name(joint_name)
            for joint_name in self.config.LIMB_JOINT_NAMES
        ]
        self._limb_inds = [joint.index for joint in self._limb_joints]

        self._end_effector = self._arm.get_link_by_name(
            self.config.END_EFFCTOR_NAME)

        try:
            self._l_finger_joint = self._arm.get_joint_by_name(
                self.config.L_FINGER_NAME)
            self._r_finger_joint = self._arm.get_joint_by_name(
                self.config.R_FINGER_NAME)
            self._l_finger_tip = self._arm.get_link_by_name(
                self.config.L_FINGER_TIP_NAME)
            self._r_finger_tip = self._arm.get_link_by_name(
                self.config.R_FINGER_TIP_NAME)
            self._gripper_inds = [
                self._l_finger_joint.index,
                self._r_finger_joint.index
            ]
            self._gripper_ready_time = 0
            self._has_gripper = True
        except Exception:
            logger.warn('The gripper is not found.')
            self._has_gripper = False

        # Reset the positions.
        assert len(self._limb_inds) == len(
            self.config.LIMB_NEUTRAL_POSITIONS)
        assert len(self._limb_inds) == len(self._initial_joint_positions)

        for i, joint_ind in enumerate(self._limb_inds):
            limb_joint = self._arm.joints[joint_ind]
            limb_joint.position = self._initial_joint_positions[i]

        if self._has_gripper:
            self._l_finger_joint.position = (
                self._l_finger_joint.upper_limit)
            self._r_finger_joint.position = (
                self._r_finger_joint.lower_limit)

            if self.config.OPEN_GRIPPER_WHEN_RESET:
                self.grip(0)

    def reset(self, positions=None):
        """Reset the robot.
        """
        # Move the limb to the neural positions or a specified positions.
        if positions is None:
            self.move_to_joint_positions(self.config.LIMB_NEUTRAL_POSITIONS)
        else:
            self.move_to_joint_positions(positions)

        if self._has_gripper:
            # Start and open the gripper.
            self.grip(0)

    def move_to_joint_positions(self,
                                positions,
                                speed=None,
                                timeout=None,
                                threshold=None):
        """Move the arm to the specified joint positions.

        See the parent class.
        """
        self._arm.reset_targets()

        if speed is None:
            speed = self.config.LIMB_MAX_VELOCITY_RATIO

        if timeout is None:
            timeout = self.config.LIMB_TIMEOUT

        if threshold is None:
            threshold = self.config.LIMB_POSITION_THRESHOLD

        # Set the maximal joint velocities.
        joint_velocities = dict([
            (joint.name, speed * joint.max_velocity)
            for joint in self._limb_joints
            ])
        kwargs = {
            'joint_velocities': joint_velocities,
        }

        robot_command = RobotCommand(
            component=self._arm.name,
            command_type='set_max_joint_velocities',
            arguments=kwargs)

        self._send_robot_command(robot_command)

        # Command the position control.
        kwargs = {
            'joint_positions': positions,
            'timeout': timeout,
            'threshold': threshold,
        }

        robot_command = RobotCommand(
            component=self._arm.name,
            command_type='set_target_joint_positions',
            arguments=kwargs)

        self._send_robot_command(robot_command)

    def move_to_gripper_pose(self,
                             pose,
                             speed=None,
                             timeout=None,
                             threshold=None,
                             straight_line=False):
        """Move the arm to the specified gripper pose.

        See the parent class.
        """
        if speed is None:
            speed = self.config.LIMB_MAX_VELOCITY_RATIO

        if timeout is None:
            timeout = self.config.LIMB_TIMEOUT

        if threshold is None:
            threshold = self.config.LIMB_POSITION_THRESHOLD

        self._arm.reset_targets()

        pose = Pose(pose)

        if straight_line:
            start_pose = self.end_effector.pose
            end_pose = pose
            delta_position = end_pose.position - start_pose.position
            num_waypoints = int(np.linalg.norm(delta_position)
                                / self.config.END_EFFECTOR_STEP)
            waypoints = []

            for i in range(num_waypoints):
                scale = float(i) / float(num_waypoints)
                position = start_pose.position + delta_position * scale
                euler = end_pose.euler
                waypoint = Pose([position, euler])
                waypoints.append(waypoint)

            waypoints.append(end_pose)

            self.move_along_gripper_path(waypoints, speed=speed)

        else:
            # Set the maximal joint velocities.
            joint_velocities = dict([
                (joint.name, speed * joint.max_velocity)
                for joint in self._limb_joints
                ])
            kwargs = {
                'joint_velocities': joint_velocities,
            }

            robot_command = RobotCommand(
                component=self._arm.name,
                command_type='set_max_joint_velocities',
                arguments=kwargs)

            self._send_robot_command(robot_command)

            # Command the IK control.
            kwargs = {
                'link_ind': self._end_effector.index,
                'link_pose': pose,
                'timeout': timeout,
                'threshold': threshold,
            }

            robot_command = RobotCommand(
                component=self._arm.name,
                command_type='set_target_link_pose',
                arguments=kwargs)

            self._send_robot_command(robot_command)

    def move_along_gripper_path(self,
                                poses,
                                speed=None,
                                timeout=None,
                                threshold=None):
        """Move the arm to follow a path of the gripper.

        See the parent class.
        """
        if speed is None:
            speed = self.config.LIMB_MAX_VELOCITY_RATIO

        if timeout is None:
            timeout = self.config.LIMB_TIMEOUT

        if threshold is None:
            threshold = self.config.LIMB_POSITION_THRESHOLD

        self._arm.reset_targets()

        # Set the maximal joint velocities.
        joint_velocities = dict([
            (joint.name, speed * joint.max_velocity)
            for joint in self._limb_joints
        ])
        kwargs = {
            'joint_velocities': joint_velocities,
        }

        robot_command = RobotCommand(
            component=self._arm.name,
            command_type='set_max_joint_velocities',
            arguments=kwargs)

        self._send_robot_command(robot_command)

        # Command the IK control.

        kwargs = {
            'link_ind': self._end_effector.index,
            'link_poses': poses,
            'timeout': timeout,
            'threshold': threshold,
        }

        robot_command = RobotCommand(
            component=self._arm.name,
            command_type='set_target_link_poses',
            arguments=kwargs)

        self._send_robot_command(robot_command)

    def grip(self, value=1):
        """Control the gripper to grip.

        See the parent class.
        """
        # Clip the value. The fingers somtimes stuck in the max/min positions.
        value = np.clip(value, 0.01, 0.99)

        l_finger_position = (self._l_finger_joint.upper_limit -
                             value * self._l_finger_joint.range)
        r_finger_position = (self._r_finger_joint.lower_limit +
                             value * self._r_finger_joint.range)

        joint_positions = {
            self._l_finger_joint.name: l_finger_position,
            self._r_finger_joint.name: r_finger_position,
        }

        # Command the position control.
        kwargs = {
            'joint_positions': joint_positions,
            'timeout': 10000,
        }

        robot_command = RobotCommand(
            component=self._arm.name,
            command_type='set_target_joint_positions',
            arguments=kwargs)

        self._send_robot_command(robot_command)
        self._gripper_ready_time = self._arm.physics.time() + 0.5

    def is_limb_ready(self):
        """Check if the limb is busy with control commands.

        Returns:
            True if all control commands are done, False otherwise.
        """
        return self._arm.is_ready(joint_inds=self._limb_inds)

    def is_gripper_ready(self):
        """Check if the gripper is busy with control commands.

        Returns:
            True if all control commands are done, False otherwise.
        """
        return self._arm.physics.time() >= self._gripper_ready_time

    def _send_robot_command(self, robot_command):
        """Send a robot command to the server.

        Args:
            robot_command: An instance of RobotCommand.
        """
        self._simulator.receive_robot_commands(robot_command)
