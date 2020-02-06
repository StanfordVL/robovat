"""The base class of the Sawyer robot."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

from robovat.robots import robot
from robovat.utils.yaml_config import YamlConfig


class Sawyer(robot.Robot):
    """Sawyer robot."""

    __metaclass__ = abc.ABCMeta

    def __init__(self, config=None):
        """Initialize.

        config: The configuartion as a dictionary.
        """
        self.config = config or self.default_config
        if isinstance(self.config, str):
            self.config = YamlConfig(self.config).as_easydict()

    @abc.abstractmethod
    def reboot(self):
        """Reboot the robot.
        """
        pass

    @abc.abstractmethod
    def reset(self, positions=None):
        """Reset the robot.

        If the positions are not specified, the robot will go to the neutral
        positions.

        Args:
            positions: The target joint positions, as a list or a dictionary of
                joint_name:joint_position.
        """
        pass

    @abc.abstractmethod
    def move_to_joint_positions(self,
                                positions,
                                speed=None,
                                timeout=None,
                                threshold=None):
        """Move the arm to the specified joint positions.

        Args:
            positions: The target joint positions, as a dictionary of
                joint_name:joint_position.
            speed: The maximum joint velocity.
            timeout: Seconds to wait for move to finish.
            threshold: Joint position threshold in radians across each joint
                when move is considered successful.
        """
        pass

    @abc.abstractmethod
    def move_to_gripper_pose(self,
                             pose,
                             speed=None,
                             timeout=None,
                             threshold=None):
        """Move the arm to the specified gripper pose.

        Args:
            pose: The target gripper pose, as a tuple or an instance of Pose.
            speed: The maximum joint velocity.
            timeout: Seconds to wait for move to finish.
            threshold: Position threshold in radians across each joint when
                move is considered successful.
        """
        pass

    @abc.abstractmethod
    def move_along_gripper_path(self,
                                poses,
                                speed=None):
        """Move the arm to follow a path of the gripper.

        Args:
            poses: The path of the gripper as a list of Pose instances.
            speed: The maximum joint velocity.
        """
        pass

    @abc.abstractmethod
    def grip(self, value=1):
        """Control the gripper to grip.

        Caution: This method is asynchronous control.

        Args:
            value: The value between 0 and 1 to control the gripper to close.
        """
        pass

    @abc.abstractmethod
    def is_limb_ready(self):
        """Check if the limb is ready.

        Returns:
            True if all control commands are done, False otherwise.
        """
        pass

    @abc.abstractmethod
    def is_gripper_ready(self):
        """Check if the gripper is ready.

        Returns:
            True if all control commands are done, False otherwise.
        """
        pass
