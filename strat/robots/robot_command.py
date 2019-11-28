"""The class of robot command."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class RobotCommand(object):
    """Robot command."""

    POSITION_CONTROL = 'POSITION_CONTROL'
    IK_CONTROL = 'IK_CONTROL'
    SET_MAX_JOINT_VELOCITIES = 'SET_MAX_JOINT_VELOCITIES'

    def __init__(self,
                 component,
                 command_type,
                 arguments,
                 timeout=None,
                 async=False):
        """Initialize.

        Args:
            component: The robot component to execute this command.
            command_type: The type of the command to be executed.
            args: The arguments of the command.
            timeout: The maximal execution time.
            async: If the command is asynchronous.
            """
        self.component = component
        self.command_type = command_type
        self.arguments = arguments
        self.timeout = timeout
        self.async = async
