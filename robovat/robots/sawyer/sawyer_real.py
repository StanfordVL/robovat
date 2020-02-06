"""The class of the real-world Sawyer robot."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time

import numpy as np

from robovat.math import get_transform
from robovat.math import Pose
from robovat.robots.sawyer import sawyer
from robovat.utils.logging import logger

try:
    import rospy
    import intera_interface
    from intera_core_msgs.msg import JointCommand
    import intera_core_msgs.srv
    import geometry_msgs.msg
    from std_msgs.msg import Header
except ImportError:
    logger.warning('Failed to import rospy packages.')

try:
    import moveit_commander
except ImportError:
    logger.warning('Failed to import moveit packages.')


def convert_pose_for_ros(pose):
    """Convert an instance from robovat.math.Pose into geometry_msgs.msg.Pose.

    Args:
        pose: An instance of robovat.math.Pose.

    Returns:
        An instance of geometry_msgs.msg.Pose.
    """
    # The Orientation is defined as x, y, z, w.
    pose = Pose(pose)
    position = pose.position.tolist()
    orientation = pose.quaternion.tolist()

    return geometry_msgs.msg.Pose(
            position=geometry_msgs.msg.Point(*position),
            orientation=geometry_msgs.msg.Quaternion(*orientation))


class SawyerReal(sawyer.Sawyer):
    """Sawyer robot in the real world."""

    def __init__(self, config=None):
        """Initialize.

        See the parent class.
        """
        super(SawyerReal, self).__init__(config=config)

        if rospy.get_name() == '/unnamed':
            rospy.init_node('sawyer')

        assert rospy.get_name() != '/unnamed', 'Must call rospy.init_node().'

        tic = time.time()
        rospy.loginfo('Initializing robot...')

        self._head = intera_interface.Head()
        self._display = intera_interface.HeadDisplay()
        self._lights = intera_interface.Lights()

        self._limb = intera_interface.Limb()
        self._joint_names = self._limb.joint_names()

        self.cmd = []

        self._command_msg = JointCommand()
        self._command_msg.names = self._joint_names

        self._commanders = dict(velocity=None, torque=None)

        try:
            self._gripper = intera_interface.Gripper()
            self._has_gripper = True
        except Exception:
            self._has_gripper = False

        self._robot_enable = intera_interface.RobotEnable(True)

        self._params = intera_interface.RobotParams()

        self._motion_planning = self.config.MOTION_PLANNING

        if self._motion_planning == 'moveit':
            rospy.loginfo('Initializing moveit toolkit...')
            moveit_commander.roscpp_initialize(sys.argv)
            self._scene = moveit_commander.PlanningSceneInterface()
            self._group = moveit_commander.MoveGroupCommander('right_arm')

        toc = time.time()
        rospy.loginfo('Initialization finished after %.3f seconds.'
                      % (toc - tic))

    @property
    def version(self):
        """List current versions of wrapped SDK, gripper, and robot.

        Returns:
            Dictionary of version information.
        """
        return {
            'SDKVersion': intera_interface.settings.SDK_VERSION,
            'SDK2Gripper': intera_interface.settings.VERSIONS_SDK2GRIPPER,
            'SDK2Robot': intera_interface.settings.VERSIONS_SDK2ROBOT
        }

    @property
    def pose(self):
        raise NotImplementedError

    @property
    def arm(self):
        return self._limb

    @property
    def end_effector(self):
        ros_pose = self._limb.endpoint_pose()
        ros_position = ros_pose['position']
        ros_quaternion = ros_pose['orientation']

        position = [ros_position.x, ros_position.y, ros_position.z]
        orientation = [
            ros_quaternion.x,
            ros_quaternion.y,
            ros_quaternion.z,
            ros_quaternion.w
        ]

        return Pose([position, orientation])

    @property
    def joint_positions(self):
        return self._limb.joint_angles()

    @property
    def joint_velocities(self):
        return self._limb.joint_velocities()

    def reboot(self):
        """Reboot the robot.
        """
        if self._robot_enable.state().error:
            self._robot_enable.reset()

        if self._has_gripper:
            self._gripper.reboot()
            self._gripper.calibrate()

        self.reset()

    def reset(self, positions=None):
        """Reset the robot.
        """
        # Move the limb to the neural positions or a specified positions.
        if positions is None:
            self._limb.move_to_neutral()
        else:
            self.move_to_joint_positions(positions)

        # Start and open the gripper.
        if self._has_gripper:
            self._gripper.start()
            # self.grip(0)

    def move_to_joint_positions(self,
                                positions,
                                speed=None,
                                timeout=None,
                                threshold=None):
        """Move the arm to the specified joint positions.


        Please refer to:
        https://rethinkrobotics.github.io/intera_sdk_docs/5.0.4/intera_interface/html/intera_interface.limb.Limb-class.html

        See the parent class.
        """
        if speed is None:
            speed = self.config.LIMB_MAX_VELOCITY_RATIO

        if timeout is None:
            timeout = self.config.LIMB_TIMEOUT

        if threshold is None:
            threshold = self.config.LIMB_POSITION_THRESHOLD

        # Set the maximum joint velocities for the limb.
        self._limb.set_joint_position_speed(speed)

        # Convert the input positions into
        if isinstance(positions, dict):
            pass
        else:
            positions = {
                self._joint_names[joint_ind]: joint_position
                for joint_ind, joint_position in enumerate(positions)
            }

        # Send the synchronous command.
        self._limb.move_to_joint_positions(
            positions=positions, timeout=timeout, threshold=threshold)

    def move_to_gripper_pose(self,
                             pose,
                             speed=None,
                             timeout=None,
                             threshold=None,
                             straight_line=False):
        """Move the arm to the specified joint positions.


        Please refer to:
        https://rethinkrobotics.github.io/intera_sdk_docs/5.0.4/intera_interface/html/intera_interface.limb.Limb-class.html

        See the parent class.
        """
        if speed is None:
            speed = self.config.LIMB_MAX_VELOCITY_RATIO

        if timeout is None:
            timeout = self.config.LIMB_TIMEOUT

        if threshold is None:
            threshold = self.config.LIMB_POSITION_THRESHOLD

        if straight_line:
            start_pose = self.end_effector
            end_pose = pose

            if self._motion_planning is None:
                # A hacky way of straight line motion without motion planning.
                rospy.logwarn('No motion planning is available. Use the hacky'
                              'way of straight line motion.')
                delta_position = end_pose.position - start_pose.position
                end_effector_step = 0.1
                num_waypoints = int(np.linalg.norm(delta_position)
                                    / end_effector_step)
                waypoints = []

                for i in range(num_waypoints):
                    scale = float(i) / float(num_waypoints)
                    position = start_pose.position + delta_position * scale
                    euler = end_pose.euler
                    waypoint = Pose([position, euler])
                    waypoints.append(waypoint)

                waypoints.append(end_pose)

            else:
                # With motion planning, the path is always a straight line.
                waypoints = [start_pose, end_pose]

            self.move_along_gripper_path(waypoints, speed=speed)

        else:
            positions = self._compute_inverse_kinematics(
                self.config.END_EFFCTOR_NAME, pose)

            if positions is None:
                rospy.logerr("IK response is not valid.")
            else:
                self.move_to_joint_positions(
                    positions=positions,
                    speed=speed,
                    timeout=timeout,
                    threshold=threshold)

    def move_along_gripper_path(self,
                                poses,
                                speed=None):
        """Move the arm to follow a path of the gripper.

        Please refer to:
        http://sdk.rethinkrobotics.com/intera/MoveIt_Tutorial
        http://docs.ros.org/kinetic/api/moveit_tutorials/html/doc/pr2_tutorials/planning/scripts/doc/move_group_python_interface_tutorial.html#cartesian-paths

        See the parent class.
        """
        if speed is None:
            speed = self.config.LIMB_MAX_VELOCITY_RATIO

        if self._motion_planning == 'moveit':
            # Plan with moveit.
            waypoints = []

            for pose in poses:
                # Compute the 'right_gripper' pose from the 'right_hand' pose
                # for moveit. This is a temporary fix.
                pose_frame = get_transform(source=pose)
                pose = pose_frame.transform([[0, 0, 0.0495], [0, 0, 0]])
                waypoint = convert_pose_for_ros(pose)
                waypoints.append(waypoint)

            # Move to the starting pose.
            rospy.loginfo('Moving to the starting pose...')
            self._group.set_pose_target(waypoints[0])
            plan = self._group.plan()
            self._group.execute(plan)

            # Move long the path.
            rospy.loginfo('Planning the trajectory with moveit...')
            plan, _ = self._group.compute_cartesian_path(
                waypoints, self.config.END_EFFECTOR_STEP, 0.0)
            rospy.loginfo('Moving along the planned path...')
            self._group.set_max_velocity_scaling_factor(speed)
            self._group.execute(plan)

        else:
            rospy.loginfo('Moving along the path without planning...')
            for pose in poses:
                self.move_to_gripper_pose(pose, speed=speed)

    def grip(self, value=1.0):
        """Control the gripper to grip.

        Please refer to:
        https://rethinkrobotics.github.io/intera_sdk_docs/5.0.4/intera_interface/html/intera_interface.gripper.Gripper-class.html

        See the parent class.
        """
        if self._has_gripper:
            joint_position = (1.0 - value) * self._gripper.MAX_POSITION
            self._gripper.set_position(joint_position)

    def is_limb_ready(self):
        """Check if the limb is busy with control commands.

        Returns:
            True if all control commands are done, False otherwise.
        """
        # TODO(kuanfang): Implement this method.
        return True

    def is_gripper_ready(self):
        """Check if the gripper is busy with control commands.

        Returns:
            True if all control commands are done, False otherwise.
        """
        # TODO(kuanfang): Implement this method.
        return True

    def _compute_inverse_kinematics(self, link_name, link_pose):
        """Move the arm to the specified joint positions.

        Args:
            link_uid: The name of the link.
            link_pose: The target pose of the link.

        Returns:
            A ditionary of the IK result. Return None if the IK result is not
                valid.
        """
        # Create an IK service.
        service_name = 'ExternalTools/right/PositionKinematicsNode/IKService'
        ik_service = rospy.ServiceProxy(
            service_name, intera_core_msgs.srv.SolvePositionIK)
        ik_request = intera_core_msgs.srv.SolvePositionIKRequest()

        # The Header.
        header = Header(stamp=rospy.Time.now(), frame_id='base')

        # The Orientation is defined as x, y, z, w.
        pose = convert_pose_for_ros(link_pose)
        pose_stamped = geometry_msgs.msg.PoseStamped(header=header, pose=pose)

        # Request inverse kinematics from base to "right_hand" link.
        ik_request.tip_names.append(self.config.END_EFFCTOR_NAME)

        # Add desired pose for inverse kinematics.
        ik_request.pose_stamp.append(pose_stamped)

        # Solve the IK.
        try:
            rospy.wait_for_service(service_name, 5.0)
            ik_response = ik_service(ik_request)
        except (rospy.ServiceException, rospy.ROSException) as e:
            rospy.logerr("Service call failed: %s." % e)
            return False

        # Return the results.
        if ik_response.result_type[0] > 0:
            positions = dict(
                zip(ik_response.joints[0].name,
                    ik_response.joints[0].position)
            )
            return positions
        else:
            rospy.logerr("IK response is not valid.")
            return None
