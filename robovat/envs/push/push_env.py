"""Pushing task environment.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import glob
import random
import socket
import shutil

import cv2
import gym
import numpy as np
from matplotlib import pyplot as plt

from robovat.envs import arm_env
from robovat.envs import robot_env
from robovat.envs.push import layouts
from robovat.observations import attribute_obs
from robovat.observations import camera_obs
from robovat.observations import pose_obs
from robovat.reward_fns import push_reward
from robovat.math import Pose
from robovat.utils import time_utils
from robovat.utils.logging import logger


class PushEnv(arm_env.ArmEnv):
    """Pushing task environment."""

    def __init__(self,
                 simulator=None,
                 config=None,
                 debug=True):
        """Initialize.

        Args:
            simulator: Instance of the simulator.
            config: Environment configuration.
            debug: True if it is debugging mode, False otherwise.
        """
        self._simulator = simulator
        self._config = config or self.default_config
        self._debug = debug

        self.camera = self._create_camera(
            height=self.config.KINECT2.DEPTH.HEIGHT,
            width=self.config.KINECT2.DEPTH.WIDTH,
            intrinsics=self.config.KINECT2.DEPTH.INTRINSICS,
            translation=self.config.KINECT2.DEPTH.TRANSLATION,
            rotation=self.config.KINECT2.DEPTH.ROTATION)

        # Layout.
        self.task_name = self.config.TASK_NAME
        self.layout_id = self.config.LAYOUT_ID

        if self.task_name is None:
            self.layouts = None
            self.num_layouts = 1
        elif self.task_name == 'data_collection':
            self.layouts = None
            self.num_layouts = 1
        else:
            self.layouts = layouts.TASK_NAME_TO_LAYOUTS[self.task_name]
            self.num_layouts = len(self.layouts)

        # Action and configuration space.
        self.num_goal_steps = self.config.NUM_GOAL_STEPS
        self.cspace = gym.spaces.Box(
            low=np.array(self.config.ACTION.CSPACE.LOW),
            high=np.array(self.config.ACTION.CSPACE.HIGH),
            dtype=np.float32)
        start_low = np.array(self.config.ACTION.CSPACE.LOW, dtype=np.float32)
        start_high = np.array(self.config.ACTION.CSPACE.HIGH, dtype=np.float32)
        self.start_offset = 0.5 * (start_high + start_low)
        self.start_range = 0.5 * (start_high - start_low)
        self.start_z = self.config.ARM.FINGER_TIP_OFFSET + self.start_offset[2]

        table_x = self.config.SIM.TABLE.POSE[0][0]
        table_y = self.config.SIM.TABLE.POSE[0][1]
        self.table_workspace = gym.spaces.Box(
            low=np.array([table_x - 0.5 * self.config.TABLE.X_RANGE,
                         table_y - 0.5 * self.config.TABLE.Y_RANGE]),
            high=np.array([table_x + 0.5 * self.config.TABLE.X_RANGE,
                          table_y + 0.5 * self.config.TABLE.Y_RANGE]),
            dtype=np.float32)

        # Movable Objects.
        self.min_movable_bodies = self.config.MIN_MOVABLE_BODIES
        self.max_movable_bodies = self.config.MAX_MOVABLE_BODIES
        self.num_movable_bodies = None
        self.movable_body_mask = None

        if self.is_simulation:
            movable_name = self.config.MOVABLE_NAME.upper()
            self.movable_config = self.config.MOVABLE[movable_name]
            self.movable_bodies = []
            self.movable_paths = []
            for pattern in self.movable_config.PATHS:
                if not os.path.isabs(pattern):
                    pattern = os.path.join(self.simulator.assets_dir, pattern)
                self.movable_paths += glob.glob(pattern)
            assert len(self.movable_paths) > 0
            self.target_movable_paths = []
            for pattern in self.movable_config.TARGET_PATHS:
                if not os.path.isabs(pattern):
                    pattern = os.path.join(self.simulator.assets_dir, pattern)
                self.target_movable_paths += glob.glob(pattern)
            assert len(self.target_movable_paths) > 0
        else:
            self.movable_config = None
            self.movable_bodies = None
            self.movable_paths = None
            self.target_movable_paths = None

        # Execution phases.
        self.phase_list = ['initial',
                           'pre',
                           'start',
                           'motion',
                           'post',
                           'offstage',
                           'done']

        # Action related information.
        self.attributes = None
        self.start_status = None
        self.end_status = None
        self.max_phase_steps = None

        # Statistics.
        self.num_total_steps = 0
        self.num_unsafe = 0
        self.num_ineffective = 0
        self.num_useful = 0
        self.num_successes = 0
        self.num_successes_by_step = [0] * int(self.config.MAX_STEPS + 1)

        # Recording.
        self.use_recording = self.config.RECORDING.USE
        if self.use_recording:
            self.recording_camera = None
            self.recording_output_dir = None
            self.video_writer = None

        # Visualization.
        if self.debug:
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111)
            plt.ion()
            plt.show()
            self.ax = ax

        super(PushEnv, self).__init__(
            simulator=self.simulator,
            config=self.config,
            debug=self.debug)

    def _create_observations(self):
        """Create observations.

        Returns:
            List of observations.
        """
        observations = [
            attribute_obs.IntegerAttributeObs(
                'num_episodes',
                max_value=int(2**16 - 1),
                name='num_episodes'),
            attribute_obs.IntegerAttributeObs(
                'num_steps',
                max_value=int(2**16 - 1),
                name='num_steps'),
            attribute_obs.IntegerAttributeObs(
                'layout_id',
                max_value=self.num_layouts,
                name='layout_id'),
            attribute_obs.ArrayAttributeObs(
                'movable_body_mask',
                shape=[self.max_movable_bodies],
                name='body_mask'),
        ]

        # In simulation, ground truth segmented point clouds are provided. In
        # the real world, the segmented point clouds are computed using
        # clustering algorithms.
        if self.is_simulation:
            observations += [
                camera_obs.SegmentedPointCloudObs(
                    self.camera,
                    num_points=self.config.OBS.NUM_POINTS,
                    num_bodies=self.max_movable_bodies,
                    name='point_cloud'),
            ]

        else:
            observations += [
                camera_obs.SegmentedPointCloudObs(
                    self.camera,
                    num_points=self.config.OBS.NUM_POINTS,
                    num_bodies=self.max_movable_bodies,
                    crop_min=self.config.OBS.CROP_MIN,
                    crop_max=self.config.OBS.CROP_MAX,
                    confirm_target=True,
                    name='point_cloud'),
            ]

        # Prestiged information.
        if self.is_simulation and self.config.USE_PRESTIGE_OBS:
            observations += [
                pose_obs.PoseObs(
                    num_bodies=self.max_movable_bodies,
                    modality='position',
                    name='position'),
                attribute_obs.FlagObs('is_safe', name='is_safe'),
                attribute_obs.FlagObs('is_effective', name='is_effective'),
            ]

        # Visual observations for visualization.
        if self.config.USE_VISUALIZATION_OBS:
            observations += [
                camera_obs.CameraObs(
                    self.camera,
                    modality='rgb',
                    name='rgb'),
                camera_obs.CameraObs(
                    self.camera,
                    modality='depth',
                    name='depth'),
            ]

        return observations

    def _create_reward_fns(self):
        """Initialize reward functions.

        Returns:
            List of reward functions.
        """
        return [
            push_reward.PushReward(
                name='reward',
                task_name=self.task_name,
                layout_id=self.layout_id,
                is_planning=False
            )
        ]

    def _create_action_space(self):
        """Create the action space.

        Returns:
            The action space.
        """
        if self.num_goal_steps is None:
            action_shape = [4]
        else:
            action_shape = [self.num_goal_steps, 4]

        return gym.spaces.Box(
            low=-np.ones(action_shape, dtype=np.float32),
            high=np.ones(action_shape, dtype=np.float32),
            dtype=np.float32)

    def _reset(self):
        """Reset."""
        observations = super(PushEnv, self)._reset()

        self._reset_camera(
            self.camera,
            intrinsics=self.config.KINECT2.DEPTH.INTRINSICS,
            translation=self.config.KINECT2.DEPTH.TRANSLATION,
            rotation=self.config.KINECT2.DEPTH.ROTATION,
            intrinsics_noise=self.config.KINECT2.DEPTH.INTRINSICS_NOISE,
            translation_noise=self.config.KINECT2.DEPTH.TRANSLATION_NOISE,
            rotation_noise=self.config.KINECT2.DEPTH.ROTATION_NOISE)

        if self.use_recording:
            hostname = socket.gethostname().split('.')[0]
            self.recording_camera = self._create_camera(
                height=self.config.RECORDING.CAMERA.HEIGHT,
                width=self.config.RECORDING.CAMERA.WIDTH,
                intrinsics=self.config.RECORDING.CAMERA.INTRINSICS,
                translation=self.config.RECORDING.CAMERA.TRANSLATION,
                rotation=self.config.RECORDING.CAMERA.ROTATION)

            recording_tmp_dir = os.path.join('/tmp', 'recording')
            if not os.path.exists(recording_tmp_dir):
                os.makedirs(recording_tmp_dir)

            if self.task_name is None:
                recording_output_dir = os.path.join(
                        self.config.RECORDING.OUTPUT_DIR,
                        'data_collection')
            else:
                recording_output_dir = os.path.join(
                        self.config.RECORDING.OUTPUT_DIR,
                        '%s_layout%02d' % (self.task_name, self.layout_id)
                )

            if not os.path.exists(recording_output_dir):
                logger.info('Saving recorded videos to %s ...',
                            recording_output_dir)
                os.makedirs(recording_output_dir)

            if self.video_writer is not None:
                self.video_writer.release()
                shutil.copyfile(self.recording_tmp_path,
                                self.recording_output_path)

            name = '%s_%s.avi' % (
                hostname, time_utils.get_timestamp_as_string())
            resolution = (self.config.RECORDING.CAMERA.WIDTH,
                          self.config.RECORDING.CAMERA.HEIGHT)
            self.recording_tmp_path = os.path.join(
                recording_tmp_dir, name)
            self.recording_output_path = os.path.join(
                recording_output_dir, name)
            self.video_writer = cv2.VideoWriter(
                self.recording_tmp_path,
                cv2.VideoWriter_fourcc(*'XVID'),
                self.config.RECORDING.FPS,
                resolution)

        return observations

    def _reset_scene(self):
        """Reset the scene in simulation or the real world."""
        super(PushEnv, self)._reset_scene()

        # Simulation.
        if self.is_simulation:
            self.num_movable_bodies = np.random.randint(
                low=self.min_movable_bodies,
                high=self.max_movable_bodies + 1)

            self._load_movable_bodies()

            if self.layouts is not None:
                layout = self.layouts[self.layout_id]
                self._load_tiles(
                    layout.region,
                    layout.size,
                    layout.offset,
                    z_offset=0.001 - 0.025,
                    rgba=layout.region_rgba)
                if layout.goal is not None:
                    self._load_tiles(
                        layout.goal,
                        layout.size,
                        layout.offset,
                        z_offset=0.0015 - 0.025,
                        rgba=layout.goal_rgba)
        else:
            self.num_movable_bodies = self.max_movable_bodies
            logger.info('Assume there are %d movable objects on the table.',
                        self.num_movable_bodies)

        self.movable_body_mask = np.array(
            [1] * self.num_movable_bodies +
            [0] * (self.max_movable_bodies - self.num_movable_bodies))

        # Attributes
        self.attributes = {
            'num_episodes': self.num_episodes,
            'num_steps': self.num_steps,
            'layout_id': self.layout_id,
            'movable_body_mask': self.movable_body_mask,
            'is_safe': True,
            'is_effective': True,
        }

    def _load_tiles(self, centers, size, offset, z_offset=0.0, rgba=[0, 0, 0]):
        """Load tiles to the scene.

        Args:
            centers: Configuration of the tiles.
            size: Side length of each tile.
            offset: List of x-y position offsets.
            z_offset: Offset of the z dimension.
            rgba: The color of the tile.
        """
        path = self.config.SIM.TILE.PATH
        for i, center in enumerate(centers):
            position = np.array(offset) + np.array(center) * size
            height = self.table.position.z + z_offset
            position = np.array([position[0], position[1], height])
            euler = np.array([0, 0, 0])
            pose = np.array([position, euler])
            name = 'tile_%d' % i
            body = self.simulator.add_body(
                    path, pose, is_static=True, name=name)
            body.set_color(rgba=rgba, specular=[0, 0, 0])

    def _load_movable_bodies(self):
        """Load movable bodies."""
        assert self.simulator is not None

        is_valid = False
        while not is_valid:
            logger.info('Loading movable objects...')
            is_valid = True
            self.movable_bodies = []

            # Sample placements of the movable objects.
            is_target = False
            if self.layouts is None:
                movable_poses = self._sample_body_poses(
                    self.num_movable_bodies, self.movable_config)
            else:
                layout = self.layouts[self.layout_id]
                movable_poses = self._sample_body_poses_on_tiles(
                    self.num_movable_bodies,
                    self.movable_config,
                    layout)
                is_target = (layout.target is not None)

            for i in range(self.num_movable_bodies):

                if i == 0 and is_target:
                    path = random.choice(self.target_movable_paths)
                else:
                    path = random.choice(self.movable_paths)

                pose = movable_poses[i]
                scale = np.random.uniform(*self.movable_config.SCALE)
                name = 'movable_%d' % i

                # Add the body.
                body = self.simulator.add_body(path, pose, scale, name=name)

                if self.config.USE_RANDOM_RGBA:
                    r = np.random.uniform(0., 1.)
                    g = np.random.uniform(0., 1.)
                    b = np.random.uniform(0., 1.)
                    body.set_color(rgba=[r, g, b, 1.0], specular=[0, 0, 0])

                # Wait for the new body to be dropped onto the table.
                self.simulator.wait_until_stable(
                    body,
                    linear_velocity_threshold=0.1,
                    angular_velocity_threshold=0.1,
                    max_steps=500)

                # Change physical properties.
                mass = robot_env.get_config_value(self.movable_config.MASS)
                lateral_friction = robot_env.get_config_value(
                    self.movable_config.FRICTION)
                body.set_dynamics(
                    mass=mass,
                    lateral_friction=lateral_friction,
                    rolling_friction=None,
                    spinning_friction=None)
                self.movable_bodies.append(body)

            for body in self.movable_bodies:
                if body.position.z < self.table.position.z:
                    is_valid = False
                    break

            if not is_valid:
                logger.info('Invalid arrangement, reset the scene...')
                for i, body in enumerate(self.movable_bodies):
                    self.simulator.remove_body(body.name)

        logger.info('Waiting for movable objects to be stable...')
        self.simulator.wait_until_stable(self.movable_bodies)

    def _sample_body_poses(self,
                           num_samples,
                           body_config,
                           max_attemps=32):
        """Sample body poses.

        Args:
            num_samples: Number of samples.
            body_config: Configuration of the body.
            max_attemps: Maximum number of attemps to find a feasible
                placement.

        Returns:
            List of poses.
        """
        while True:
            movable_poses = []

            for i in range(num_samples):
                num_attemps = 0
                is_valid = False
                while not is_valid and num_attemps <= max_attemps:
                    pose = Pose.uniform(x=body_config.POSE.X,
                                        y=body_config.POSE.Y,
                                        z=body_config.POSE.Z,
                                        roll=body_config.POSE.ROLL,
                                        pitch=body_config.POSE.PITCH,
                                        yaw=body_config.POSE.YAW)

                    # Check if the new pose is distant from other bodies.
                    is_valid = True
                    for other_pose in movable_poses:
                        dist = np.linalg.norm(
                                pose.position[:2] - other_pose.position[:2])

                        if dist < body_config.MARGIN:
                            is_valid = False
                            num_attemps += 1
                            break

                if not is_valid:
                    logger.info('Cannot find the placement of body %d. '
                                'Start re-sampling.', i)
                    break
                else:
                    movable_poses.append(pose)

            if i == num_attemps:
                break

        return movable_poses

    def _sample_body_poses_on_tiles(self,
                                    num_samples,
                                    body_config,
                                    layout,
                                    safe_drop_height=0.2,
                                    max_attemps=32):
        """Sample tile poses on the tiles.

        Args:
            num_samples: Number of samples.
            body_config: Configuration of the body.
            layout: Configuration of the layout.
            safe_drop_height: Dropping height of the body.
            max_attemps: Maximum number of attemps to find a feasible
                placement.

        Returns:
            List of poses.
        """
        tile_size = layout.size
        tile_offset = layout.offset

        while True:
            movable_poses = []

            for i in range(num_samples):
                num_attemps = 0
                is_valid = False

                if i == 0 and layout.target is not None:
                    tile_config = layout.target
                else:
                    tile_config = layout.obstacle

                while not is_valid and num_attemps <= max_attemps:
                    num_tiles = len(tile_config)
                    tile_id = np.random.choice(num_tiles)
                    tile_center = tile_config[tile_id]
                    x_range = [
                        tile_offset[0] + (tile_center[0] - 0.5) * tile_size,
                        tile_offset[0] + (tile_center[0] + 0.5) * tile_size]
                    y_range = [
                        tile_offset[1] + (tile_center[1] - 0.5) * tile_size,
                        tile_offset[1] + (tile_center[1] + 0.5) * tile_size]
                    z = self.table_pose.position.z + safe_drop_height
                    pose = Pose.uniform(x=x_range,
                                        y=y_range,
                                        z=z,
                                        roll=[-np.pi, np.pi],
                                        pitch=[-np.pi / 2, np.pi / 2],
                                        yaw=[-np.pi, np.pi])

                    is_valid = True
                    for other_pose in movable_poses:
                        dist = np.linalg.norm(
                            pose.position[:2] - other_pose.position[:2])

                        if dist < body_config.MARGIN:
                            is_valid = False
                            num_attemps += 1
                            break

                if not is_valid:
                    logger.info('Cannot find the placement of body %d. '
                                'Start re-sampling.', i)
                    break
                else:
                    movable_poses.append(pose)

            if i == num_attemps:
                break

        return movable_poses

    def step(self, action):
        """Take a step.

        See parent class.
        """
        observation, reward, done, info = super(PushEnv, self).step(action)

        if done and reward >= self.config.SUCCESS_THRESH:
            self.num_successes += 1
            self.num_successes_by_step[self._num_steps] += 1

        if done:
            logger.info(
                'num_successes: %d, success_rate: %.3f',
                self.num_successes,
                self.num_successes / float(self._num_episodes + 1e-14),
            )
            text = ('num_successes_by_step: ' +
                    ', '.join(['%d'] * int(self.config.MAX_STEPS + 1)))

            logger.info(text, *self.num_successes_by_step)

            logger.info(
                'num_total_steps %d, '
                'unsafe: %.3f, ineffective: %.3f, useful: %.3f',
                self.num_total_steps,
                self.num_unsafe / float(self.num_total_steps + 1e-14),
                self.num_ineffective / float(self.num_total_steps + 1e-14),
                self.num_useful / float(self.num_total_steps + 1e-14))

        return observation, reward, done, info

    def _execute_action(self, action):  # NOQA
        """Execute the robot action.

        Args:
            action: A dictionary of mode and argument of the action.
        """
        self.attributes = {
            'num_episodes': self.num_episodes,
            'num_steps': self.num_steps,
            'layout_id': self.layout_id,
            'movable_body_mask': self.movable_body_mask,
            'is_safe': True,
            'is_effective': True,
        }

        waypoints = self._compute_all_waypoints(action)

        self.phase = 'initial'
        self.num_waypoints = 0
        self.interrupt = False
        self.start_status = self._get_movable_status()
        while(self.phase != 'done'):

            if self.is_simulation:
                self.simulator.step()

                if self.use_recording:
                    if (self.simulator.num_steps %
                            self.config.RECORDING.NUM_STEPS == 0):
                        self._record_screenshot()

                if not (self.simulator.num_steps %
                        self.config.SIM.STEPS_CHECK == 0):
                    continue

            # Phase transition.
            if self._is_phase_ready():
                self.phase = self._get_next_phase()
                if self.config.DEBUG and self.debug:
                    logger.info('phase: %s, num_waypoints: %d',
                                self.phase, self.num_waypoints)

                if self.is_simulation:
                    self.max_phase_steps = self.simulator.num_steps
                    if self.phase == 'motion':
                        self.max_phase_steps += (
                                self.config.SIM.MAX_MOTION_STEPS)
                    elif self.phase == 'offstage':
                        self.max_phase_steps += (
                                self.config.SIM.MAX_OFFSTAGE_STEPS)
                    else:
                        self.max_phase_steps += (
                                self.config.SIM.MAX_PHASE_STEPS)

                if self.phase == 'pre':
                    pose = waypoints[self.num_waypoints][0].copy()
                    pose.z = self.config.ARM.GRIPPER_SAFE_HEIGHT
                    self.robot.move_to_gripper_pose(pose)

                elif self.phase == 'start':
                    pose = waypoints[self.num_waypoints][0]
                    self.robot.move_to_gripper_pose(pose)

                elif self.phase == 'motion':
                    pose = waypoints[self.num_waypoints][1]
                    self.robot.move_to_gripper_pose(pose)

                elif self.phase == 'post':
                    self.num_waypoints += 1
                    pose = self.robot.end_effector.pose
                    pose.z = self.config.ARM.GRIPPER_SAFE_HEIGHT
                    self.robot.move_to_gripper_pose(pose)

                elif self.phase == 'offstage':
                    self.robot.move_to_joint_positions(
                            self.config.ARM.OFFSTAGE_POSITIONS)

            self.interrupt = False

            if self._check_singularity():
                self.interrupt = True

            if not self._check_safety():
                self.interrupt = True
                self.attributes['is_safe'] = False

            if self.interrupt:
                if self.phase == 'done':
                    self._done = True
                    break

        if self.is_simulation:
            self.simulator.wait_until_stable(self.movable_bodies)

        # Update attributes.
        self.end_status = self._get_movable_status()
        self.attributes['is_effective'] = self._check_effectiveness()

        self.num_total_steps += 1
        self.num_unsafe += int(not self.attributes['is_safe'])
        self.num_ineffective += int(not self.attributes['is_effective'])
        self.num_useful += int(self.attributes['is_safe'] and
                               self.attributes['is_effective'])

    def _compute_all_waypoints(self, action):
        """Convert action of a single step or multiple steps to waypoints.

        Args:
            action: Action of a single step or actions of multiple steps.

        Returns:
            List of waypoints of a single step or multiple steps.
        """
        if self.num_goal_steps is None:
            waypoints = [self._compute_waypoints(action)]
        else:
            waypoints = [
                self._compute_waypoints(action[i])
                for i in range(self.num_goal_steps)]
        return waypoints

    def _compute_waypoints(self, action):
        """Convert action of a single step to waypoints.

        Args:
            action: The action of a single step.

        Returns:
            List of waypoints of a single step.
        """
        action = np.reshape(action, [2, 2])
        start = action[0, :]
        motion = action[1, :]

        # Start.
        x = start[0] * self.start_range[0] + self.start_offset[0]
        y = start[1] * self.start_range[1] + self.start_offset[1]
        z = self.start_z
        angle = 0.0
        start = Pose(
            [[x, y, z], [np.pi, 0, (angle + np.pi) % (2 * np.pi) - np.pi]]
        )

        # End.
        delta_x = motion[0] * self.config.ACTION.MOTION.TRANSLATION_X
        delta_y = motion[1] * self.config.ACTION.MOTION.TRANSLATION_Y
        x = x + delta_x
        y = y + delta_y
        x = np.clip(x, self.cspace.low[0], self.cspace.high[0])
        y = np.clip(y, self.cspace.low[1], self.cspace.high[1])
        end = Pose(
            [[x, y, z], [np.pi, 0, (angle + np.pi) % (2 * np.pi) - np.pi]]
        )

        waypoints = [start, end]
        return waypoints

    def _get_next_phase(self):
        """Get the next phase of the current phase.

        Returns:
            The next phase as a string variable.
        """
        if self.phase in self.phase_list:
            if self.interrupt:
                if self.phase not in ['post', 'offstage', 'done']:
                    return 'post'

            i = self.phase_list.index(self.phase)
            if i >= len(self.phase_list):
                raise ValueError('phase %r does not have a next phase.')
            else:
                if self.num_goal_steps is not None:
                    if (self.phase == 'post' and
                            self.num_waypoints < self.num_goal_steps):
                        return 'pre'

                return self.phase_list[i + 1]
        else:
            raise ValueError('Unrecognized phase: %r' % self.phase)

    def _is_phase_ready(self):
        """Check if the current phase is ready.

        Returns:
            The boolean value indicating if the current phase is ready.
        """
        if self.interrupt:
            return True

        if self.is_simulation:
            if self.robot.is_limb_ready() and self.robot.is_gripper_ready():
                self.robot.arm.reset_targets()
                return True

            if self.max_phase_steps is None:
                return True
            if self.simulator.num_steps >= self.max_phase_steps:
                if self.config.DEBUG:
                    logger.warning('Phase %s timeout.', self.phase)
                self.robot.arm.reset_targets()
                return True

            return False

        else:
            return True

    def _check_singularity(self):
        """Check singularity.

        Returns:
            True if it is in simulation and the arm contacts the table, False
                otherwise.
        """
        if self.phase != 'motion':
            return False

        if self.is_simulation:
            if self.simulator.check_contact(self.robot.arm, self.table):
                if self.config.DEBUG:
                    logger.warning('Arm collides with the table.')
                return True

        return False

    def _check_safety(self):
        """Check if the action is safe.

        Returns:
            True if all the safty conditions are satisfied, False otherwise.

        """
        if self.is_simulation:
            if self.phase == 'pre':
                if self.simulator.check_contact(
                        self.robot.arm, self.movable_bodies):
                    logger.warning('Unsafe action: Bodies stuck on robots.')
                    return False

            if self.phase == 'start':
                if self.simulator.check_contact(
                        self.robot.arm, self.movable_bodies):
                    dist = self.robot.end_effector.position.z - self.start_z
                    if abs(dist) <= 0.01:
                        return True
                    logger.warning('Unsafe action: Bodies stuck on robots.')
                    return False

            if self.phase == 'done':
                if self.simulator.check_contact(
                        self.robot.arm, self.movable_bodies):
                    logger.warning('Unsafe action: Bodies stuck on robots.')
                    return False

                for body in self.movable_bodies:
                    if (
                            body.position.x < self.table_workspace.low[0] or
                            body.position.x > self.table_workspace.high[0] or
                            body.position.y < self.table_workspace.low[1] or
                            body.position.y > self.table_workspace.high[1]):
                        logger.warning('Unsafe action: Body left table.')
                        return False
        else:
            has_collided = self.robot.arm.has_collided()
            return not has_collided

        return True

    def _check_effectiveness(self):
        """Check if the action is effective.

        Returns:
            True if at least one of the object has a translation or orientation
                larger than the threshold, False otherwise.
        """
        if self.is_simulation:
            delta_position = np.linalg.norm(
                    self.end_status[0] - self.start_status[0], axis=-1)
            delta_position = np.sum(delta_position)

            delta_angle = self.end_status[1] - self.start_status[1]
            delta_angle = (delta_angle + np.pi) % (2 * np.pi) - np.pi
            delta_angle = np.abs(delta_angle)
            delta_angle = np.sum(delta_angle)

            if (delta_position <= self.config.ACTION.MIN_DELTA_POSITION and
                    delta_angle <= self.config.ACTION.MIN_DELTA_ANGLE):
                if self.config.DEBUG:
                    logger.warning('Ineffective action.')
                return False

        return True

    def _get_movable_status(self):
        """Get the status of the movable objects.

        Returns:
            Concatenation of the positions and Euler angles of all objects in
                the simulation, None in the real world.
        """
        if self.is_simulation:
            positions = [body.position for body in self.movable_bodies]
            angles = [body.euler[2] for body in self.movable_bodies]
            return [np.stack(positions, axis=0), np.stack(angles, axis=0)]

        return None

    def visualize(self, action, info):  # NOQA
        """Visualize the action.

        Args:
            action: A selected action.
            info: The policy infomation.
        """
        num_info_samples = self.config.NUM_INFO_SAMPLES

        # Reset.
        images = self.camera.frames()
        rgb = images['rgb']
        self.ax.cla()
        self.ax.imshow(rgb)

        if 'position' in self.obs_data:
            states = self.obs_data['position'][..., :2]
        else:
            point_clouds = self.obs_data['point_cloud']
            states = np.mean(point_clouds[..., :2], axis=-2)

        if 'pred_goals' in info:
            pred_states = info['pred_goals']
            terminations = info['goal_terminations']
            max_plots = min(num_info_samples, self.config.MAX_STATE_PLOTS)
            for i in range(max_plots):
                self._plot_pred_states(self.ax,
                                       states,
                                       pred_states[i],
                                       terminations[i],
                                       c='gold',
                                       alpha=0.8)

            t = 0
            self._plot_states_distribution(
                self.ax,
                pred_states[:, t],
                terminations[:, t],
                body_index=0,
                num_plots=num_info_samples,
                c='gold',
                alpha=0.5)

        if 'pred_states' in info:
            pred_states = info['pred_states']
            terminations = info['terminations']
            max_plots = min(num_info_samples, self.config.MAX_STATE_PLOTS)
            for i in range(max_plots):
                self._plot_pred_states(self.ax,
                                       states,
                                       pred_states[i],
                                       terminations[i],
                                       c='lawngreen',
                                       alpha=0.5)

        if 'actions' in info:
            actions = info['actions']

            max_plots = min(num_info_samples, self.config.MAX_ACTION_PLOTS)
            for t in range(self.num_goal_steps):
                if t == 0:
                    c = 'royalblue'
                elif t == 1:
                    c = 'deepskyblue'
                elif t == 2:
                    c = 'azure'
                else:
                    raise ValueError

                for i in range(max_plots):
                    if i == 0:
                        linewidth = 3.0
                    else:
                        linewidth = 1.0
                    waypoints = self._compute_waypoints(actions[i, t])
                    self._plot_waypoints(self.ax,
                                         waypoints,
                                         linewidth=linewidth,
                                         c=c,
                                         alpha=0.5)

        # Plot waypoints in simulation.
        if self.simulator and self.num_goal_steps is None:
            waypoints = self._compute_waypoints(actions)
            self._plot_waypoints_in_simulation(waypoints)

        plt.draw()
        plt.pause(1e-3)

    def _plot_waypoints(self,
                        ax,
                        waypoints,
                        linewidth=1.0,
                        c='blue',
                        alpha=1.0):
        """Plot waypoints.

        Args:
            ax: An instance of Matplotlib Axes.
            waypoints: List of waypoints.
            linewidth: Width of the lines connecting waypoints.
            c: Color of the lines connecting waypoints.
            alpha: Alpha value of the lines connecting waypoints.
        """
        z = self.table_pose.position.z

        p1 = None
        p2 = None
        for i, waypoint in enumerate(waypoints):
            point1 = waypoint.position
            point1 = np.array([point1[0], point1[1], z])
            p1 = self.camera.project_point(point1)
            if i == 0:
                ax.scatter(p1[0], p1[1],
                           c=c, alpha=alpha, s=2.0)
            else:
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                        c=c, alpha=alpha, linewidth=linewidth)
            p2 = p1

    def _plot_pred_states(self,
                          ax,
                          states,
                          pred_states,
                          terminations,
                          c='lawngreen',
                          alpha=1.0):
        """Plot predicted states.

        Args:
            ax: An instance of Matplotlib Axes.
            states: The current states.
            pred_states: The predicted states.
            terminations: Termination flags of the predicted states.
            c: Color of the body centers.
            alpha: Alpha value of the body centers
        """
        num_steps = pred_states.shape[0]
        z = self.table_pose.position.z

        for j in range(self.num_movable_bodies):
            points1 = np.array(list(states[j]) + [z])
            p1 = self.camera.project_point(points1)

            for t in range(num_steps):
                points2 = np.array(list(pred_states[t, j]) + [z])
                p2 = self.camera.project_point(points2)

                # if np.linalg.norm(points2 - points1) < 0.1:
                #     continue

                ax.arrow(p1[0], p1[1], p2[0] - p1[0], p2[1] - p1[1],
                         head_width=10, head_length=10,
                         fc=c, ec=c, alpha=alpha,
                         zorder=100)
                points1 = points2
                p1 = p2

                # Stop plotting after termination.
                if terminations[t]:
                    break

    def _plot_states_distribution(self,
                                  ax,
                                  pred_states,
                                  terminations,
                                  body_index,
                                  num_plots,
                                  c='lawngreen',
                                  alpha=1.0):
        """Plot the distribution of predicted states.

        Args:
            ax: An instance of Matplotlib Axes.
            pred_states: The predicted states.
            terminations: Termination flags of the predicted states.
            body_index: The index of the body to plot.
            num_plots: Number of bodies to plot.
            c: Color of the body centers.
            alpha: Alpha value of the body centers
        """
        z = self.table_pose.position.z

        for i in range(num_plots):
            position = pred_states[i, body_index]
            points = np.array(list(position) + [z])
            p = self.camera.project_point(points)

            if terminations[i]:
                c_i = 'r'
            else:
                c_i = c

            ax.scatter(p[0], p[1], c=c_i, alpha=alpha)

    def _plot_waypoints_in_simulation(self, waypoints):
        """Plot waypoints in simulation.

        Args:
            waypoints: List of waypoints.
        """
        self.simulator.clear_visualization()

        for i, waypoints_i in enumerate(waypoints):
            for j in range(len(waypoints_i)):
                waypoint = waypoints_i[j]
                if j == 0:
                    text = '%d' % (i)
                else:
                    text = None
                self.simulator.plot_pose(waypoint, 0.05, text)

            for j in range(1, len(waypoints_i)):
                self.simulator.plot_line(waypoints_i[j - 1].position,
                                         waypoints_i[j].position)

    def _record_screenshot(self):
        """Record a screenshot and save to file."""
        assert self.simulator is not None
        images = self.recording_camera.frames()
        image = images['rgb']
        image = image.astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        self.video_writer.write(image)
