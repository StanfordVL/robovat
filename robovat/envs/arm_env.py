"""The environment of robot arm."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from robovat.envs import robot_env
from robovat.math import Pose
from robovat.robots import sawyer
from robovat.perception.camera import Kinect2
from robovat.simulation.camera import BulletCamera


class ArmEnv(robot_env.RobotEnv):
    """The environment of robot arm."""

    def __init__(self,
                 simulator=None,
                 config=None,
                 debug=False):
        """Initialize.

        See parent class.
        """
        super(ArmEnv, self).__init__(
            simulator=simulator,
            config=config,
            debug=debug)

        # Scene.
        self.robot = None
        self.ground = None
        self.table = None
        self.table_pose = None

    def _create_camera(self,
                       height,
                       width,
                       intrinsics,
                       translation,
                       rotation):
        """Create a camera instance.

        Args:
            height: The height of the camera image.
            width: The width of the camera image.
            intrinsics: Camera intrinsics matrix.
            translation: Translation of the camera extrinsics.
            rotation: Rotation of the camera extrinsics.

        Returns:
            The camera instance.
        """
        if self.is_simulation:
            camera = BulletCamera(
                simulator=self.simulator,
                height=height,
                width=width)
        else:
            camera = Kinect2(
                height=height,
                width=width)

        intrinsics = np.copy(intrinsics).astype(np.float32)
        translation = np.copy(translation).astype(np.float32)
        rotation = np.copy(rotation).astype(np.float32)
        camera.set_calibration(intrinsics, translation, rotation)

        return camera

    def _reset(self):
        """Reset the environment in simulation or the real world."""
        self._reset_scene()
        self._reset_robot()

    def _reset_scene(self):
        """Reset the scene in simulation or the real world."""
        self.table_pose = Pose(self.config.SIM.TABLE.POSE)
        self.table_pose.position.z += np.random.uniform(
            *self.config.TABLE.HEIGHT_RANGE)

        if self.is_simulation:
            self.ground = self.simulator.add_body(self.config.SIM.GROUND.PATH,
                                                  self.config.SIM.GROUND.POSE,
                                                  is_static=True,
                                                  name='ground')

            self.table = self.simulator.add_body(self.config.SIM.TABLE.PATH,
                                                 self.table_pose,
                                                 is_static=True,
                                                 name='table')

            if self.config.SIM.WALL.USE:
                self.wall = self.simulator.add_body(self.config.SIM.WALL.PATH,
                                                    self.config.SIM.WALL.POSE,
                                                    is_static=True,
                                                    name='wall')

    def _reset_robot(self):
        """Reset the robot in simulation or the real world."""
        self.robot = sawyer.factory(
            simulator=self.simulator,
            config=self.config.SIM.ARM.CONFIG)
        self.robot.move_to_joint_positions(
            self.config.ARM.OFFSTAGE_POSITIONS)

    def _reset_camera(self,
                      camera,
                      intrinsics,
                      translation,
                      rotation,
                      intrinsics_noise=None,
                      translation_noise=None,
                      rotation_noise=None):
        """Reset camera.

        If camera is in the real world, reset the intrinsics and extrinsics to
        be the input values; if camera is in simulation, reset the intrinsics
        and extrinsics with optional random noises.

        Args:
            camera: The camera instance.
            intrinsics: Camera intrinsics matrix.
            translation: Translation of the camera extrinsics.
            rotation: Rotation of the camera extrinsics.
            intrinsics_noise: Range of random noise of the camera intrinsics.
            translation_noise: Range of random noise of the camera translation.
            rotation_noise: Range of random noise of the camera rotation.
        """
        intrinsics = np.copy(intrinsics).astype(np.float32)
        translation = np.copy(translation).astype(np.float32)
        rotation = np.copy(rotation).astype(np.float32)

        if self.is_simulation:
            if intrinsics_noise is not None:
                intrinsics += np.random.uniform(
                    -np.array(intrinsics_noise),
                    np.array(intrinsics_noise))

            if translation_noise is not None:
                translation += np.random.uniform(
                    -np.array(translation_noise),
                    np.array(translation_noise))

            if rotation_noise is not None:
                rotation += np.random.uniform(
                    -np.array(rotation_noise),
                    np.array(rotation_noise))

        camera.set_calibration(intrinsics, translation, rotation)
