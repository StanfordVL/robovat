"""Camera observation.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import gym
import numpy as np
import matplotlib.pyplot as plt

from robovat.observations import observation
from robovat.perception import point_cloud_utils as pc_utils
from robovat.utils.logging import logger


INF = 2**32 - 1


class ClickController(object):

    def __init__(self, obs):
        self.obs = obs

    def __call__(self, event):
        pixel = [event.xdata, event.ydata]
        z = self.obs.depth[int(pixel[1]), int(pixel[0])]
        position = self.obs.camera.deproject_pixel(pixel, z)
        self.obs.target_position = position


class CameraObs(observation.Observation):
    """Camera observation."""

    def __init__(self,
                 camera,
                 modality,
                 max_visible_distance_m=None,
                 name=None):
        """Initialize."""
        self.name = name or 'camera_obs'
        self.camera = camera
        self.modality = modality
        self.max_visible_distance_m = max_visible_distance_m or INF

        self.env = None

    def initialize(self, env):
        self.env = env
        self.camera.start()

    def on_episode_start(self):
        """Called at the start of each episode."""
        self.camera.reset()

    def get_gym_space(self):
        """Returns gym space of this observation."""
        height = self.camera.height
        width = self.camera.width

        if self.modality == 'rgb':
            shape = (height, width, 3)
            return gym.spaces.Box(0, 255, shape, dtype=np.uint8)
        elif self.modality == 'depth':
            shape = (height, width, 1)
            return gym.spaces.Box(0.0, self.max_visible_distance_m, shape,
                                  dtype=np.float32)
        elif self.modality == 'segmask':
            shape = (height, width, 1)
            return gym.spaces.Box(0, int(INF), shape, dtype=np.uint32)
        else:
            raise ValueError('Unrecognized modality: %r.' % (self.modality))

    def get_observation(self):
        """Returns the observation data of the current step."""
        images = self.camera.frames()

        if self.modality == 'rgb':
            image = images[self.modality]
        elif self.modality == 'depth' or self.modality == 'segmask':
            image = images[self.modality]
            image = image[:, :, np.newaxis]
        else:
            raise ValueError('Unrecognized modality: %r.' % (self.modality))

        return image


class PointCloudObs(CameraObs):
    """Point cloud observation."""

    def __init__(self,
                 camera,
                 num_points,
                 remove_table=True,
                 max_visible_distance_m=None,
                 name=None):
        """Initialize."""
        self.name = name or 'camera_obs'
        self.camera = camera
        self.num_points = num_points
        self.remove_table = remove_table
        self.max_visible_distance_m = max_visible_distance_m or INF

        self.env = None

    def get_gym_space(self):
        """Returns gym space of this observation."""
        shape = (self.num_points, 3)
        return gym.spaces.Box(-INF, INF, shape, dtype=np.float32)

    def get_observation(self):
        """Returns the observation data of the current step."""
        images = self.camera.frames()
        point_cloud = self.camera.deproject_depth_image(images['depth'])

        if self.remove_table:
            point_cloud = pc_utils.remove_table(point_cloud)

        point_cloud = pc_utils.downsample(
            point_cloud, num_samples=self.num_points)

        return point_cloud


class SegmentedPointCloudObs(CameraObs):
    """Point cloud observation."""

    def __init__(self,
                 camera,
                 num_points,
                 num_bodies=None,
                 crop_min=None,
                 crop_max=None,
                 max_visible_distance_m=None,
                 confirm_target=False,
                 name=None):
        """Initialize."""
        self.name = name or 'camera_obs'
        self.camera = camera
        self.num_points = num_points
        self.num_bodies = num_bodies

        if crop_max is not None and crop_min is not None:
            self.crop_max = np.array(crop_max)[np.newaxis, :]
            self.crop_min = np.array(crop_min)[np.newaxis, :]
        else:
            self.crop_max = None
            self.crop_min = None

        self.max_visible_distance_m = max_visible_distance_m or INF
        self.confirm_target = confirm_target

        self.env = None

        if self.confirm_target:
            fig = plt.figure(figsize=(10, 5))
            ax = fig.add_subplot(111)
            self.ax = ax
            self.fig = fig

            self.target_position = None
            self.depth = None

            onclick = ClickController(obs=self)
            fig.canvas.mpl_connect('button_press_event', onclick)
            plt.ion()
            plt.show()

    def on_episode_start(self):
        """Called at the start of each episode."""
        self.camera.reset()
        if self.env.simulator:
            self.body_ids = [body.uid for body in self.env.movable_bodies]

    def get_gym_space(self):
        """Returns gym space of this observation."""
        shape = (self.num_bodies, self.num_points, 3)
        return gym.spaces.Box(-INF, INF, shape, dtype=np.float32)

    def get_observation(self):
        """Returns the observation data of the current step."""
        images = self.camera.frames()
        image = images['rgb']
        depth = images['depth']
        point_cloud = self.camera.deproject_depth_image(depth)

        # Crop.
        if self.crop_max is not None and self.crop_min is not None:
            crop_mask = np.logical_and(
                np.all(point_cloud >= self.crop_min, axis=-1),
                np.all(point_cloud <= self.crop_max, axis=-1))
            point_cloud = point_cloud[crop_mask]

        # Segment.
        if self.env.simulator:
            segmask = images['segmask']
            segmask = segmask.flatten()
            segmask = pc_utils.convert_segment_ids(segmask, self.body_ids)
            point_cloud = pc_utils.group_by_labels(
                point_cloud, segmask, self.num_bodies, self.num_points)
        else:
            point_cloud = pc_utils.remove_table(point_cloud)
            segmask = pc_utils.cluster(
                point_cloud, num_clusters=self.num_bodies, method='dbscan')
            point_cloud = point_cloud[segmask != -1]
            segmask = pc_utils.cluster(
                point_cloud, num_clusters=self.num_bodies)
            point_cloud = pc_utils.group_by_labels(
                point_cloud, segmask, self.num_bodies, self.num_points)

        # Confirm target.
        if self.confirm_target:
            # Click the target position.
            self.target_position = None
            self.depth = depth
            self.ax.cla()
            self.ax.imshow(image)
            logger.info('Please click the target object...')
            while self.target_position is None:
                plt.pause(1e-3)
            logger.info('Target Position: %r', self.target_position)

            # Exchange the target object with the first object.
            centers = np.mean(point_cloud, axis=1)
            dists = np.linalg.norm(
                centers - self.target_position[np.newaxis, :], axis=-1)
            target_id = dists.argmin()
            if target_id != 0:
                tmp = copy.deepcopy(point_cloud)
                point_cloud[0, :] = tmp[target_id, :]
                point_cloud[target_id, :] = tmp[0, :]

            # Show the segmented point cloud.
            pc_utils.show2d(point_cloud, self.camera, self.ax, image=image)

        return point_cloud


class CameraIntrinsicsObs(observation.Observation):
    """Camera intrinsics observation.

    Returns the camera instrincs matrix.
    """

    def __init__(self,
                 camera,
                 name=None):
        """Initialize."""
        self.name = name or 'camera_intrinsics_obs'
        self.camera = camera

        self.env = None

    def get_gym_space(self):
        """Returns gym space of this observation."""
        return gym.spaces.Box(
            low=-INF * np.ones((3, 3)),
            high=INF * np.ones((3, 3)),
            dtype=np.float32)

    def get_observation(self):
        """Returns the observation data of the current step."""
        return self.camera.intrinsics


class CameraTranslationObs(observation.Observation):
    """Camera translation observation.

    Returns the translation matrix of the camera extrinsics.
    """

    def __init__(self,
                 camera,
                 name=None):
        """Initialize."""
        self.name = name or 'camera_translation_obs'
        self.camera = camera

        self.env = None

    def get_gym_space(self):
        """Returns gym space of this observation."""
        return gym.spaces.Box(
            low=-INF * np.ones((3,)),
            high=INF * np.ones((3,)),
            dtype=np.float32)

    def get_observation(self):
        """Returns the observation data of the current step."""
        return self.camera.translation


class CameraRotationObs(observation.Observation):
    """Camera translation observation.

    Returns the translation matrix of the camera extrinsics.
    """

    def __init__(self,
                 camera,
                 name=None):
        """Initialize."""
        self.name = name or 'camera_rotation_obs'
        self.camera = camera

        self.env = None

    def get_gym_space(self):
        """Returns gym space of this observation."""
        # TODO: Correct the space limits.
        return gym.spaces.Box(
            low=-INF * np.ones((3, 3)),
            high=INF * np.ones((3, 3)),
            dtype=np.float32)

    def get_observation(self):
        """Returns the observation data of the current step."""
        return self.camera.rotation
