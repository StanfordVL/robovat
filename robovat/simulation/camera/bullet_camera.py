"""The class for camera utilities of Bullet.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pybullet

from robovat.math import Orientation
from robovat.perception.camera import Camera


# TODO(kuanfang): If the Kinect use regirobovation, then the shape of the rgb
# image matches that of the depth image.
FOV = 60
NEAR_PLANE = 0.02
FAR_PLANE = 100
IMAGE_HEIGHT = 1080
IMAGE_WIDTH = 1920
DEPTH_HEIGHT = 424
DEPTH_WIDTH = 512
# DEPTH_HEIGHT = 428
# DEPTH_WIDTH = 482


def intrinsic_to_projection_matrix(intrinsics, height, width, near, far,
                                   upside_down=True):
    """Convert the camera intrinsics to the projection matrix.

    Takes a Hartley-Zisserman intrinsic matrix and returns a Bullet/OpenGL
    style projection matrix. We pad with zeros on right and bottom and a 1
    in the corner.

    Uses algorithm found at:
    https://strawlab.org/2011/11/05/augmented-reality-with-OpenGL#note-about-image-coordinates

    Args:
        intrinsics: The camera intrinsincs matrix.
        height: The image height.
        width: The image width.
        near: The distance to the near plane.
        far: The distance to the far plane.

    Returns:
        projection_matrix: The projection matrix.
    """
    projection_matrix = np.empty((4, 4), dtype=np.float32)

    f_x = intrinsics[0, 0]
    f_y = intrinsics[1, 1]
    x_0 = intrinsics[0, 2]
    y_0 = intrinsics[1, 2]
    s = intrinsics[0, 1]

    if upside_down:
        x_0 = width - x_0
        y_0 = height - y_0

    projection_matrix[0, 0] = 2 * f_x / width
    projection_matrix[0, 1] = -2 * s / width
    projection_matrix[0, 2] = (width - 2 * x_0) / width
    projection_matrix[0, 3] = 0

    projection_matrix[1, 0] = 0
    projection_matrix[1, 1] = 2 * f_y / height
    projection_matrix[1, 2] = (-height + 2 * y_0) / height
    projection_matrix[1, 3] = 0

    projection_matrix[2, 0] = 0
    projection_matrix[2, 1] = 0
    projection_matrix[2, 2] = (-far - near) / (far - near)
    projection_matrix[2, 3] = -2 * far * near / (far - near)

    projection_matrix[3, 0] = 0
    projection_matrix[3, 1] = 0
    projection_matrix[3, 2] = -1
    projection_matrix[3, 3] = 0

    projection_matrix = list(projection_matrix.transpose().flatten())

    return projection_matrix


def extrinsic_to_view_matrix(translation, rotation, distance):
    """Convert the camera extrinsics to the view matrix.

    The function takes HZ-style rotation matrix R and translation matrix t
    and converts them to a Bullet/OpenGL style view matrix. the derivation
    is pretty simple if you consider x_camera = R * x_world + t.

    Args:
        distance: The distance from the camera to the focus.

    Returns:
        view_matrix: The view matrix.
    """
    # The camera position in the world frame.
    camera_position = rotation.T.dot(-translation)

    # The focus in the world frame.
    focus = rotation.T.dot(np.array([0, 0, distance]) - translation)

    # The up vector is the Y-axis of the camera in the world frame.
    up_vector = rotation.T.dot(np.array([0, 1, 0]))

    # Compute the view matrix.
    view_matrix = pybullet.computeViewMatrix(
        cameraEyePosition=camera_position,
        cameraTargetPosition=focus,
        cameraUpVector=up_vector)

    return view_matrix


class BulletCamera(Camera):
    """The simulated camera of Bullet.
    """

    def __init__(self,
                 simulator,
                 height=DEPTH_HEIGHT,
                 width=DEPTH_WIDTH,
                 intrinsics=None,
                 translation=None,
                 rotation=None,
                 crop=None,
                 near=NEAR_PLANE,
                 far=FAR_PLANE,
                 distance=1.0,
                 upside_down=True):
        """Initialize.

        If there are extrinsics of the form RGB_intrisics.npy,
        robot_RGB_rotation.npy, robot_RGB_translation.npy in some
        directory, use that as an argument which takes higher priority than
        the view_matrix and projection_matrix.

        Args:
            simulator: The simulator to render, as an instance of World.
            height: The height of the image.
            width: The width of the image.
            intrinsics: The intrinsics matrix.
            translation: The translation vector.
            rotation: The rotation matrix.
            crop: The cropping box as [y1, x1, y2, x2].
            near: The distance to the near plane.
            far: The distance to the far plane.
            distance: The distance from the camera to the object.
            upside_down: To match the real-world Kinect2 camera, we optionally
                rotate the rendered images by 180 degree.
        """
        self._simulator = simulator
        self._near = near
        self._far = far
        self._distance = distance
        self._upside_down = upside_down

        self._render_height = height
        self._render_width = width

        super(BulletCamera, self).__init__(
            height=height,
            width=width,
            intrinsics=intrinsics,
            translation=translation,
            rotation=rotation,
            crop=crop)

    @property
    def simulator(self):
        return self._simulator

    @property
    def view_matrix(self):
        return self._view_matrix

    @property
    def projection_matrix(self):
        return self._projection_matrix

    def start(self):
        """Starts the camera stream.
        """
        pass

    def _frames(self):
        """Render the world at the current time step.

        Returns:
            A dictionary of RGB image, depth image and segmentation image.
            'rgb': The RGB image as an uint8 np array of [width, height, 3].
            'depth': The depth image as a float32 np array of [width, height].
            'segmask': The segmentation mask image as an uint8 np array of
                [width, height].
        """
        _, _, rgba, depth, segmask = pybullet.getCameraImage(
            height=self._render_height,
            width=self._render_width,
            viewMatrix=self._view_matrix,
            projectionMatrix=self._projection_matrix,
            physicsClientId=self.simulator.physics.uid)

        rgba = np.array(rgba).astype('uint8')
        rgba = rgba.reshape((self._render_height, self._render_width, 4))
        rgb = rgba[:, :, :3]

        depth = np.array(depth).astype('float32')
        depth = depth.reshape((self._render_height, self._render_width))

        segmask = np.array(segmask).astype('uint8')
        segmask = segmask.reshape((self._render_height, self._render_width))

        # TODO(kuanfang): This rotate the image so that it is consistent with
        # the real-world Kinect. Not sure if this is actually necessary.
        if self._upside_down:
            rgb = rgb[::-1, ::-1, :]
            depth = depth[::-1, ::-1]
            segmask = segmask[::-1, ::-1]

        # TODO(kuanfang): This is a fix for the depth image rendering in
        # pybullet, by following:
        # https://stackoverflow.com/questions/6652253/getting-the-true-z-value-from-the-depth-buffer
        z_b = depth
        z_n = 2.0 * z_b - 1.0
        z_e = (2.0 * self._near * self._far /
               (self._far + self._near - z_n * (self._far - self._near)))
        depth = z_e

        return {
            'rgb': rgb,
            'depth': depth,
            'segmask': segmask,
        }

    def set_calibration(self, intrinsics, translation, rotation):
        """Set the camera calibration data.

        Args:
            intrinsics: The intrinsics matrix.
            translation: The translation vector.
            rotation: The rotation matrix.
        """
        if intrinsics is not None:
            intrinsics = np.array(intrinsics).reshape((3, 3))
            self._projection_matrix = intrinsic_to_projection_matrix(
                intrinsics, self._render_height, self._render_width,
                self._near, self._far)

        if translation is not None and rotation is not None:
            translation = np.array(translation).reshape((3,))
            rotation = Orientation(rotation).matrix3
            self._view_matrix = extrinsic_to_view_matrix(
                translation, rotation, self._distance)

        super(BulletCamera, self).set_calibration(
            intrinsics, translation, rotation)
