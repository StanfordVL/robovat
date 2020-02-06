"""Grasp classes.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from robovat.math import get_transform
from robovat.math import Pose


class Grasp2D(object):
    """Parallel-jaw grasp in image space."""

    def __init__(self, center, angle, depth, width=0.0, camera=None):
        """Initialize.

        Args:
            center: Point (x, y) in image space.
            angle: Grasp axis angle with the camera x-axis.
            depth: Depth of the grasp center in 3D space.
            width: Distance between the jaws in meters.
            camera: The camera sensor for projection and deprojection.
        """
        self.center = center
        self.angle = angle
        self.depth = depth
        self.width = width
        self.camera = camera

    @property
    def axis(self):
        """Grasp axis.
        """
        return np.array([np.cos(self.angle), np.sin(self.angle)])

    @property
    def endpoints(self):
        """Grasp endpoints.
        """
        p1 = self.center - (float(self.width_pixel) / 2) * self.axis
        p2 = self.center + (float(self.width_pixel) / 2) * self.axis
        return p1, p2

    @property
    def width_pixel(self):
        """Width in pixels.
        """
        if self.camera is None:
            raise ValueError('Must specify camera intrinsics to compute '
                             'gripper width in 3D space.')

        # form the jaw locations in 3D space at the given depth
        p1 = np.array([0, 0, self.depth])
        p2 = np.array([self.width, 0, self.depth])

        # project into pixel space
        u1 = self.camera.project_point(p1, is_world_frame=False)
        u2 = self.camera.project_point(p2, is_world_frame=False)

        return np.linalg.norm(u1 - u2)

    @property
    def pose(self):
        """Computes the 3D pose of the grasp relative to the camera.

        If an approach direction is not specified then the camera
        optical axis is used.

        Returns:
            The pose of the grasp in the camera frame.
        """
        if self.camera is None:
            raise ValueError('Must specify camera intrinsics to compute 3D '
                             'grasp pose.')

        # Compute 3D grasp center in camera basis.
        grasp_center_camera = self.camera.deproject_pixel(
                self.center, self.depth, is_world_frame=False)

        # Compute 3D grasp axis in camera basis.
        grasp_axis_image = self.axis
        grasp_axis_image = grasp_axis_image / np.linalg.norm(grasp_axis_image)
        grasp_axis_camera = np.array(
            [grasp_axis_image[0], grasp_axis_image[1], 0])
        grasp_axis_camera = (
            grasp_axis_camera / np.linalg.norm(grasp_axis_camera))

        # Aligned with camera Z axis.
        grasp_x_camera = np.array([0, 0, 1])
        grasp_y_camera = grasp_axis_camera
        grasp_z_camera = np.cross(grasp_x_camera, grasp_y_camera)
        grasp_x_camera = np.cross(grasp_z_camera, grasp_y_camera)
        grasp_rot_camera = np.array(
            [grasp_x_camera, grasp_y_camera, grasp_z_camera]).T

        if np.linalg.det(grasp_rot_camera) < 0:
            # Fix possible reflections due to SVD.
            grasp_rot_camera[:, 0] = -grasp_rot_camera[:, 0]

        pose = Pose([grasp_center_camera, grasp_rot_camera])

        return pose

    @property
    def vector(self):
        """Returns the feature vector for the grasp.

        v = [x1, y1, x2, y2, depth], where p1 = [x1, y1] and p2 = [x2, y2] are
        the jaw locations in image space.
        """
        p1, p2 = self.endpoints
        return np.r_[p1, p2, self.depth]

    @staticmethod
    def from_vector(value, camera=None):
        """Creates a Grasp2D instance from a feature and additional parameters.

        Args:
            value: Feature vector.
            width: Grasp opening width, in meters.
            camera: The camera sensor for projection and deprojection.
        """
        # Read feature vector.
        p1 = value[:2]
        p2 = value[2:4]
        depth = value[4]

        # project into pixel space
        u1 = np.array([p1[1], p1[0]])
        u2 = np.array([p2[1], p2[0]])
        point_1 = camera.deproject_pixel(u1, depth, is_world_frame=False)
        point_2 = camera.deproject_pixel(u2, depth, is_world_frame=False)
        width = np.linalg.norm(point_1 - point_2)

        # Compute center and angle.
        center = (p1 + p2) / 2
        axis = p2 - p1
        angle = np.arctan2(axis[1], axis[0])

        return Grasp2D(center, angle, depth, width, camera)

    def as_4dof(self):
        """Computes the 4-DOF pose of the grasp in the world frame.

        Returns:
            The 4-DOF gripper pose in the world.
        """
        angle = (self.angle + np.pi / 2)
        grasp_pose_in_camera = Pose([self.pose.position, [0, 0, angle]])
        grasp_pose_in_world = get_transform(source=self.camera.pose).transform(
            grasp_pose_in_camera)

        x, y, z = grasp_pose_in_world.position
        angle = grasp_pose_in_world.euler[2]

        return [x, y, z, angle]
