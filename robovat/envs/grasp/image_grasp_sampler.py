"""Image grasp samplers.

Adapted from Jeff Mahler's code.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta, abstractmethod

import numpy as np
import scipy.misc
import scipy.spatial.distance
import scipy.ndimage.filters
from PIL import Image

from robovat.perception import depth_utils
from robovat.perception.camera.camera import Camera
from robovat.utils.logging import logger


def surface_normals(depth, edge_pixels):
    """Return an array of the surface normals at the edge pixels.

    Args:
        depth: The depth image.
        edge_pixels: The edges of pixels of the image.

    Returns:
        The array of surface normals.
    """
    # Compute the gradients.
    grad = np.gradient(depth.astype(np.float32))

    # Compute surface normals.
    normals = np.zeros([edge_pixels.shape[0], 2])

    for i, pixel in enumerate(edge_pixels):
        dx = grad[1][pixel[0], pixel[1]]
        dy = grad[0][pixel[0], pixel[1]]

        normal = np.array([dy, dx])

        if np.linalg.norm(normal) == 0:
            normal = np.array([1, 0])

        normal = normal / np.linalg.norm(normal)
        normals[i, :] = normal

    return normals


def force_closure(p1, p2, n1, n2, mu):
    """Check if the point and normal pairs are in force closure.

    Args:
        p1: The first point.
        p2: The second point.
        n1: The surface normal of the first point.
        n2: The surface normal of the second point.
        mu: The friction coefficient.

    Returns:
        True if the force closure condition is satisfied, False otherwise.
    """
    # Line between the contacts.
    v = p2 - p1
    v = v / np.linalg.norm(v)

    # Compute cone membership.
    alpha = np.arctan(mu)
    in_cone_1 = (np.arccos(n1.dot(-v)) < alpha)
    in_cone_2 = (np.arccos(n2.dot(v)) < alpha)

    return (in_cone_1 and in_cone_2)


def image_dist(g1, g2, alpha=1.0):
    """Computes the distance between grasps in image space.

    Euclidean distance with alpha weighting of angles

    Args:
        g1: First grasp.
        g2: Second grasp.
        alpha: Weight of angle distance (rad to meters).

    Returns:
        Distance between grasps.
    """
    g1_center = 0.5 * (g1[:, 0:2] + g1[:, 2:4])
    g1_axis = g1[:, 2:4] - g1[:, 0:2]
    g1_axis = g1_axis / np.linalg.norm(g1_axis)

    g2_center = 0.5 * (g2[:, 0:2] + g2[:, 2:4])
    g2_axis = g2[:, 2:4] - g2[:, 0:2]
    g2_axis = g2_axis / np.linalg.norm(g2_axis)

    point_dist = np.linalg.norm(g1_center - g2_center, axis=-1)
    axis_dist = np.arccos(np.sum(g1_axis * g2_axis, axis=-1))

    return point_dist + alpha * axis_dist


class ImageGraspSampler(object):
    """Image grasp sampler.

    Wraps image to crane grasp candidate generation for easy deployment of
    GQ-CNN.
    """

    __metaclass__ = ABCMeta

    def sample(self, depth, camera, num_samples):
        """Samples a set of 2D grasps from a given RGB-D image.

        Args:
            depth: Depth image.
            camera: The camera model.
            num_samples: Number of grasps to sample.

        Returns:
            List of 2D grasp candidates
        """
        # Sample an initial set of grasps (without depth).
        logger.debug('Sampling grasp candidates...')
        grasps = self._sample(depth, camera, num_samples)
        logger.debug('Sampled %d grasp candidates from the image.'
                     % (len(grasps)))

        return grasps

    @abstractmethod
    def _sample(self, image, camera, num_samples):
        """Sample a set of 2D grasp candidates from a depth image.

        Args:
            image: Depth image.
            camera: The camera model.
            num_samples: Number of grasps to sample.

        Returns:
            List of 2D grasp candidates
        """
        pass


class AntipodalDepthImageGraspSampler(ImageGraspSampler):
    """Grasp sampler for antipodal point pairs from depth image gradients.
    """

    def __init__(self,
                 friction_coef,
                 depth_grad_thresh,
                 depth_grad_gaussian_sigma,
                 downsample_rate,
                 max_rejection_samples,
                 crop,
                 min_dist_from_boundary,
                 min_grasp_dist,
                 angle_dist_weight,
                 depth_samples_per_grasp,
                 min_depth_offset,
                 max_depth_offset,
                 depth_sample_window_height,
                 depth_sample_window_width,
                 gripper_width=0.0):
        """Initialize the sampler.

        Args:
            friction_coef: Friction coefficient for 2D force closure.
            depth_grad_thresh: Threshold for depth image gradients to determine
                edge points for sampling.
            depth_grad_gaussian_sigma: Sigma used for pre-smoothing the depth
            image for better gradients.
            downsample_rate: Factor to downsample the depth image by before
                sampling grasps.
            max_rejection_samples: Ceiling on the number of grasps to check in
                antipodal grasp rejection sampling.
            crop: The rectangular crop of the grasping region on images.
            min_dist_from_boundary: Minimum distance from the crop of the
                grasping region
            min_grasp_dist: Threshold on the grasp distance.
            angle_dist_weight: Amount to weight the angle difference in grasp
                distance computation.
            depth_samples_per_grasp: Number of depth samples to take per grasp.
            min_depth_offset: Offset from the minimum depth at the grasp center
                pixel to use in depth sampling.
            max_depth_offset: Offset from the maximum depth across all edges.
            depth_sample_window_height: Height of a window around the grasp
                center pixel used to determine min depth.
            depth_sample_window_width: Width of a window around the grasp
                center pixel used to determine min depth.
            gripper_width: Maximum width of the gripper.
        """
        # Antipodality parameters.
        self.friction_coef = friction_coef
        self.depth_grad_thresh = depth_grad_thresh
        self.depth_grad_gaussian_sigma = depth_grad_gaussian_sigma
        self.downsample_rate = downsample_rate
        self.max_rejection_samples = max_rejection_samples

        # Distance thresholds for rejection sampling.
        self.crop = crop
        self.min_dist_from_boundary = min_dist_from_boundary
        self.min_grasp_dist = min_grasp_dist
        self.angle_dist_weight = angle_dist_weight

        # Depth sampling parameters.
        self.depth_samples_per_grasp = max(depth_samples_per_grasp, 1)
        self.min_depth_offset = min_depth_offset
        self.max_depth_offset = max_depth_offset
        self.depth_sample_window_height = depth_sample_window_height
        self.depth_sample_window_width = depth_sample_window_width

        # Gripper width.
        self.gripper_width = gripper_width

    def _sample(self, image, camera, num_samples):  # NOQA
        """Sample antipodal grasps.

        Sample a set of 2D grasp candidates from a depth image by finding depth
        edges, then uniformly sampling point pairs and keeping only antipodal
        grasps with width less than the maximum allowable.

        Args:
            depth: Depth image.
            camera: The camera model.
            num_samples: Number of grasps to sample.

        Returns:
            List of 2D grasp candidates
        """
        if not isinstance(camera, Camera):
            intrinsics = camera
            camera = Camera()
            camera.set_calibration(intrinsics, np.zeros((3,)), np.zeros((3,)))

        image = np.squeeze(image, -1)

        if self.crop is None:
            crop = [0, 0, image.shape[0], image.shape[1]]
            cropped_image = image
        else:
            crop = self.crop
            cropped_image = image[crop[0]:crop[2],
                                  crop[1]:crop[3]]

        # Crope the image.
        image_filtered = scipy.ndimage.filters.gaussian_filter(
            cropped_image, sigma=self.depth_grad_gaussian_sigma)

        # Compute edge pixels.
        new_size = (int(image_filtered.shape[1] / self.downsample_rate),
                    int(image_filtered.shape[0] / self.downsample_rate))
        image_downsampled = np.array(
            Image.fromarray(image_filtered).resize(new_size, Image.BILINEAR))
        image_threshed = depth_utils.threshold_gradients(
            image_downsampled, self.depth_grad_thresh)
        image_zero = np.where(image_threshed == 0)
        image_zero = np.c_[image_zero[0], image_zero[1]]
        edge_pixels = self.downsample_rate * image_zero

        # Return if no edge pixels
        num_pixels = edge_pixels.shape[0]
        if num_pixels == 0:
            return []

        # Compute surface normals.
        edge_normals = surface_normals(image_filtered, edge_pixels)

        # Prune surface normals. Form set of valid candidate point pairs.
        if self.gripper_width > 0:
            _depth = np.max(image_filtered) + self.min_depth_offset
            p1 = np.array([0, 0, _depth])
            p2 = np.array([self.gripper_width, 0, _depth])
            u1 = camera.project_point(p1, is_world_frame=False)
            u2 = camera.project_point(p2, is_world_frame=False)
            max_grasp_width_pixel = np.linalg.norm(u1 - u2)
        else:
            max_grasp_width_pixel = np.inf

        normal_ip = edge_normals.dot(edge_normals.T)
        distances = scipy.spatial.distance.squareform(
            scipy.spatial.distance.pdist(edge_pixels))
        valid_indices = np.where(
            (normal_ip < -np.cos(np.arctan(self.friction_coef))) &
            (distances < max_grasp_width_pixel) &
            (distances > 0.0))
        valid_indices = np.c_[valid_indices[0], valid_indices[1]]

        # Return if no antipodal pairs.
        num_pairs = valid_indices.shape[0]
        if num_pairs == 0:
            return []

        sample_size = min(self.max_rejection_samples, num_pairs)
        candidate_pair_indices = np.random.choice(
            num_pairs, size=sample_size, replace=False)

        # Iteratively sample grasps.
        grasps = np.zeros([num_samples, 5], dtype=np.float32)
        num_grasps = 0

        for sample_ind in candidate_pair_indices:
            if num_grasps >= num_samples:
                break

            # Sample a random pair without replacement.
            pair_ind = valid_indices[sample_ind, :]
            p1 = edge_pixels[pair_ind[0], :]
            p2 = edge_pixels[pair_ind[1], :]
            n1 = edge_normals[pair_ind[0], :]
            n2 = edge_normals[pair_ind[1], :]

            # Check the force closure.
            if not force_closure(p1, p2, n1, n2, self.friction_coef):
                continue

            # Convert the coordinates.
            point1 = np.array([p1[1] + crop[1], p1[0] + crop[0]])
            point2 = np.array([p2[1] + crop[1], p2[0] + crop[0]])

            # Compute grasp parameters.
            grasp_center = 0.5 * (point1 + point2)
            dist_from_boundary = min(
                np.abs(crop[0] - grasp_center[1]),
                np.abs(crop[1] - grasp_center[0]),
                np.abs(grasp_center[1] - crop[2]),
                np.abs(grasp_center[0] - crop[3]))

            if dist_from_boundary < self.min_dist_from_boundary:
                continue

            # Skip if the grasp is close to any previously sampled grasp.
            if num_grasps > 0:
                grasp = np.expand_dims(np.r_[point1, point2, 0.0], 0)
                grasp_dists = image_dist(grasp, grasps[:num_grasps, :])

                if np.min(grasp_dists) <= self.min_grasp_dist:
                    continue

            # Get depth in the neighborhood of the center pixel.
            window = [
                int(grasp_center[1] - self.depth_sample_window_height),
                int(grasp_center[1] + self.depth_sample_window_height),
                int(grasp_center[0] - self.depth_sample_window_width),
                int(grasp_center[0] + self.depth_sample_window_width)]
            image_window = image[window[0]:window[1], window[2]:window[3]]
            center_depth = np.min(image_window)

            if center_depth == 0 or np.isnan(center_depth):
                continue

            min_depth = np.min(center_depth) + self.min_depth_offset
            max_depth = np.max(center_depth) + self.max_depth_offset

            # Sample depth between the min and max.
            for i in range(self.depth_samples_per_grasp):
                sample_depth = (
                    min_depth + np.random.rand() * (max_depth - min_depth))
                grasp = np.expand_dims(np.r_[point1, point2, sample_depth], 0)

                if num_grasps == 0:
                    grasps[:, :] = grasp
                else:
                    grasps[num_grasps] = grasp

                num_grasps += 1

        if num_grasps == 0:
            raise ValueError('Failed to sample any valid grasp.')

        return grasps
