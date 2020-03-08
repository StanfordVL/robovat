"""Antipodal grasp 4-DoF policy.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import numpy as np

from robovat.envs.grasp import image_grasp_sampler
from robovat.policies import policy
from robovat.utils.yaml_config import YamlConfig


class AntipodalGrasp4DofPolicy(policy.Policy):
    """Antipodal grasp 4-DoF policy."""

    def __init__(self,
                 env,
                 config=None):
        """Initialize.

        Args:
            env: Environment.
            config: Policy configuration.
        """
        super(AntipodalGrasp4DofPolicy, self).__init__(env, config)
        config = self.config
        self._sampler = image_grasp_sampler.AntipodalDepthImageGraspSampler(
            friction_coef=config.SAMPLER.FRICTION_COEF,
            depth_grad_thresh=config.SAMPLER.DEPTH_GRAD_THRESH,
            depth_grad_gaussian_sigma=config.SAMPLER.DEPTH_GRAD_GAUSSIAN_SIGMA,
            downsample_rate=config.SAMPLER.DOWNSAMPLE_RATE,
            max_rejection_samples=config.SAMPLER.MAX_REJECTION_SAMPLES,
            crop=config.SAMPLER.CROP,
            min_dist_from_boundary=config.SAMPLER.MIN_DIST_FROM_BOUNDARY,
            min_grasp_dist=config.SAMPLER.MIN_GRASP_DIST,
            angle_dist_weight=config.SAMPLER.ANGLE_DIST_WEIGHT,
            depth_samples_per_grasp=config.SAMPLER.DEPTH_SAMPLES_PER_GRASP,
            min_depth_offset=config.SAMPLER.MIN_DEPTH_OFFSET,
            max_depth_offset=config.SAMPLER.MAX_DEPTH_OFFSET,
            depth_sample_window_height=(
                    config.SAMPLER.DEPTH_SAMPLE_WINDOW_HEIGHT),
            depth_sample_window_width=(
                config.SAMPLER.DEPTH_SAMPLE_WINDOW_WIDTH),
            gripper_width=config.GRIPPER_WIDTH)

    @property
    def default_config(self):
        """Load the default configuration file."""
        config_path = os.path.join('configs', 'policies',
                                   'antipodal_grasp_4dof_policy.yaml')
        assert os.path.exists(config_path), (
            'Default configuration file %s does not exist' % (config_path)
        )
        return YamlConfig(config_path).as_easydict()

    def _action(self, observation):
        """Implementation of action.

        Args:
            observation: The observation of the current step.

        Returns:
            action: The action of the current step.
        """
        depth = observation['depth']
        intrinsics = observation['intrinsics']
        grasps = self._sampler.sample(depth, intrinsics, 1)
        return np.squeeze(grasps, axis=0)
