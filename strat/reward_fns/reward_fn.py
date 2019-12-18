"""Reward function of the environments.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc


class RewardFn(object):
    """Reward function of the environments."""

    __metaclass__ = abc.ABCMeta

    def initialize(self, env):
        """Initialize the reward function.

        Args:
            env: Environment.
        """
        self.env = env

    def on_episode_start(self):
        """Called at the start of each episode."""
        pass

    def on_episode_end(self):
        """Called at the end of each episode."""
        pass

    @abc.abstractmethod
    def get_reward(self):
        """Returns the reward value of the current step."""
        pass
