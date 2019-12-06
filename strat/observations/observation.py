"""Observation of the gym environments.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc


class Observation(object):
    """Observation of the gym environments."""
    
    __metaclass__ = abc.ABCMeta

    def initialize(self, env):
        """Initialize the observation.

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
    def get_gym_space(self):
        """Returns gym space of this observation."""
        pass

    @abc.abstractmethod
    def get_observation(self):
        """Returns the observation data of the current step."""
        pass
