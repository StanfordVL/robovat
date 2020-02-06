"""Observations about the action.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import gym

from robovat.observations import observation


class LastActionValidObs(observation.Observation):
    """Observation which indicates if the action is valid."""

    def __init__(self, name=None):
        """Initialize."""
        self.name = name or 'last_action_valid'
        self.env = None

    def initialize(self, env):
        self.env = env

    def get_gym_space(self):
        """Returns gym space of this observation."""
        return gym.spaces.Discrete(1)

    def get_observation(self):
        """Returns the observation data of the current step."""
        return np.array(self.env.last_action_is_valid, dtype=np.int32)


class ModeObs(observation.Observation):
    """Observation of the action mode."""

    def __init__(self, name=None):
        """Initialize."""
        self.name = name or 'mode'
        self.env = None

    def initialize(self, env):
        self.env = env

    def get_gym_space(self):
        """Returns gym space of this observation."""
        return gym.spaces.Discrete(len(self.env.ACTION_MODES))

    def get_observation(self):
        """Returns the observation data of the current step."""
        return np.array(self.env.mode, dtype=np.int32)
