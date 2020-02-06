"""Observations about the action.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import gym

from robovat.observations import observation


INF = 2**32 - 1


class FlagObs(observation.Observation):
    """Observation which indicates if the action is valid."""

    def __init__(self, key, name=None):
        """Initialize."""
        self.key = key
        self.name = name or 'attribute_obs'
        self.env = None

    def initialize(self, env):
        self.env = env

    def get_gym_space(self):
        """Returns gym space of this observation."""
        return gym.spaces.Discrete(1)

    def get_observation(self):
        """Returns the observation data of the current step."""
        attrib = self.env.attributes[self.key]

        if attrib is True:
            value = 1
        elif attrib is False:
            value = 0
        else:
            raise ValueError('Unrecognized flag value: %r' % (attrib))

        return np.array(value, dtype=np.int64)


class IntegerAttributeObs(observation.Observation):
    """Observation of an integer attribute."""

    def __init__(self, key, max_value, name=None):
        """Initialize."""
        self.key = key
        self.max_value = max_value
        self.name = name or 'attribute_obs'
        self.env = None

    def initialize(self, env):
        self.env = env

    def get_gym_space(self):
        """Returns gym space of this observation."""
        return gym.spaces.Discrete(self.max_value)

    def get_observation(self):
        """Returns the observation data of the current step."""
        value = self.env.attributes[self.key]
        return np.array(value, dtype=np.int64)


class RandomIndexObs(observation.Observation):
    """Observation of a random index."""

    def __init__(self, name=None):
        """Initialize."""
        self.name = name or 'random_index_obs'
        self.env = None
        self.random_id = None

    def initialize(self, env):
        self.env = env

    def on_episode_start(self):
        """Called at the start of each episode."""
        self.random_id = np.random.randint(2**32 - 1)

    def get_gym_space(self):
        """Returns gym space of this observation."""
        return gym.spaces.Box(-INF, INF, (), dtype=np.int64)

    def get_observation(self):
        """Returns the observation data of the current step."""
        return np.array(self.random_id, dtype=np.int64)


class ArrayAttributeObs(observation.Observation):
    """Observation of an integer attribute."""

    def __init__(self, key, shape, dtype=np.float32, name=None):
        """Initialize."""
        self.key = key
        self.shape = shape
        self.dtype = dtype
        self.name = name or 'attribute_obs'
        self.env = None

    def initialize(self, env):
        self.env = env

    def get_gym_space(self):
        """Returns gym space of this observation."""
        return gym.spaces.Box(-INF, INF, self.shape, dtype=self.dtype)

    def get_observation(self):
        """Returns the observation data of the current step."""
        value = self.env.attributes[self.key]
        return np.array(value, dtype=self.dtype)
