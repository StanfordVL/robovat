"""The parent class for robot environments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import random
import os.path

import gym
import gym.spaces
import numpy as np

from strat.simulation.simulator import Simulator
from strat.utils.string_utils import camelcase_to_snakecase
from strat.utils.yaml_config import YamlConfig
from strat.utils.logging import logger


def get_config_value(config):
    """Get the value of an configuration item.

    If the config is None, return None. If the config is a value, return the
    value. Otherwise, the config must by a tuple or list represents low and
    high values to sample the property from.
    """
    if config is None:
        return None
    elif isinstance(config, (int, float)):
        return config
    elif isinstance(config, (list, tuple)):
        return np.random.uniform(low=config[0], high=config[1])
    else:
        raise ValueError('config %r of type %r is not supported.'
                         % (config, type(config)))


class RobotEnv(gym.Env):
    """The parent class for robot environments."""

    __metaclass__ = abc.ABCMeta

    def __init__(self,
                 observations,
                 reward_fns,
                 simulator=True,
                 config=None,
                 debug=False):
        """Initialize.

        Args:
            observations: List of observations.
            reward_fns: List of reward functions.
            simulator: Instance of the simulator.
            config: Environment configuration.
            debug: True if it is debugging mode, False otherwise.
        """
        if simulator is True:
            worker_id = random.randint(0, 2**32-1)
            simulator = Simulator(worker_id=worker_id)

        self.simulator = simulator

        self.config = config or self.default_config
        self.debug = debug

        self._num_episodes = 0
        self._num_steps = 0
        self._episode_reward = 0.0
        self._total_reward = 0.0
        self._is_done = True

        self.observations = observations
        self.reward_fns = reward_fns

        for obs in self.observations:
            obs.initialize(self)
        for reward_fn in self.reward_fns:
            reward_fn.initialize(self)

        self.observation_space = gym.spaces.Dict([
            (obs.name, obs.get_gym_space()) for obs in self.observations
        ])

        self.obs_data = None
        self.prev_obs_data = None
        self._obs_step = None

    @property
    def default_config(self):
        """Load the default configuration file."""
        env_name = camelcase_to_snakecase(type(self).__name__)
        config_path = os.path.join('configs', 'envs', '%s.yaml' % (env_name))
        assert os.path.exists(config_path), (
            'Default configuration file %s does not exist' % (config_path)
        )
        return YamlConfig(config_path).as_easydict()

    @property
    def info(self):
        """Information of the environment."""
        return {'name': type(self).__name__}

    @property
    def num_episodes(self):
        """Number of episodes."""
        return self._num_episodes

    @property
    def num_steps(self):
        """Number of steps of the current episode."""
        return self._num_steps

    @property
    def episode_reward(self):
        """Received reward of the current episode."""
        return self._episode_reward

    @property
    def total_reward(self):
        """Total reward received so far."""
        return self._total_reward

    @property
    def is_done(self):
        """If the episode is done."""
        return self._is_done

    @property
    def action_space(self):
        """The action space."""
        raise NotImplementedError

    def reset(self):
        """Reset."""
        if not self._is_done and self._num_steps > 0:
            for obs in self.observations:
                obs.on_episode_end()
            for reward_fn in self.reward_fns:
                reward_fn.on_episode_end()

        self._num_episodes += 1
        self._num_steps = 0
        self._episode_reward = 0.0
        self._is_done = False

        if self.config.MAX_STEPS is not None:
            if self.config.MAX_STEPS == 0:
                self._is_done = True

        if self.simulator:
            self.simulator.reset()
            self.simulator.start()

        self.reset_scene()
        self.reset_robot()

        for obs in self.observations:
            obs.on_episode_start()
        for reward_fn in self.reward_fns:
            reward_fn.on_episode_start()

        logger.info('episode: %d', self.num_episodes)

        self.obs_data = None
        self.prev_obs_data = None
        self._obs_step = None
        return self.get_observation()

    def step(self, action):
        """Take a step.

        See parent class.
        """
        if self._is_done:
            raise ValueError('The environment is done. Forget to reset?')

        self.execute_action(action)
        self._num_steps += 1

        observation = self.get_observation()

        reward = 0.0

        for reward_fn in self.reward_fns:
            reward_value, termination = reward_fn.get_reward()
            reward += reward_value

            if termination:
                self._is_done = True

        reward = float(reward)
        self._episode_reward += reward

        if self.config.MAX_STEPS is not None:
            if self.num_steps >= self.config.MAX_STEPS:
                self._is_done = True

        logger.info('step: %d, reward: %.3f', self.num_steps, reward)

        if self._is_done:
            self._total_reward += self.episode_reward
            logger.info(
                'episode_reward: %.3f, avg_episode_reward: %.3f',
                self.episode_reward,
                float(self.total_reward) / (self._num_episodes + 1e-14),
            )

        return observation, reward, self._is_done, None

    def get_observation(self):
        """Return the observation."""
        if self._obs_step != self._num_steps:
            self._obs_step = self._num_steps
            self.prev_obs_data = self.obs_data

            self.obs_data = collections.OrderedDict()
            for obs in self.observations:
                self.obs_data[obs.name] = obs.get_observation()

        return self.obs_data

    @abc.abstractmethod
    def reset_scene(self):
        """Reset the scene in simulation or the real world.
        """
        pass

    @abc.abstractmethod
    def reset_robot(self):
        """Reset the robot in simulation or the real world.
        """
        pass

    @abc.abstractmethod
    def execute_action(self, action):
        """Execute the robot action.
        """
        pass
