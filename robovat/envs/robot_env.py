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

from robovat.simulation.simulator import Simulator
from robovat.utils.string_utils import camelcase_to_snakecase
from robovat.utils.yaml_config import YamlConfig
from robovat.utils.logging import logger


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
                 simulator=True,
                 config=None,
                 debug=False):
        """Initialize.

        Args:
            simulator: Instance of the simulator.
            config: Environment configuration.
            debug: True if it is debugging mode, False otherwise.
        """
        if simulator is True:
            worker_id = random.randint(0, 2**32-1)
            simulator = Simulator(worker_id=worker_id)

        self._simulator = simulator
        self._config = config or self.default_config
        self._debug = debug

        self._num_episodes = 0
        self._num_steps = 0
        self._episode_reward = 0.0
        self._total_reward = 0.0
        self._done = True

        self._observations = self._create_observations()
        for obs in self.observations:
            obs.initialize(self)

        self._reward_fns = self._create_reward_fns()
        for reward_fn in self.reward_fns:
            reward_fn.initialize(self)

        self._action_space = self._create_action_space()

        self._observation_space = gym.spaces.Dict([
            (obs.name, obs.get_gym_space()) for obs in self.observations
        ])

        self._obs_data = None
        self._prev_obs_data = None
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
    def is_simulation(self):
        """A flag indicating if it is in the simulation."""
        return (self._simulator is not None)

    @property
    def simulator(self):
        """The physics engine."""
        return self._simulator

    @property
    def config(self):
        """Configuration data."""
        return self._config

    @property
    def debug(self):
        """A flag of the debugging mode."""
        return self._debug

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
    def done(self):
        """If the episode is done."""
        return self._done

    @property
    def observations(self):
        """The observations."""
        return self._observations

    @property
    def reward_fns(self):
        """The reward functions."""
        return self._reward_fns

    @property
    def observation_space(self):
        """The observation space."""
        return self._observation_space

    @property
    def action_space(self):
        """The action space."""
        return self._action_space

    @property
    def obs_data(self):
        """The observation data of the current step."""
        return self._obs_data

    @property
    def prev_obs_data(self):
        """The observation data of the previous step."""
        return self._prev_obs_data

    def _create_observations(self):
        """Create observations.

        Returns:
            List of observations.
        """
        raise NotImplementedError

    def _create_reward_fns(self):
        """Initialize reward functions.

        Returns:
            List of reward functions.
        """
        raise NotImplementedError

    def _create_action_space(self):
        """Create the action space.

        Returns:
            The action space.
        """
        raise NotImplementedError

    def reset(self):
        """Reset."""
        if not self._done and self._num_steps > 0:
            for obs in self.observations:
                obs.on_episode_end()
            for reward_fn in self.reward_fns:
                reward_fn.on_episode_end()

        self._num_steps = 0
        self._episode_reward = 0.0
        self._done = False

        if self.config.MAX_STEPS is not None:
            if self.config.MAX_STEPS == 0:
                self._done = True

        if self.is_simulation:
            self.simulator.reset()
            self.simulator.start()

        self._reset()

        for obs in self.observations:
            obs.on_episode_start()
        for reward_fn in self.reward_fns:
            reward_fn.on_episode_start()

        logger.info('episode: %d', self.num_episodes)

        self._obs_data = None
        self._prev_obs_data = None
        self._obs_step = None

        return self.get_observation()

    def step(self, action):
        """Take a step.

        See parent class.
        """
        if self._done:
            raise ValueError('The environment is done. Forget to reset?')

        self._execute_action(action)
        self._num_steps += 1

        observation = self.get_observation()

        reward, termination = self.get_reward()

        self._episode_reward += reward
        self._done = (self._done or termination)

        if self.config.MAX_STEPS is not None:
            if self.num_steps >= self.config.MAX_STEPS:
                self._done = True

        logger.info('step: %d, reward: %.3f', self.num_steps, reward)

        if self._done:
            self._num_episodes += 1
            self._total_reward += self.episode_reward
            logger.info(
                'episode_reward: %.3f, avg_episode_reward: %.3f',
                self.episode_reward,
                float(self.total_reward) / (self._num_episodes + 1e-14),
            )

        if self.debug:
            self.render()

        return observation, reward, self._done, None

    def get_observation(self):
        """Return the observation.

        Args:
            The observation data.
        """
        if self._obs_step != self._num_steps:
            self._obs_step = self._num_steps
            self._prev_obs_data = self._obs_data

            self._obs_data = collections.OrderedDict()
            for obs in self.observations:
                self._obs_data[obs.name] = obs.get_observation()

        return self._obs_data

    def get_reward(self):
        """Return the reward.

        Args:
            reward: The sum of rewards as a float number.
            termination: The termination flag.
        """
        reward = 0.0
        termination = False

        for reward_fn in self.reward_fns:
            reward_value, termination_value = reward_fn.get_reward()
            reward += reward_value

            if termination_value:
                termination = True

        reward = float(reward)

        return reward, termination

    def render(self):
        """Render the current step of the environment."""
        pass

    @abc.abstractmethod
    def _reset(self):
        """Reset the environment in simulation or the real world."""
        pass

    @abc.abstractmethod
    def _execute_action(self, action):
        """Execute the robot action."""
        pass
