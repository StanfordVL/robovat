"""Base class for policies.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import os.path

from robovat.utils.string_utils import camelcase_to_snakecase
from robovat.utils.yaml_config import YamlConfig


class Policy(object):
    """Abstract class for policies.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, env, config=None):
        """Initialize.

        Args:
            env: The environment to run the policy, as an instance of Env.
            config: The configuration.
        """
        self.env = env
        self.config = config or self.default_config

    @property
    def default_config(self):
        """Load the default configuration file."""
        policy_name = camelcase_to_snakecase(type(self).__name__)
        config_path = os.path.join('configs', 'policies',
                                   '%s.yaml' % (policy_name))
        assert os.path.exists(config_path), (
            'Default configuration file %s does not exist' % (config_path))
        return YamlConfig(config_path).as_easydict()

    def action(self, observation):
        """Returns an action for a given state.

        Args:
            observation: The observation of the current step.

        Returns:
            action: The action of the current step.
        """
        return self._action(observation)

    @abc.abstractmethod
    def _action(self, observation):
        """Implementation of action.

        Args:
            observation: The observation of the current step.

        Returns:
            action: The action of the current step.
        """
        raise NotImplementedError
