"""Base class for policies.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc


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
        self.config = config

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
