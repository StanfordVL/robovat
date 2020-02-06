"""Base class for policies.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from robovat.policies import policy


class RandomPolicy(policy.Policy):

    def _action(self, observation):
        """Implementation of action.

        Args:
            observation: The observation of the current step.

        Returns:
            action: The action of the current step.
        """
        return self.env.action_space.sample()
