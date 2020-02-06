"""Heuristic push policy.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from robovat.envs.push import heuristic_push_sampler
from robovat.policies import policy


class HeuristicPushPolicy(policy.Policy):
    """Heuristic push policy."""

    def __init__(self,
                 env,
                 config=None):
        """Initialize.

        Args:
            env: Environment.
            config: Policy configuration.
        """
        super(HeuristicPushPolicy, self).__init__(env, config)
        config = self.config
        self._sampler = heuristic_push_sampler.HeuristicPushSampler(
            cspace_low=config.ACTION.CSPACE.LOW,
            cspace_high=config.ACTION.CSPACE.HIGH,
            translation_x=config.ACTION.MOTION.TRANSLATION_X,
            translation_y=config.ACTION.MOTION.TRANSLATION_Y,
            max_attemps=config.HEURISTICS.MAX_ATTEMPS)

    def _action(self, observation):
        """Implementation of action.

        Args:
            observation: The observation of the current step.

        Returns:
            action: The action of the current step.
        """
        position = observation['position']
        body_mask = observation['body_mask']
        num_episodes = observation['num_episodes']
        num_steps = observation['num_steps']
        action = self._sampler.sample(
            position,
            body_mask,
            num_episodes,
            num_steps,
            num_samples=1)
        return action
