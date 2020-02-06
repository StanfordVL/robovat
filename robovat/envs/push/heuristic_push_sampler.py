"""Heuristics push sampler.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from robovat.utils.logging import logger


SEED = 42


class HeuristicPushSampler(object):
    """Heuristics push sampler."""

    def __init__(self,
                 cspace_low,
                 cspace_high,
                 translation_x,
                 translation_y,
                 start_margin=0.05,
                 motion_margin=0.01,
                 max_attemps=20000):
        """Initialize.

        Args:
            cspace_low:
            cspace_high:
            translation_x:
            translation_y:
            start_margin:
            motion_margin:
            max_attemps:
        """
        self.cspace_low = np.array(cspace_low)
        self.cspace_high = np.array(cspace_high)
        self.cspace_offset = 0.5 * (self.cspace_high + self.cspace_low)
        self.cspace_range = 0.5 * (self.cspace_high - self.cspace_low)

        self.translation_x = translation_x
        self.translation_y = translation_y

        self.start_margin = start_margin
        self.motion_margin = motion_margin
        self.max_attemps = max_attemps

        self.last_end = None

    def sample(self,
               position,
               body_mask,
               num_episodes,
               num_steps,
               num_samples=1):
        list_action = []

        for i in range(num_samples):
            action = self._sample(position, body_mask, num_episodes, num_steps)
            list_action.append(action)

        return np.stack(list_action, axis=0)

    def _sample(self, position, body_mask, num_episodes, num_steps):
        num_bodies = int(np.sum(body_mask))
        body_id = int(num_episodes) % int(num_bodies)

        position = position[:num_bodies]
        target_position = position[body_id:body_id+1]

        base_angle = (num_episodes * SEED) % (2 * np.pi)

        if num_steps == 0:
            self.last_end = None

        for i in range(self.max_attemps):

            start = np.random.uniform(-1., 1., [2])

            delta_angle = np.random.uniform(-0.25 * np.pi, 0.25 * np.pi)
            angle = base_angle + delta_angle
            # angle = base_angle

            motion = [1.0 * np.cos(angle), 1.0 * np.sin(angle)]
            motion = np.array(motion, dtype=np.float32)
            motion += np.random.uniform(-0.3, 0.3, [2])
            motion = np.clip(motion, -1.0, 1.0)

            waypoints = self.get_waypoints(start, motion)

            # Check if is_safe.
            is_safe = self.is_waypoint_clear(
                waypoints[0], None, position, self.start_margin)
            if not is_safe:
                continue

            # Check if is_effective.
            is_effective = not self.is_waypoint_clear(
                waypoints[0],
                waypoints[1],
                target_position,
                self.motion_margin)

            if not is_effective:
                continue

            if is_effective and is_safe:
                x = ((waypoints[1][0] - self.cspace_offset[0]) /
                     (self.cspace_range[0]))
                y = ((waypoints[1][1] - self.cspace_offset[1]) /
                     (self.cspace_range[1]))
                self.last_end = np.asarray([x, y])
                break

        if i == self.max_attemps - 1:
            logger.info('HeuristicSampler did not find a good sample.')

        start = np.array(start, dtype=np.float32)
        motion = np.array(motion, dtype=np.float32)
        action = np.concatenate([start, motion], axis=-1)
        return action

    def get_waypoints(self, start, motion):
        motion = np.reshape(motion, [-1, 2])

        x = start[0] * self.cspace_range[0] + self.cspace_offset[0]
        y = start[1] * self.cspace_range[1] + self.cspace_offset[1]
        start = [x, y]
        waypoints = [start]

        for i in range(motion.shape[0]):
            delta_x = motion[i, 0] * self.translation_x
            delta_y = motion[i, 1] * self.translation_y

            x = x + delta_x
            y = y + delta_y

            x = np.clip(x, self.cspace_low[0], self.cspace_high[0])
            y = np.clip(y, self.cspace_low[1], self.cspace_high[1])

            waypoint = [x, y]
            waypoints.append(waypoint)

        return waypoints

    def is_waypoint_clear(self, waypoint1, waypoint2, position, margin):
        if waypoint2 is None:
            delta_x = position[..., 0] - waypoint1[0]
            delta_y = position[..., 1] - waypoint1[1]
            dist = np.sqrt(delta_x * delta_x + delta_y * delta_y)
            is_clear = dist > margin
        else:
            delta_x = position[..., 0] - waypoint1[0]
            delta_y = position[..., 1] - waypoint1[1]
            dist1 = np.sqrt(delta_x * delta_x + delta_y * delta_y)

            delta_x = position[..., 0] - waypoint2[0]
            delta_y = position[..., 1] - waypoint2[1]
            dist2 = np.sqrt(delta_x * delta_x + delta_y * delta_y)

            is_clear = np.logical_and(
                dist1 >= margin,
                dist2 >= margin)

        is_all_clear = np.all(is_clear)
        return is_all_clear
