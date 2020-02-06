"""Reward function of the environments.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from robovat.envs.push import layouts as push_layouts
from robovat.reward_fns import reward_fn
from robovat.utils.logging import logger  # NOQA


CLEARING_GOAL_X = 0.7
CLEARING_GOAL_Y = 0.9


def process_state(state):
    if isinstance(state, dict):
        if 'position' in state:
            state = state['position'][..., :2]
        else:
            point_cloud = state['point_cloud']
            centers = np.mean(point_cloud, axis=-2)
            state = centers[..., :2]

        if len(state.shape) == 2:
            state = np.expand_dims(state, axis=0)

    assert state.shape[-1] == 2
    return state


def dummy_reward_fn(state, next_state):
    state = process_state(state)
    next_state = process_state(next_state)

    if len(state.shape) == 2:
        shape = ()
    elif len(state.shape) == 3:
        shape = (state.shape[0],)
    else:
        raise ValueError

    reward = np.ones(shape, dtype=np.float32)
    termination = np.zeros_like(reward, dtype=np.bool)
    return reward, termination


def get_tiles(centers, size, offset):
    tiles = [np.array(offset) + np.array(center) * size
             for center in centers]
    tiles = np.stack(tiles, axis=0)
    return tiles


def check_on_tiles(position, centers, size, offset, max_dist=None):
    if max_dist is None:
        max_dist = size

    tiles = get_tiles(centers, size, offset)
    assert len(position.shape) == 2
    assert len(tiles.shape) == 2
    dists = np.expand_dims(position, axis=1) - np.expand_dims(tiles, axis=0)
    is_on_tiles = np.logical_and(np.abs(dists[:, :, 0]) <= 0.5 * max_dist,
                                 np.abs(dists[:, :, 1]) <= 0.5 * max_dist)
    return np.any(is_on_tiles, axis=1)


def get_tile_dists(position, centers, size, offset):
    tiles = get_tiles(centers, size, offset)
    dists = np.expand_dims(position, axis=1) - np.expand_dims(tiles, axis=0)
    dists = np.linalg.norm(dists, axis=-1)
    dists = dists.min(axis=1)
    return dists


def clearing_termination(state, next_state, layout, is_planning):
    raise NotImplementedError


def clearing_goal(state, layout):
    num_bodies = state.shape[1]
    for i in range(num_bodies):
        is_clear_i = np.logical_not(
            check_on_tiles(state[:, i, :],
                           layout.region,
                           layout.size * 1.25,
                           layout.offset)
        )
        if i == 0:
            goal_reached = is_clear_i
        else:
            goal_reached = np.logical_and(
                goal_reached,
                is_clear_i)

    return goal_reached


def clearing_score(state, layout):
    dists1 = np.mean(np.abs(state[:, :, 0] - CLEARING_GOAL_X), axis=1)
    dists2 = np.mean(np.abs(state[:, :, 1] - CLEARING_GOAL_Y), axis=1)
    dists3 = np.mean(np.abs(state[:, :, 1] + CLEARING_GOAL_Y), axis=1)
    dists = np.minimum(dists1, dists2)
    dists = np.minimum(dists1, dists3)
    return -dists


def insertion_termination(state, next_state, layout, is_planning):
    num_bodies = next_state.shape[1]

    middle_state1 = state + (1. / 3.) * (next_state - state)
    middle_state2 = state + (2. / 3.) * (next_state - state)

    for i in range(num_bodies):
        if is_planning:
            termination_i_0 = check_on_tiles(
                next_state[:, i, :],
                layout.region,
                layout.size * 1.25,
                layout.offset)
            termination_i_1 = check_on_tiles(
                middle_state1[:, i, :],
                layout.region,
                layout.size * 1.25,
                layout.offset)
            termination_i_2 = check_on_tiles(
                middle_state2[:, i, :],
                layout.region,
                layout.size * 1.25,
                layout.offset)
            termination_i = np.logical_or(termination_i_1, termination_i_2)
            termination_i = np.logical_or(termination_i_0, termination_i)
        else:
            termination_i = check_on_tiles(
                next_state[:, i, :],
                layout.region,
                layout.size * 1.25,
                layout.offset)

        if i == 0:
            termination = termination_i
        else:
            termination = np.logical_or(termination, termination_i)

    # TODO(Debug)
    if is_planning:
        return termination
    else:
        return np.zeros_like(termination)


def insertion_goal(state, layout):
    return check_on_tiles(state[:, 0, :],
                          layout.goal,
                          layout.size,
                          layout.offset)


def insertion_score(state, layout):
    return -get_tile_dists(state[:, 0, :],
                           layout.goal,
                           layout.size,
                           layout.offset)


def crossing_termination(state, next_state, layout, is_planning):
    middle_state1 = state + (1. / 3.) * (next_state - state)
    middle_state2 = state + (2. / 3.) * (next_state - state)

    if is_planning:
        is_on_brige0 = check_on_tiles(
            next_state[:, 0, :],
            layout.region,
            layout.size,
            layout.offset,
            max_dist=layout.size)
        is_on_brige1 = check_on_tiles(
            middle_state1[:, 0, :],
            layout.region,
            layout.size,
            layout.offset,
            max_dist=layout.size)
        is_on_brige2 = check_on_tiles(
            middle_state2[:, 0, :],
            layout.region,
            layout.size,
            layout.offset,
            max_dist=layout.size)
        is_on_brige = np.logical_and(is_on_brige1, is_on_brige2)
        is_on_brige = np.logical_and(is_on_brige0, is_on_brige)
    else:
        is_on_brige = check_on_tiles(
            next_state[:, 0, :],
            layout.region,
            layout.size,
            layout.offset,
            max_dist=layout.size * 1.5)
    return np.logical_not(is_on_brige)


def crossing_goal(state, layout):
    return check_on_tiles(state[:, 0, :],
                          layout.goal,
                          layout.size,
                          layout.offset)


def crossing_score(state, layout):
    return -get_tile_dists(state[:, 0, :],
                           layout.goal,
                           layout.size,
                           layout.offset)


def check_collision(state,
                    layout,
                    is_planning,
                    min_dist=0.15,
                    weight=10.0):
    target = state[:, 0, :]
    dists = np.expand_dims(target, axis=1) - state[:, 1:, :]
    dists = np.linalg.norm(dists, axis=-1)
    collision = np.less(dists, min_dist)
    collision = np.any(collision, axis=1)
    return collision


def check_stride(state,
                 next_state,
                 min_stride,
                 max_stride):
    strides = np.linalg.norm(next_state - state, axis=-1)

    return np.logical_or(
        np.all(np.less(strides, min_stride), axis=1),
        np.any(np.greater(strides, max_stride), axis=1))


def check_border(state,
                 next_state,
                 x_range=[0.22, 0.98],
                 # x_range=[0.3, 0.8],
                 y_range=[-0.56, 0.66],
                 tolerance=0.02):
    violate_border_x = np.logical_or(
        np.less(next_state[..., 0], x_range[0] - tolerance),
        np.greater(next_state[..., 0], x_range[1] + tolerance))
    violate_border_y = np.logical_or(
        np.less(next_state[..., 1], y_range[0] - tolerance),
        np.greater(next_state[..., 1], y_range[1] + tolerance))
    violate_border = np.logical_or(violate_border_x, violate_border_y)
    return np.any(violate_border, axis=1)


def check_target_border(state,
                        next_state,
                        x_range=[0.3, 0.8],
                        y_range=[-0.56, 0.66],
                        tolerance=0.02):
    violate_border_x = np.logical_or(
        np.less(next_state[:, 0, 0], x_range[0] - tolerance),
        np.greater(next_state[:, 0, 0], x_range[1] + tolerance))
    violate_border_y = np.logical_or(
        np.less(next_state[:, 0, 1], y_range[0] - tolerance),
        np.greater(next_state[:, 0, 1], y_range[1] + tolerance))
    violate_border = np.logical_or(violate_border_x, violate_border_y)
    return violate_border


def get_reward_fn(task_name,  # NOQA
                  layout_id,
                  goal_reward=100.0,
                  termination_reward=-100.0,
                  dense_reward=1.0,
                  time_reward=-1.0,
                  use_dense_reward=True,
                  use_time_penalty=True,
                  is_planning=False,
                  is_high_level=False):
    if task_name is None or task_name == 'data_collection':
        return dummy_reward_fn

    if task_name == 'clearing':
        termination_fns = []
        goal_fns = [clearing_goal]
        score_fns = [clearing_score]
    elif task_name == 'insertion':
        termination_fns = [insertion_termination]
        goal_fns = [insertion_goal]
        score_fns = [insertion_score]
    elif task_name == 'crossing':
        termination_fns = [crossing_termination]
        goal_fns = [crossing_goal]
        score_fns = [crossing_score]
    else:
        raise ValueError('Unrecognized manipulation task: %r' % task_name)

    layouts = push_layouts.TASK_NAME_TO_LAYOUTS[task_name]

    def reward_fn(state, next_state):
        layout = layouts[layout_id]

        state = process_state(state)
        next_state = process_state(next_state)

        batch_size = state.shape[0]
        reward = np.zeros([batch_size], dtype=np.float32)
        termination = np.zeros([batch_size], dtype=np.bool)

        if is_planning:
            if is_high_level:
                termination_i = check_stride(state,
                                             next_state,
                                             min_stride=0.1,
                                             max_stride=0.3)
            else:
                termination_i = check_stride(state,
                                             next_state,
                                             min_stride=0.01,
                                             max_stride=0.15)
            termination = np.logical_or(termination, termination_i)

        if is_planning:
            termination_i = check_border(state, next_state)
            termination = np.logical_or(termination, termination_i)

            if task_name == 'insertion':
                termination_i = check_target_border(state, next_state)
                termination = np.logical_or(termination, termination_i)

        for termination_fn in termination_fns:
            termination_i = termination_fn(
                state, next_state, layout, is_planning)
            termination = np.logical_or(termination, termination_i)

        penalty_ternmination = termination

        # Sparse reward.
        for goal_fn in goal_fns:
            goal_reached = goal_fn(next_state, layout)
            goal_reached = np.logical_and(
                goal_reached, np.logical_not(termination))
            reward_i = goal_reward * np.array(goal_reached, dtype=np.float32)
            termination_i = np.array(goal_reached, dtype=np.bool)
            reward += reward_i
            termination = np.logical_or(termination, termination_i)

        # Termination reward.
        if not is_planning:
            penalty_ternmination = np.logical_and(
                penalty_ternmination,
                np.logical_not(goal_reached))

        reward += termination_reward * np.array(
            penalty_ternmination, dtype=np.float32)

        # Dense reward.
        if use_dense_reward:
            for score_fn in score_fns:
                score = score_fn(state, layout)
                next_score = score_fn(next_state, layout)

                reward_i = np.abs(next_score - score) * dense_reward
                reward += reward_i

        # Time.
        if use_time_penalty:
            reward += time_reward

        return reward, termination

    return reward_fn


class PushReward(reward_fn.RewardFn):
    """Reward function of the pushing tasks."""

    def __init__(self,
                 name,
                 task_name,
                 layout_id,
                 is_planning=False):
        """Initialize."""
        self.name = name

        self.env = None
        self.reward_fn = get_reward_fn(
            task_name=task_name,
            layout_id=layout_id,
            is_planning=is_planning)

        self.history = []

    def get_reward(self):
        """Returns the reward value of the current step."""
        assert self.env.prev_obs_data is not None
        assert self.env.obs_data is not None
        reward, termination = self.reward_fn(self.env.prev_obs_data,
                                             self.env.obs_data)
        reward = float(reward)
        termination = bool(termination)

        return reward, termination
