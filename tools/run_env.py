#!/usr/bin/env python

"""
Run an environment with the chosen policy.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import ast
import os
import random
import socket
import uuid
from builtins import input

import numpy as np
import h5py

import _init_paths  # NOQA
from robovat import envs
from robovat import policies
from robovat.io import hdf5_utils
from robovat.io.episode_generation import generate_episodes
from robovat.simulation.simulator import Simulator
from robovat.utils import time_utils
from robovat.utils.logging import logger
from robovat.utils.yaml_config import YamlConfig


def parse_args():
    """Parse arguments.

    Returns:
        args: The parsed arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--env',
        dest='env',
        type=str,
        help='The environment.',
        required=True)

    parser.add_argument(
        '--policy',
        dest='policy',
        type=str,
        help='The policy.',
        default=None)

    parser.add_argument(
        '--env_config',
        dest='env_config',
        type=str,
        help='The configuration file for the environment.',
        default=None)

    parser.add_argument(
        '--policy_config',
        dest='policy_config',
        type=str,
        help='The configuration file for the policy.',
        default=None)

    parser.add_argument(
        '--config_bindings',
        dest='config_bindings',
        type=str,
        help='The configuration bindings.',
        default=None)

    parser.add_argument(
        '--use_simulator',
        dest='use_simulator',
        type=int,
        help='Run experiments in the simulation is it is True.',
        default=1)

    parser.add_argument(
        '--assets',
        dest='assets_dir',
        type=str,
        help='The assets directory.',
        default='./assets')

    parser.add_argument(
        '--output',
        dest='output_dir',
        type=str,
        help='The output directory to save the episode history.',
        default=None)

    parser.add_argument(
        '--num_steps',
        dest='num_steps',
        type=int,
        help='Maximum number of time steps for each episode.',
        default=None)

    parser.add_argument(
        '--num_episodes',
        dest='num_episodes',
        type=int,
        help='Maximum number of episodes.',
        default=None)

    parser.add_argument(
        '--num_episodes_per_file',
        dest='num_episodes_per_file',
        type=int,
        help='The maximum number of episodes saved in each file.',
        default=1000)

    parser.add_argument(
        '--debug',
        dest='debug',
        type=int,
        help='True for debugging, False otherwise.',
        default=0)

    parser.add_argument(
        '--worker_id',
        dest='worker_id',
        type=int,
        help='The worker ID for running multiple simulations in parallel.',
        default=0)

    parser.add_argument(
        '--seed',
        dest='seed',
        type=int,
        help='None for random; any fixed integers for deterministic.',
        default=None)

    parser.add_argument(
        '--pause',
        dest='pause',
        type=bool,
        help='Whether to pause between episodes.',
        default=False)

    parser.add_argument(
        '--timeout',
        dest='timeout',
        type=float,
        help='Seconds of timeout for an episode.',
        default=120)

    args = parser.parse_args()

    return args


def parse_config_files_and_bindings(args):
    if args.env_config is None:
        env_config = None
    else:
        env_config = YamlConfig(args.env_config).as_easydict()

    if args.policy_config is None:
        policy_config = None
    else:
        policy_config = YamlConfig(args.policy_config).as_easydict()

    if args.config_bindings is not None:
        parsed_bindings = ast.literal_eval(args.config_bindings)
        logger.info('Config Bindings: %r', parsed_bindings)
        env_config.update(parsed_bindings)
        policy_config.update(parsed_bindings)

    return env_config, policy_config


def main():
    args = parse_args()

    # Configuration.
    env_config, policy_config = parse_config_files_and_bindings(args)

    # Set the random seed.
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    # Simulator.
    if args.use_simulator:
        simulator = Simulator(worker_id=args.worker_id,
                              use_visualizer=bool(args.debug),
                              assets_dir=args.assets_dir)
    else:
        simulator = None

    # Environment.
    env_class = getattr(envs, args.env)
    env = env_class(simulator=simulator,
                    config=env_config,
                    debug=args.debug)

    # Policy.
    policy_class = getattr(policies, args.policy)
    policy = policy_class(env=env, config=policy_config)

    # Output directory.
    if args.output_dir is not None:
        hostname = socket.gethostname()
        hostname = hostname.split('.')[0]
        output_dir = os.path.abspath(args.output_dir)
        output_dir = os.path.join(output_dir, hostname, '%02d' % (args.key))
        if not os.path.isdir(output_dir):
            logger.info('Making output directory %s...', output_dir)
            os.makedirs(output_dir)

    # Generate and write episodes.
    logger.info('Start running...')
    env.reset()
    num_episodes_this_file = 0
    for episode_index, episode in generate_episodes(
            env,
            policy,
            num_steps=args.num_steps,
            num_episodes=args.num_episodes,
            timeout=args.timeout,
            debug=args.debug):

        if args.output_dir:
            # Create a file for saving the episode data.
            if num_episodes_this_file == 0:
                timestamp = time_utils.get_timestamp_as_string()
                filename = 'episodes_%s.hdf5' % (timestamp)
                output_path = os.path.join(output_dir, filename)
                logger.info('Created a new file %s...', output_path)

            # Append the episode to the file.
            logger.info('Saving episode %d to file %s (%d / %d)...',
                        episode_index,
                        output_path,
                        num_episodes_this_file,
                        args.num_episodes_per_file)

            with h5py.File(output_path, 'a') as fout:
                name = str(uuid.uuid4())
                group = fout.create_group(name)
                hdf5_utils.write_data_to_hdf5(group, episode)

        num_episodes_this_file += 1
        num_episodes_this_file %= args.num_episodes_per_file

        if args.pause:
            input('Press [Enter] to start a new episode.')


if __name__ == '__main__':
    main()
