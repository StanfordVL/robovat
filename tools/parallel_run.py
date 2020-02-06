#!/usr/bin/env python

"""Run multiple copies of the specified command.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import atexit
import os
import socket
import subprocess

import _init_paths  # NOQA
from robovat.utils.logging import logger


def parse_args():
    """Parse arguments.

    Returns:
        args: The parsed arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
            '--num_copies',
            dest='num_copies',
            type=int,
            help='Number of copies to run.',
            required=True)

    parser.add_argument(
            '--command',
            dest='command',
            type=str,
            help='The command to run.',
            required=True)

    parser.add_argument(
            '--log',
            dest='log_dir',
            type=str,
            help='The output directory to save the episode history.',
            required=True)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    if os.path.isdir(args.log_dir):
        logger.warn('Output directory %s already exists.',
                    os.path.abspath(args.log_dir))

    processes = []
    for key in range(args.num_copies):
        logger.info('Starting copy %d / %d...', key, args.num_copies)

        hostname = socket.gethostname()
        hostname = hostname.split('.')[0]
        log_dir = os.path.abspath(args.log_dir)
        log_dir = os.path.join(log_dir, 'logs')
        if not os.path.isdir(log_dir):
            logger.info('Making output directory %s...', log_dir)
            os.makedirs(log_dir)

        log_path = os.path.join(log_dir, '%s_%02d.log' % (hostname, key))

        command = '%s >> %s' % (args.command, log_path)
        logger.info('Run the command: %s', command)
        process = subprocess.Popen(command, shell=True)
        processes.append(process)

    def cleanup():
        for ind, process in enumerate(processes):
            process.kill()
            logger.info('Killed process %d.', ind)

    atexit.register(cleanup)

    logger.info('All copies have been started. Running episode generation...')

    while True:
        pass


if __name__ == '__main__':
    main()
