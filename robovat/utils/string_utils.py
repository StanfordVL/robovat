"""Utilities for environments.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import re


def snakecase_to_camelcase(value):
    """Convert a snakecase string to a camelcase string.

    Args:
        value: The input snakecase string (e.g. 'grasp_env').

    Returns:
        The converted camelcase string (e.g. 'GraspEnv').
    """
    words = value.split('_')
    return ''.join(word.title() for word in words)


def camelcase_to_snakecase(value):
    """Convert a camelcase string to a snakecase string.

    Args:
        value: The converted camelcase string (e.g. 'GraspEnv').

    Returns:
        The input snakecase string (e.g. 'grasp_env').
    """
    s1 = re.sub('(.)([0-9]*[A-Z][a-z]+)', r'\1_\2', value)
    return re.sub('([a-z])([A-Z])', r'\1_\2', s1).lower()
