"""Define the abstract class of physics backend.

A physics backend is an API wrapper that handles all the physics simulation
utilities.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc


class Physics(object):
    """Base class for physics backends."""

    __metaclass__ = abc.ABCMeta

    def __init__(self):
        return
