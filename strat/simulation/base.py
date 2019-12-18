"""The Base class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import os.path


class Base(object):
    """The base class for simulatables."""

    __metaclass__ = abc.ABCMeta

    def __init__(self, simulator, name=None):
        """Initialize.

        Args:
            simulator: The simulator of the base.
            name: The name of the base.
        """
        self._simulator = simulator
        self._physics = None

        self._uid = None
        self._name = name
        self._scoped_name = None

    @property
    def simulator(self):
        return self._simulator

    @property
    def physics(self):
        if self._physics is None:
            self._physics = self.simulator.physics
        return self._physics

    @property
    def uid(self):
        return self._uid

    @property
    def name(self):
        return self._name

    @property
    def scoped_name(self):
        if self._scoped_name is None:
            if self._parent is None:
                self._scoped_name = self._name
            else:
                self._scoped_name = os.path.join(
                    self.parent.scoped_name, self._name)
        return self._scoped_name
