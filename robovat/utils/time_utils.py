"""Utilities for timing.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import signal
import time


class TimeoutException(Exception):
    """Exception for timeout.
    """
    pass
 

class Timeout():
    """Timeout class using ALARM signal.
    """
 
    def __init__(self, sec):
        self.sec = sec
 
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.raise_timeout_exception)
        signal.alarm(self.sec)
 
    def __exit__(self, *args):
        signal.alarm(0)    # disable alarm
 
    def raise_timeout_exception(self, *args):
        raise TimeoutException()


def get_timestamp_as_string():
    """Get the current timestamp as a string.

    Returns:
        The timestamp as a string of the format YYYYMMDD_hhmmss_ffffff.
    """
    t = time.time()
    return datetime.datetime.fromtimestamp(t).strftime('%Y%m%d_%H%M%S_%f')
