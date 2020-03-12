"""Logging utilites.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import logging
import logging.config


try:
    config_path = os.path.join('configs', 'logging.config')
    logging.config.fileConfig(config_path)
except Exception:
    print('Unable to set the formatters for logging.')


logger = logging.getLogger('root')
