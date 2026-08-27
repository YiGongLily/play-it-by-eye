"""
Python module serving as a project/extension template.
"""

import os

# Register Gym environments.
from .tasks import *

LEGGED_LAB_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
