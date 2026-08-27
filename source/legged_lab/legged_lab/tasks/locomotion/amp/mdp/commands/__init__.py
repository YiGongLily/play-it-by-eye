"""Custom command terms for the AMP locomotion task.

Only the *cfg* is exported here. The command term (``velocity_command.py``) imports
``isaaclab.markers`` / pxr and must not be imported before ``SimulationApp`` starts; it is
loaded lazily at runtime via the string ``class_type`` path in the cfg.
"""

from .commands_cfg import AmpVelocityCommandCfg  # noqa: F401
