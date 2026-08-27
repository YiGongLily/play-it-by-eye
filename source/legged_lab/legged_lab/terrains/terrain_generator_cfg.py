# Adapted from:
#   instinctlab (CC BY-NC 4.0)
#     https://github.com/project-instinct/instinctlab
#   Isaac Lab (BSD-3-Clause)
#     https://github.com/isaac-sim/IsaacLab

from isaaclab.terrains import TerrainGeneratorCfg as TerrainGeneratorCfgBase
from isaaclab.utils.configclass import configclass

from .terrain_generator import FiledTerrainGenerator


@configclass
class FiledTerrainGeneratorCfg(TerrainGeneratorCfgBase):
    class_type: type = FiledTerrainGenerator
