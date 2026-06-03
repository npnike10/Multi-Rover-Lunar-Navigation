from dataclasses import MISSING

from srb.core.asset import AssetVariant, Humanoid, MobileRobot
from srb.core.env import BaseEventCfg, BaseSceneCfg, DirectMarlEnv, DirectMarlEnvCfg
from srb.core.marker import RED_ARROW_X_MARKER_CFG
from srb.core.sensor import Imu, ImuCfg
from srb.utils.cfg import configclass


@configclass
class MobileMarlSceneCfg(BaseSceneCfg):
    pass
    # We will let the specific tasks define the IMUs dynamically in __post_init__ 
    # since we have multiple robots.


@configclass
class MobileMarlEventCfg(BaseEventCfg):
    pass


@configclass
class MobileMarlEnvCfg(DirectMarlEnvCfg):
    ## Assets
    # For MARL, we define a dictionary of robots
    robots: dict[str, MobileRobot | AssetVariant] = MISSING  # type: ignore

    ## Scene
    scene: MobileMarlSceneCfg = MobileMarlSceneCfg()

    ## Events
    events: MobileMarlEventCfg = MobileMarlEventCfg()

    def __post_init__(self):
        super().__post_init__()
        
        # Populate possible agents
        if not hasattr(self, "possible_agents") or self.possible_agents is None:
            self.possible_agents = list(self.robots.keys())
            
        # We handle IMU assignments in the task config itself because 
        # it requires injecting dynamic fields into the SceneCfg for each robot.


class MobileMarlEnv(DirectMarlEnv):
    cfg: MobileMarlEnvCfg

    def __init__(self, cfg: MobileMarlEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)
        
        ## Get scene assets
        # We dynamically load IMUs for each robot if they were added to the scene
        self._imus = {
            agent_id: self.scene[f"imu_{agent_id}"]
            for agent_id in self.cfg.possible_agents
            if f"imu_{agent_id}" in self.scene.keys()
        }
