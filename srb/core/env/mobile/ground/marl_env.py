from dataclasses import MISSING

import torch

from srb.core.action import WheeledDriveAction
from srb.core.asset import AssetVariant, GroundRobot
from srb.core.env import ViewerCfg
from srb.core.env.mobile.marl_env import (
    MobileMarlEnv,
    MobileMarlEnvCfg,
    MobileMarlEventCfg,
    MobileMarlSceneCfg,
)
from srb.core.manager import EventTermCfg, SceneEntityCfg
from srb.core.mdp import reset_root_state_uniform
from srb.utils.cfg import configclass
from srb.utils.math import deg_to_rad


@configclass
class GroundMarlSceneCfg(MobileMarlSceneCfg):
    env_spacing: float = 32.0


@configclass
class GroundMarlEventCfg(MobileMarlEventCfg):
    # For MARL, we need to reset all robots. We will construct this dynamically
    # based on the robots dictionary in __post_init__ of GroundMarlEnvCfg
    pass


@configclass
class GroundMarlEnvCfg(MobileMarlEnvCfg):
    ## Scene
    scene: GroundMarlSceneCfg = GroundMarlSceneCfg()

    ## Events
    events: GroundMarlEventCfg = GroundMarlEventCfg()

    ## Time
    env_rate: float = 1.0 / 50.0
    agent_rate: float = 1.0 / 25.0

    ## Viewer
    viewer: ViewerCfg = ViewerCfg(
        eye=(7.5, -7.5, 15.0), lookat=(0.0, 0.0, 0.0), origin_type="env"
    )

    def __post_init__(self):
        super().__post_init__()

        # Dynamically add reset events for all robots
        for agent_id in self.robots.keys():
            setattr(
                self.events,
                f"randomize_{agent_id}_state",
                EventTermCfg(
                    func=reset_root_state_uniform,
                    mode="reset",
                    params={
                        "asset_cfg": SceneEntityCfg(agent_id),
                        "pose_range": {
                            "x": (-0.5, 0.5),
                            "y": (-0.5, 0.5),
                            "z": (0.4, 0.6),
                            "yaw": (-torch.pi, torch.pi),
                        },
                        "velocity_range": {
                            "x": (-0.5, 0.5),
                            "y": (-0.5, 0.5),
                            "z": (0.0, 0.5),
                            "roll": (-deg_to_rad(5.0), deg_to_rad(5.0)),
                            "pitch": (-deg_to_rad(5.0), deg_to_rad(5.0)),
                            "yaw": (-deg_to_rad(15.0), deg_to_rad(15.0)),
                        },
                    },
                )
            )


class GroundMarlEnv(MobileMarlEnv):
    cfg: GroundMarlEnvCfg

    def __init__(self, cfg: GroundMarlEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)
