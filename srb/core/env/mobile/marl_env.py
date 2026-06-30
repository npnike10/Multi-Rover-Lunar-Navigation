import copy
from dataclasses import MISSING

from srb.core.asset import AssetVariant, Humanoid, MobileRobot
from srb.core.env import BaseEventCfg, BaseSceneCfg, DirectMarlEnv, DirectMarlEnvCfg
from srb.core.marker import RED_ARROW_X_MARKER_CFG
from srb.core.sensor import Imu, ImuCfg
from srb.utils.cfg import configclass


@configclass
class MobileMarlSceneCfg(BaseSceneCfg):
    # MARL: robots are registered per-agent as robot_{agent_id}; no single .robot field
    robot: None = None  # type: ignore


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

    def _add_robot(self, **kwargs):
        from srb.core.action import ActionGroup
        from srb.core.env.common.base.env_cfg import BaseEnvCfg
        
        # Temporarily store original self.robot just in case
        original_robot = getattr(self, "robot", None)
        
        # We accumulate all action terms across robots into a shared ActionGroup.
        # The base _add_robot() resets self.actions = ActionGroup() on each call,
        # so we save/merge after each iteration.
        accumulated_actions = ActionGroup()
        map_cmd_fns = []
        
        for agent_id, robot_cfg in self.robots.items():
            self.robot = copy.deepcopy(robot_cfg)
            # Call the base class method to instantiate scene elements and ActionGroup configs,
            # but specify unique prim paths per agent
            BaseEnvCfg._add_robot(
                self, 
                prim_path=f"{{ENV_REGEX_NS}}/robot_{agent_id}",
                prim_path_manipulator=f"{{ENV_REGEX_NS}}/manipulator_{agent_id}",
                prim_path_payload=f"{{ENV_REGEX_NS}}/payload_{agent_id}",
                prim_path_end_effector=f"{{ENV_REGEX_NS}}/end_effector_{agent_id}",
            )
            
            # After each call, self.actions only contains action terms for this robot.
            # Merge them into the accumulated group.
            from srb.core.manager import ActionTermCfg
            for attr_name, attr_val in self.actions.__dict__.items():
                if isinstance(attr_val, ActionTermCfg):
                    setattr(accumulated_actions, attr_name, copy.deepcopy(attr_val))
            if hasattr(self.actions, 'map_cmd_to_action'):
                map_cmd_fns.append(self.actions.map_cmd_to_action)
        
        # Replace with the accumulated group containing all robots' action terms
        self.actions = accumulated_actions
        if map_cmd_fns:
            import torch
            self.actions.map_cmd_to_action = lambda twist, event: torch.cat(
                [func(twist, event) for func in map_cmd_fns]
            )
            
        # Restore or clean up
        if original_robot is not None:
            self.robot = original_robot


class MobileMarlEnv(DirectMarlEnv):
    cfg: MobileMarlEnvCfg

    def __init__(self, cfg: MobileMarlEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)
        
        ## Get scene assets
        # We dynamically load IMUs for each robot if they were added to the scene
        self._imus = {
            agent_id: getattr(self.scene, f"imu_{agent_id}")
            for agent_id in self.cfg.possible_agents
            if hasattr(self.scene, f"imu_{agent_id}")
        }
        
        # Get robot articulations
        self._robots = {
            agent_id: self.scene[f"robot_{agent_id}"]
            for agent_id in self.cfg.possible_agents
            if f"robot_{agent_id}" in self.scene.keys()
        }
