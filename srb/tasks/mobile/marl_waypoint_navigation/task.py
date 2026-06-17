from typing import Sequence

import torch
from dataclasses import MISSING

from srb import assets
from srb.core.env.mobile.ground.marl_env import (
    GroundMarlEnv,
    GroundMarlEnvCfg,
    GroundMarlSceneCfg,
    GroundMarlEventCfg,
)
from srb.core.manager import ActionManager
from srb.core.marker import VisualizationMarkers, VisualizationMarkersCfg
from srb.core.sim import PreviewSurfaceCfg
from srb.core.sim.spawners.shapes.extras.cfg import PinnedArrowCfg
from srb.core.asset import AssetVariant

# Import Isaac Lab sensor configs
from isaaclab.sensors.ray_caster import RayCasterCfg, patterns

from srb.utils.cfg import configclass
from srb.utils.math import matrix_from_quat, subtract_frame_transforms


@configclass
class MarlWaypointTaskCfg(GroundMarlEnvCfg):
    # Scene config uses GroundMarlSceneCfg
    scene: GroundMarlSceneCfg = GroundMarlSceneCfg()
    
    # Terrain Configuration
    # By default, SRB uses AssetVariant.PROCEDURAL which loads the simforge_foundry 
    # lunar surface generator (craters, rocks, slopes).
    # To use a simple flat plane (e.g. for initial testing/debugging), change this to:
    # scenery = assets.GroundPlane()
    scenery: assets.Scenery | AssetVariant = AssetVariant.PROCEDURAL
    
    # We define heterogeneous rovers!
    robots = {
        "supporter": assets.LeoRover(),
        "explorer_1": assets.LeoRover(),
        "explorer_2": assets.LeoRover(),
    }
    
    # Target markers
    target_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/target",
        markers={
            "target": PinnedArrowCfg(
                pin_radius=0.01,
                pin_length=2.0,
                tail_radius=0.01,
                tail_length=0.2,
                head_radius=0.04,
                head_length=0.08,
                visual_material=PreviewSurfaceCfg(emissive_color=(0.2, 0.2, 0.8)),
            )
        },
    )

    episode_length_s: float = 60.0
    is_finite_horizon: bool = False
    
    # Action/observation delays
    action_delay_steps: int = 0
    observation_delay_steps: int = 0

    def __post_init__(self):
        super().__post_init__()

        # Dynamically inject RayCasters for each rover into the scene
        for agent_id in self.possible_agents:
            setattr(
                self.scene,
                f"raycaster_{agent_id}",
                RayCasterCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/robot_{agent_id}/chassis",
                    update_period=self.agent_rate,
                    offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.5)), # Lifted 50cm above the chassis center
                    mesh_prim_paths=["/World/.*/scenery", "/World/scenery", "/World/ground"], # Cast only against terrain
                    pattern=patterns.GridPatternCfg(
                        resolution=0.15,
                        size=[1.5, 1.5], # Creates an 11x11 grid (121 points) spanning 1.5m
                        direction=(0.0, 0.0, -1.0) # Pointing straight down
                    ),
                    max_distance=2.0,
                    debug_vis=False,
                )
            )


class MarlWaypointTask(GroundMarlEnv):
    cfg: MarlWaypointTaskCfg

    def __init__(self, cfg: MarlWaypointTaskCfg, **kwargs):
        super().__init__(cfg, **kwargs)

        ## Get scene assets
        self._target_marker = VisualizationMarkers(self.cfg.target_marker_cfg)

        ## Action Manager for heterogeneous mapping
        self.action_manager = ActionManager(self.cfg.actions, env=self)
        
        ## Action tracking for penalties
        self._actions_dict = {
            agent_id: torch.zeros(self.num_envs, 2, device=self.device)
            for agent_id in self.cfg.possible_agents
        }
        self._prev_actions_dict = {
            agent_id: torch.zeros(self.num_envs, 2, device=self.device)
            for agent_id in self.cfg.possible_agents
        }

        ## Extract RayCasters from scene
        self._raycasters = {
            agent_id: getattr(self.scene, f"raycaster_{agent_id}")
            for agent_id in self.cfg.possible_agents
        }

        ## Initialize buffers
        self._goals = {
            agent_id: torch.zeros(self.num_envs, 7, device=self.device)
            for agent_id in self.cfg.possible_agents
        }
        for agent_id in self.cfg.possible_agents:
            self._goals[agent_id][:, 0:3] = self.scene.env_origins
            self._goals[agent_id][:, 3] = 1.0

        # Define Observation and Action Spaces per agent
        import gymnasium
        self.action_spaces = {
            agent_id: gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(2,))
            for agent_id in self.cfg.possible_agents
        }
        self.observation_spaces = {
            # 4 state dims (x, y, dist, angle) + 121 raycast depth dims = 125
            agent_id: gymnasium.spaces.Box(low=-float('inf'), high=float('inf'), shape=(125,))
            for agent_id in self.cfg.possible_agents
        }
        
    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)

        ## Reset goals randomly for each agent
        for agent_id in self.cfg.possible_agents:
            self._goals[agent_id][env_ids, 0:3] = self.scene.env_origins[env_ids] + torch.normal(
                0, 5.0, size=(len(env_ids), 3), device=self.device
            )
            self._goals[agent_id][env_ids, 2] = 0.5 # keep it near ground
            self._goals[agent_id][env_ids, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
            
            self._actions_dict[agent_id][env_ids] = 0.0
            self._prev_actions_dict[agent_id][env_ids] = 0.0

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        flat_actions_list = []
        # Update action buffers and map dict actions to flat tensor for ActionManager
        for agent_id in self.cfg.possible_agents:
            agent_act = actions[agent_id].to(self.device)
            self._prev_actions_dict[agent_id][:] = self._actions_dict[agent_id][:]
            self._actions_dict[agent_id][:] = agent_act
            flat_actions_list.append(agent_act)
            
        # The terms in ActionGroup are registered sequentially by agent id
        flat_actions = torch.cat(flat_actions_list, dim=-1)
        self.action_manager.process_action(flat_actions)

    def _apply_action(self) -> None:
        self.action_manager.apply_action()

    def _get_observations(self) -> dict[str, torch.Tensor]:
        obs_dict = {}
        for agent_id in self.cfg.possible_agents:
            robot = self._robots[agent_id]
            robot_pose = robot.data.root_link_pose_w
            goal = self._goals[agent_id]
            
            tf_pos, _ = subtract_frame_transforms(
                t01=robot_pose[:, 0:3], q01=robot_pose[:, 3:7], 
                t02=goal[:, 0:3], q02=goal[:, 3:7]
            )
            
            # Simple local observation (relative x, y and heading to target)
            pos2d = tf_pos[:, :2]
            dist = torch.norm(pos2d, dim=-1, keepdim=True)
            angle = torch.atan2(tf_pos[:, 1], tf_pos[:, 0]).unsqueeze(-1)
            
            # Terrain heights: compute relative height of each ray hit point 
            # compared to the sensor origin (gives local elevation map)
            raycaster = self._raycasters[agent_id]
            sensor_pos_z = raycaster.data.pos_w[:, 2:3]  # (num_envs, 1)
            hit_z = raycaster.data.ray_hits_w[..., 2]     # (num_envs, num_rays)
            # Relative height: positive = ground above sensor level, negative = below (crater)
            raycast_heights = hit_z - sensor_pos_z
            # Replace inf/nan (no hit) with a safe default 
            raycast_heights = torch.nan_to_num(raycast_heights, nan=-2.0, posinf=-2.0, neginf=-2.0)
            
            obs_dict[agent_id] = torch.cat([pos2d, dist, angle, raycast_heights], dim=-1)
            
            # Visualize
            self._target_marker.visualize(goal[:, 0:3], goal[:, 3:7])
            
        return obs_dict

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        rew_dict = {}
        for agent_id in self.cfg.possible_agents:
            robot = self._robots[agent_id]
            robot_pose = robot.data.root_link_pose_w
            goal = self._goals[agent_id]
            
            tf_pos, _ = subtract_frame_transforms(
                t01=robot_pose[:, 0:3], q01=robot_pose[:, 3:7], 
                t02=goal[:, 0:3], q02=goal[:, 3:7]
            )
            dist2d = torch.norm(tf_pos[:, :2], dim=-1)
            
            # Simple progress reward
            reward = 1.0 / (1.0 + dist2d)
            
            # Action rate penalty
            act_diff = torch.mean(torch.square(self._actions_dict[agent_id] - self._prev_actions_dict[agent_id]), dim=1)
            reward -= 0.1 * act_diff
            
            rew_dict[agent_id] = reward
            
        return rew_dict

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        termination = {
            agent_id: torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            for agent_id in self.cfg.possible_agents
        }
        truncation = {
            agent_id: torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            for agent_id in self.cfg.possible_agents
        }
        return termination, truncation

