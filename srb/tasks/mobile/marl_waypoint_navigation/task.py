"""MARL Waypoint Navigation Task for Space Robotics Bench.

Multi-agent Dec-POMDP environment with 3 lunar rovers (1 supporter + 2
explorers) on procedurally generated Moon terrain. Explorers navigate to
individual waypoints; the supporter has no waypoint and must learn to
assist the team through the shared reward signal.

All agents receive the **exact same scalar reward** at each timestep (Dec-POMDP):
    R = w_progress * mean(-explorer_goal_distance / target_spawn_radius)
      + w_goal     * (reached_exp1 + reached_exp2) / 2
      - w_proximity * mean(proximity_penalty over all rover pairs)
      - w_action    * mean(action_rate_penalty over all agents)

Global State (198 dims by default, CTDE centralized critic):
    [Pose_sup(9), Pose_exp1(9), Pose_exp2(9),
     Vel_sup(6),  Vel_exp1(6),  Vel_exp2(6),
     Target_exp1(3), Target_exp2(3),
     Terrain_sup(49), Terrain_exp1(49), Terrain_exp2(49)]

Local Observation (67 dims by default per agent, decentralized actor):
    [task_xy(2), other_rover1_xy(2), other_rover2_xy(2),
     lin_vel_b(3), imu_lin_acc(3), imu_ang_vel(3), projected_gravity(3),
     terrain(49)]

Actions (2 dims per agent): [linear_velocity, angular_velocity]

Termination: both explorers reach targets OR any rover rolls over.
Truncation:  time limit exceeded.
"""

import math
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
from srb.core.marker import RED_ARROW_X_MARKER_CFG
from srb.core.sensor import ImuCfg

# Import Isaac Lab sensor configs
from isaaclab.sensors.ray_caster import RayCasterCfg, patterns

from srb.utils.cfg import configclass
from srb.utils.math import matrix_from_quat, subtract_frame_transforms, quat_to_rot6d

###############################################################################
# Scene Configuration
###############################################################################

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg


@configclass
class MarlWaypointSceneCfg(GroundMarlSceneCfg):
    target_1: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/target_1",
        spawn=sim_utils.ConeCfg(
            radius=0.2,
            height=0.4,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
                enable_gyroscopic_forces=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=PreviewSurfaceCfg(
                diffuse_color=(0.0, 0.0, 1.0), emissive_color=(0.0, 0.0, 1.0)
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 5.0)),
    )

    target_2: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/target_2",
        spawn=sim_utils.ConeCfg(
            radius=0.2,
            height=0.4,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
                enable_gyroscopic_forces=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=PreviewSurfaceCfg(
                diffuse_color=(0.0, 1.0, 0.0), emissive_color=(0.0, 1.0, 0.0)
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 5.0)),
    )


###############################################################################
# Configuration
###############################################################################


@configclass
class MarlWaypointTaskCfg(GroundMarlEnvCfg):
    """Configuration for the MARL Waypoint Navigation environment.

    Dec-POMDP with 3 lunar rovers (1 supporter + 2 explorers) on procedurally
    generated Moon terrain. All agents receive the same shared reward.

    Attributes:
        goal_reached_threshold: Distance (m) within which an explorer is
            considered to have reached its target. Default 0.5 m.
        target_spawn_radius: Maximum distance (m) from env origin at which
            explorer targets are randomly spawned. Default 5.0 m.
        target_spawn_min_radius: Minimum distance (m) from env origin at
            which explorer targets are spawned. Ensures targets are not too
            close to the starting area. Default 2.0 m.
        target_min_separation: Minimum distance (m) between the two explorer
            targets. Ensures explorers must navigate to distinct locations.
            Default 3.0 m. Uses rejection sampling (max 100 attempts).
        safe_distance: Minimum safe distance (m) between any two rovers.
            If None (default), automatically set to the length of the longest
            rover among all agents. Can be overridden with a float.
        rollover_threshold_rad: Tilt angle (radians) beyond which a rover is
            considered rolled over and the episode terminates. Default 1.31
            (~75 degrees).
        rollover_debounce_steps: Number of consecutive timesteps a rover must
            exceed the rollover threshold before the episode terminates.
            Prevents transient bounces from triggering termination. Default 5
            (0.2s at 25 Hz agent rate).
        w_progress: Weight for mean negative normalized explorer goal distance.
            Default 1.0.
        w_goal: Weight for per-explorer goal-reached bonus. Default 5.0.
        w_proximity: Weight for inter-rover proximity penalty. Default 0.5.
        w_action: Weight for action-rate smoothness penalty. Default 0.1.
        terrain_grid_size: RayCaster grid footprint (length, width) in meters.
            Default (1.5, 1.5).
        terrain_grid_resolution: RayCaster grid spacing in meters. Default
            0.25, producing a 7x7 grid over the default footprint.
        raycaster_max_distance: Maximum downward ray distance in meters.
            Default 2.0.
    """

    # -- Scene ----------------------------------------------------------------
    scene: MarlWaypointSceneCfg = MarlWaypointSceneCfg(env_spacing=32.0)

    # -- Terrain --------------------------------------------------------------
    from srb.core.asset import Scenery

    scenery: Scenery | AssetVariant = AssetVariant.PROCEDURAL
    debug_flat_scenery: bool = False
    terrain_grid_size: tuple[float, float] = (1.5, 1.5)
    terrain_grid_resolution: float = 0.25
    raycaster_max_distance: float = 2.0

    # -- Rovers ---------------------------------------------------------------
    robots = {
        "supporter": assets.LeoRover(),
        "explorer_1": assets.LeoRover(),
        "explorer_2": assets.LeoRover(),
    }

    # -- Agent role classification --------------------------------------------
    explorer_agents: list = ["explorer_1", "explorer_2"]
    supporter_agent: str = "supporter"

    # -- Episode --------------------------------------------------------------
    episode_length_s: float = 60.0
    is_finite_horizon: bool = False

    # -- Goal parameters ------------------------------------------------------
    goal_reached_threshold: float = 0.5  # meters
    target_spawn_radius: float = 5.0  # max meters from env origin
    target_spawn_min_radius: float = 2.0  # min meters from env origin
    target_min_separation: float = 3.0  # min meters between the two targets

    # -- Safety parameters ----------------------------------------------------
    safe_distance: float | None = None  # None = auto (longest rover length)
    rollover_threshold_rad: float = 1.31  # ~75 degrees
    rollover_debounce_steps: int = 5  # 0.2s at 25 Hz

    # -- Dec-POMDP reward weights (single shared reward) ----------------------
    w_progress: float = 1.0  # mean negative normalized explorer goal distance
    w_goal: float = 1.0  # per-explorer goal-reached bonus
    w_proximity: float = 1.0  # inter-rover proximity penalty
    w_action: float = 0.1  # action-rate smoothness penalty
    debug_metrics: bool = True
    live_reward_debug: bool = False
    live_reward_debug_interval: int = 25
    live_action_debug: bool = False
    live_action_debug_interval: int = 25

    # -- Delays ---------------------------------------------------------------
    action_delay_steps: int = 0
    observation_delay_steps: int = 0

    # -- Space metadata -------------------------------------------------------
    # Obs per agent : 18 non-terrain dims + terrain_num_rays
    # State (global): 51 non-terrain dims + num_agents * terrain_num_rays
    # Act per agent : 2    (linear + angular velocity)
    observation_spaces: dict = None  # type: ignore[assignment]
    action_spaces: dict = None  # type: ignore[assignment]
    state_space: int = 0  # set in __post_init__

    @staticmethod
    def _terrain_axis_ray_count(size: float, resolution: float) -> int:
        """Match Isaac Lab GridPatternCfg's inclusive arange ray count."""
        if size <= 0.0:
            raise ValueError(f"terrain_grid_size values must be > 0. Received: {size}")
        if resolution <= 0.0:
            raise ValueError(
                f"terrain_grid_resolution must be > 0. Received: {resolution}"
            )
        return math.floor(size / resolution + 1.0e-9) + 1

    @property
    def terrain_num_rays(self) -> int:
        x_count = self._terrain_axis_ray_count(
            self.terrain_grid_size[0], self.terrain_grid_resolution
        )
        y_count = self._terrain_axis_ray_count(
            self.terrain_grid_size[1], self.terrain_grid_resolution
        )
        return x_count * y_count

    @property
    def local_observation_dim(self) -> int:
        return 18 + self.terrain_num_rays

    @property
    def global_state_dim(self) -> int:
        return 51 + len(self.robots) * self.terrain_num_rays

    def __post_init__(self):
        if self.debug_flat_scenery:
            self.scenery = assets.GroundPlane()

        # Terrain must be stacked so the single /World/scenery mesh is
        # visible to all RayCasters.
        self.stack = True

        # Populate space metadata BEFORE super().__post_init__()
        agent_ids = list(self.robots.keys())
        self.observation_spaces = {aid: self.local_observation_dim for aid in agent_ids}
        self.action_spaces = {aid: 2 for aid in agent_ids}
        self.possible_agents = agent_ids

        self.state_space = self.global_state_dim

        super().__post_init__()

        # -- Fix base GroundMarlEnv randomize events --
        # Keep the rovers separated and reset them above the terrain.
        # Starting near z=0 can place wheels/chassis inside uneven procedural
        # terrain, which is enough to poison GPU PhysX with CUDA error 700.
        spawn_offsets = {
            "supporter": (-0.5, 0.0),
            "explorer_1": (0.5, -0.5),
            "explorer_2": (0.5, 0.5),
        }
        for agent_id in self.robots.keys():
            event_cfg = getattr(self.events, f"randomize_{agent_id}_state")
            dx, dy = spawn_offsets.get(agent_id, (0.0, 0.0))
            event_cfg.params["pose_range"]["x"] = (dx - 0.2, dx + 0.2)
            event_cfg.params["pose_range"]["y"] = (dy - 0.2, dy + 0.2)
            event_cfg.params["pose_range"]["z"] = (0.4, 0.5)
            event_cfg.params["velocity_range"]["z"] = (0.0, 0.0)
            event_cfg.params["velocity_range"]["roll"] = (0.0, 0.0)
            event_cfg.params["velocity_range"]["pitch"] = (0.0, 0.0)

        # ---- Per-rover sensor injection ----
        for agent_id, robot_cfg in self.robots.items():
            # RayCaster: downward-facing terrain grid. Default is 7x7 rays
            # over a 1.5 m x 1.5 m footprint.
            setattr(
                self.scene,
                f"raycaster_{agent_id}",
                RayCasterCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/robot_{agent_id}/chassis",
                    update_period=self.agent_rate,
                    offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.5)),
                    mesh_prim_paths=["/World/scenery"],
                    pattern_cfg=patterns.GridPatternCfg(
                        resolution=self.terrain_grid_resolution,
                        size=self.terrain_grid_size,
                        direction=(0.0, 0.0, -1.0),
                    ),
                    max_distance=self.raycaster_max_distance,
                    debug_vis=False,
                ),
            )

            # IMU: attached to chassis (frame_base) for each rover.
            # Provides lin_acc_b, ang_vel_b, and projected_gravity_b.
            # gravity_bias=(0,0,0) matches SRB convention (no bias added).
            prim_path = f"{{ENV_REGEX_NS}}/robot_{agent_id}/chassis"
            imu_cfg = ImuCfg(
                prim_path=prim_path,
                gravity_bias=(0.0, 0.0, 0.0),
                visualizer_cfg=RED_ARROW_X_MARKER_CFG.replace(
                    prim_path=f"/Visuals/imu_{agent_id}/lin_acc"
                ),
            )
            # Use frame_imu offset if available, otherwise frame_base
            if hasattr(robot_cfg, "frame_imu") and robot_cfg.frame_imu is not None:
                imu_cfg.offset.pos = robot_cfg.frame_imu.offset.pos
                imu_cfg.offset.rot = robot_cfg.frame_imu.offset.rot
            elif hasattr(robot_cfg, "frame_base") and robot_cfg.frame_base is not None:
                imu_cfg.offset.pos = robot_cfg.frame_base.offset.pos
                imu_cfg.offset.rot = robot_cfg.frame_base.offset.rot

            setattr(self.scene, f"imu_{agent_id}", imu_cfg)


###############################################################################
# Task
###############################################################################


class MarlWaypointTask(GroundMarlEnv):
    """Multi-Agent Waypoint Navigation on procedural lunar terrain (Dec-POMDP).

    MDP Components
    --------------
    * ``_get_observations``:  local obs per agent (67 dims by default).
    * ``_get_states``:        privileged global state for CTDE critic
                              (198 dims by default).
    * ``_get_rewards``:       Single shared Dec-POMDP reward for all agents.
    * ``_get_dones``:         Terminate on joint goal-reach or rollover.
    """

    cfg: MarlWaypointTaskCfg

    def __init__(self, cfg: MarlWaypointTaskCfg, **kwargs):
        super().__init__(cfg, **kwargs)

        self.action_manager = ActionManager(self.cfg.actions, env=self)
        self._wheeled_drive_terms = {
            aid: self.action_manager._terms.get(f"robot_{aid}/wheeled_drive")
            for aid in self.cfg.possible_agents
        }

        # -- Per-agent action buffers (for action-rate penalty) --
        self._actions_dict = {
            aid: torch.zeros(self.num_envs, 2, device=self.device)
            for aid in self.cfg.possible_agents
        }
        self._prev_actions_dict = {
            aid: torch.zeros(self.num_envs, 2, device=self.device)
            for aid in self.cfg.possible_agents
        }

        # -- RayCaster handles --
        self._raycasters = {
            aid: self.scene[f"raycaster_{aid}"] for aid in self.cfg.possible_agents
        }

        # -- IMU handles --
        # Retrieved from scene; the base MobileMarlEnv.__init__ populates
        # self._imus if the sensors were added to the scene config.
        # We ensure they exist here as well.
        if not self._imus:
            self._imus = {
                aid: self.scene[f"imu_{aid}"] for aid in self.cfg.possible_agents
            }

        # -- Goal buffers (explorers only — supporter has no waypoint) --
        self._goals = {
            aid: torch.zeros(self.num_envs, 3, device=self.device)
            for aid in self.cfg.explorer_agents
        }
        for aid in self.cfg.explorer_agents:
            self._goals[aid][:, :3] = self.scene.env_origins[:, :3]

        # -- Per-explorer "reached" flag --
        self._explorer_reached = {
            aid: torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            for aid in self.cfg.explorer_agents
        }
        self._prev_goal_distances = {
            aid: torch.zeros(self.num_envs, device=self.device)
            for aid in self.cfg.explorer_agents
        }
        self._last_goal_distances = {
            aid: torch.zeros(self.num_envs, device=self.device)
            for aid in self.cfg.explorer_agents
        }

        # -- Rollover debounce counter (per rover, per env) --
        self._rollover_count = {
            aid: torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
            for aid in self.cfg.possible_agents
        }

        # -- Safe distance (auto-compute from rover extent if not specified) --
        if self.cfg.safe_distance is not None:
            self._safe_distance = self.cfg.safe_distance
        else:
            # Compute max rover extent: distance from root to farthest body
            # link, doubled to get full length. This gives a conservative
            # estimate of the rover's physical size.
            max_extent = 0.0
            for aid in self.cfg.possible_agents:
                robot = self._robots[aid]
                body_pos = robot.data.body_pos_w[0]  # (num_bodies, 3)
                root_pos = robot.data.root_pos_w[0]  # (3,)
                extents = torch.norm(body_pos - root_pos, dim=-1)
                max_extent = max(max_extent, extents.max().item() * 2.0)
            self._safe_distance = max_extent
            print(
                f"[MarlWaypointTask] Auto safe_distance = {self._safe_distance:.3f} m"
            )

        # -- Ordered list of "other agents" per agent (for inter-rover obs) --
        self._other_agents = {
            aid: [a for a in self.cfg.possible_agents if a != aid]
            for aid in self.cfg.possible_agents
        }
        self._debug_action_metrics: dict[str, torch.Tensor] = {}
        self._debug_termination_metrics: dict[str, torch.Tensor] = {}

    # --------------------------------------------------------------------- #
    #  Reset                                                                  #
    # --------------------------------------------------------------------- #

    def _sample_target_xy(self, n: int) -> torch.Tensor:
        """Sample n target XY offsets in an annular region.

        Samples uniformly by area within the annulus defined by
        [target_spawn_min_radius, target_spawn_radius]. Targets are placed
        at ground level (env_origins z).

        Returns:
            ``(n, 2)`` tensor of XY offsets from env origin.
        """
        r_min = self.cfg.target_spawn_min_radius
        r_max = self.cfg.target_spawn_radius
        # Uniform sampling by area: r = sqrt(U * (r_max² - r_min²) + r_min²)
        u = torch.rand(n, device=self.device)
        r = torch.sqrt(u * (r_max**2 - r_min**2) + r_min**2)
        theta = torch.rand(n, device=self.device) * 2.0 * torch.pi
        xy = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=-1)
        return xy  # (n, 2)

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)

        n = len(env_ids)
        origins = self.scene.env_origins[env_ids]  # (n, 3)
        explorer_ids = self.cfg.explorer_agents  # ["explorer_1", "explorer_2"]

        # -- Reset explorer targets with min separation constraint --
        # Sample target 1
        xy1 = self._sample_target_xy(n)  # (n, 2)

        # Sample target 2 with rejection sampling to enforce min separation
        xy2 = self._sample_target_xy(n)  # (n, 2)
        for attempt in range(100):
            dist = torch.norm(xy1 - xy2, dim=-1)  # (n,)
            too_close = dist < self.cfg.target_min_separation
            if not too_close.any():
                break
            # Resample only the environments where targets are too close
            m = too_close.sum().item()
            xy2[too_close] = self._sample_target_xy(m)

        # Assign targets (at ground level = env_origins z)
        for i, aid in enumerate(explorer_ids):
            xy = xy1 if i == 0 else xy2
            self._goals[aid][env_ids, 0] = origins[:, 0] + xy[:, 0]
            self._goals[aid][env_ids, 1] = origins[:, 1] + xy[:, 1]
            self._goals[aid][env_ids, 2] = origins[:, 2]  # ground level

            # Update visual target poses in simulation
            target_obj = self.scene[f"target_{i+1}"]
            target_pose = torch.zeros((len(env_ids), 7), device=self.device)
            target_pose[:, :3] = self._goals[aid][env_ids]
            target_pose[:, 2] += 1.5  # Hover 1.5m above ground
            target_pose[:, 3:7] = torch.tensor(
                [0.0, 0.0, 1.0, 0.0], device=self.device
            )  # Rotate 180 deg (point down)
            target_obj.write_root_pose_to_sim(target_pose, env_ids=env_ids)

        # -- Reset flags and buffers --
        for aid in self.cfg.explorer_agents:
            self._explorer_reached[aid][env_ids] = False
        for aid in self.cfg.possible_agents:
            self._actions_dict[aid][env_ids] = 0.0
            self._prev_actions_dict[aid][env_ids] = 0.0
            self._rollover_count[aid][env_ids] = 0
        for aid in self.cfg.explorer_agents:
            pos_w = self._robots[aid].data.root_link_pose_w[env_ids, :3]
            dist2d = torch.norm(pos_w[:, :2] - self._goals[aid][env_ids, :2], dim=-1)
            self._prev_goal_distances[aid][env_ids] = dist2d
            self._last_goal_distances[aid][env_ids] = dist2d

    # --------------------------------------------------------------------- #
    #  Actions                                                                #
    # --------------------------------------------------------------------- #

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        flat_parts = []
        for aid in self.cfg.possible_agents:
            act = actions[aid].to(self.device)
            self._prev_actions_dict[aid][:] = self._actions_dict[aid]
            self._actions_dict[aid][:] = act
            flat_parts.append(act)

        self.action_manager.process_action(torch.cat(flat_parts, dim=-1))
        self._update_action_debug_metrics()

        if self.cfg.live_action_debug:
            interval = max(1, int(self.cfg.live_action_debug_interval))
            step = int(getattr(self, "common_step_counter", 0))
            if step % interval == 0 and self.num_envs > 0:
                env_idx = 0
                print(
                    f"[actions step={step} env={env_idx}] "
                    f"{self._action_debug_payload(env_idx)}",
                    flush=True,
                )

    def _apply_action(self) -> None:
        self.action_manager.apply_action()

    def _action_debug_payload(self, env_idx: int = 0) -> dict[str, dict[str, float]]:
        payload = {}
        for aid in self.cfg.possible_agents:
            action = self._actions_dict[aid][env_idx]
            entry = {
                "raw_linear_velocity": float(action[0].item()),
                "raw_angular_velocity": float(action[1].item()),
            }
            term = self._wheeled_drive_terms.get(aid)
            processed_actions = getattr(term, "processed_actions", None)
            if processed_actions is not None:
                processed = processed_actions[env_idx]
                entry["processed_linear_velocity"] = float(processed[0].item())
                entry["processed_angular_velocity"] = float(processed[1].item())
            payload[aid] = entry
        return payload

    def _update_action_debug_metrics(self) -> None:
        if not self.cfg.debug_metrics:
            self._debug_action_metrics = {}
            return

        metrics = {}
        for aid in self.cfg.possible_agents:
            action = self._actions_dict[aid]
            action_delta = action - self._prev_actions_dict[aid]
            metrics[f"Debug / Action raw linear mean / {aid}"] = action[:, 0].mean()
            metrics[f"Debug / Action raw angular mean / {aid}"] = action[:, 1].mean()
            metrics[f"Debug / Action raw abs mean / {aid}"] = action.abs().mean()
            metrics[f"Debug / Action rate mean / {aid}"] = (
                action_delta.square().mean(dim=-1).mean()
            )

            term = self._wheeled_drive_terms.get(aid)
            processed_actions = getattr(term, "processed_actions", None)
            if processed_actions is not None:
                metrics[f"Debug / Command linear mean / {aid}"] = processed_actions[
                    :, 0
                ].mean()
                metrics[f"Debug / Command angular mean / {aid}"] = processed_actions[
                    :, 1
                ].mean()
                metrics[f"Debug / Command abs mean / {aid}"] = (
                    processed_actions.abs().mean()
                )

        self._debug_action_metrics = metrics

    # --------------------------------------------------------------------- #
    #  Terrain helper (shared by observations and state)                      #
    # --------------------------------------------------------------------- #

    def _get_terrain_features(self, agent_id: str) -> torch.Tensor:
        """Relative terrain heights from the RayCaster grid.

        Returns:
            ``(num_envs, terrain_num_rays)`` tensor. Negative means
            crater/below sensor, positive means hill/above sensor. NaN/inf
            values are replaced with ``-2.0``.
        """
        rc = self._raycasters[agent_id]
        sensor_z = rc.data.pos_w[:, 2:3]
        hit_z = rc.data.ray_hits_w[..., 2]
        heights = hit_z - sensor_z
        return torch.nan_to_num(heights, nan=-2.0, posinf=-2.0, neginf=-2.0)

    # --------------------------------------------------------------------- #
    #  Observations  (per-agent, decentralized)                               #
    # --------------------------------------------------------------------- #

    def _get_observations(self) -> dict[str, torch.Tensor]:
        """Build local observations for each agent.

        Layout::

            task_xy             (2)  — explorer: rel XY to target; supporter: zeros
            other_rover1_xy     (2)  — rel XY to first other rover  (body frame)
            other_rover2_xy     (2)  — rel XY to second other rover (body frame)
            lin_vel_b           (3)  — body-frame linear velocity
            imu_lin_acc_b       (3)  — IMU linear acceleration (body frame)
            imu_ang_vel_b       (3)  — IMU angular velocity (body frame)
            projected_gravity_b (3)  — gravity direction in body frame
            terrain             (terrain_num_rays) — RayCaster relative heights
        """
        obs = {}

        # Pre-compute all rover world positions for inter-rover observations
        all_pos_w = {
            aid: self._robots[aid].data.root_link_pose_w[:, :3]
            for aid in self.cfg.possible_agents
        }

        # Identity quaternion (reused for targets/positions without orientation)
        id_quat = torch.zeros(self.num_envs, 4, device=self.device)
        id_quat[:, 0] = 1.0

        # Marker positions for batch visualization
        marker_pos_list = []
        marker_quat_list = []

        for aid in self.cfg.possible_agents:
            robot = self._robots[aid]
            pose = robot.data.root_link_pose_w  # (N, 7)
            ego_pos = pose[:, :3]
            ego_quat = pose[:, 3:7]
            imu = self._imus[aid]

            parts = []

            # ---- Task-relative XY (2 dims) ----
            if aid in self.cfg.explorer_agents:
                # Explorer: relative XY to own target in body frame
                goal = self._goals[aid]
                tf_pos, _ = subtract_frame_transforms(
                    t01=ego_pos,
                    q01=ego_quat,
                    t02=goal,
                    q02=id_quat,
                )
                parts.append(tf_pos[:, :2])  # (N, 2)
                # Collect markers
                marker_pos_list.append(goal)
                marker_quat_list.append(id_quat)
            else:
                # Supporter: no target → zeros
                parts.append(torch.zeros(self.num_envs, 2, device=self.device))

            # ---- Other rovers relative XY (2 + 2 = 4 dims) ----
            for other_aid in self._other_agents[aid]:
                other_pos = all_pos_w[other_aid]
                tf_pos, _ = subtract_frame_transforms(
                    t01=ego_pos,
                    q01=ego_quat,
                    t02=other_pos,
                    q02=id_quat,
                )
                parts.append(tf_pos[:, :2])  # (N, 2)

            # ---- Body-frame linear velocity (3 dims) ----
            # In deployment: estimated via sensor fusion (wheel odometry + IMU).
            # In simulation: ground-truth from physics, following SRB convention.
            parts.append(robot.data.root_lin_vel_b)  # (N, 3)

            # ---- IMU readings (3 + 3 + 3 = 9 dims) ----
            parts.append(imu.data.lin_acc_b)  # (N, 3) linear acceleration
            parts.append(imu.data.ang_vel_b)  # (N, 3) angular velocity
            parts.append(imu.data.projected_gravity_b)  # (N, 3) gravity direction

            # ---- Terrain features ----
            parts.append(self._get_terrain_features(aid))

            obs[aid] = torch.cat(parts, dim=-1)

        return obs

    # --------------------------------------------------------------------- #
    #  Global State  (CTDE centralized critic)                                #
    # --------------------------------------------------------------------- #

    def _get_states(self) -> torch.Tensor:
        """Build the privileged global state vector.

        Layout (all in env-local frame, orientations as 6D continuous rep)::

            Pose_supporter   (9)  = pos(3) + rot6d(6)
            Pose_explorer_1  (9)
            Pose_explorer_2  (9)
            Vel_supporter    (6)  = lin_vel_w(3) + ang_vel_w(3)
            Vel_explorer_1   (6)
            Vel_explorer_2   (6)
            Target_explorer_1(3)  = position relative to env_origin
            Target_explorer_2(3)
            Terrain_supporter  (terrain_num_rays)  = RayCaster relative heights
            Terrain_explorer_1 (terrain_num_rays)
            Terrain_explorer_2 (terrain_num_rays)
        """
        parts = []

        # -- Poses: position (3) + 6D orientation (6) = 9 per rover --
        for aid in self.cfg.possible_agents:
            robot = self._robots[aid]
            pos_w = robot.data.root_link_pose_w[:, :3]
            quat_w = robot.data.root_link_pose_w[:, 3:7]
            parts.append(pos_w - self.scene.env_origins)  # (N, 3)
            parts.append(quat_to_rot6d(quat_w))  # (N, 6)

        # -- Velocities: linear (3) + angular (3) = 6 per rover, world frame --
        for aid in self.cfg.possible_agents:
            robot = self._robots[aid]
            parts.append(robot.data.root_lin_vel_w)  # (N, 3)
            parts.append(robot.data.root_ang_vel_w)  # (N, 3)

        # -- Explorer target positions, relative to env origin --
        for aid in self.cfg.explorer_agents:
            parts.append(self._goals[aid] - self.scene.env_origins)  # (N, 3)

        # -- Terrain features --
        for aid in self.cfg.possible_agents:
            parts.append(self._get_terrain_features(aid))

        return torch.cat(parts, dim=-1)

    # --------------------------------------------------------------------- #
    #  Rewards  (Dec-POMDP: single shared reward for all agents)              #
    # --------------------------------------------------------------------- #

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        """Compute the shared Dec-POMDP reward.

        All agents receive the **exact same scalar**::

            R = w_progress * mean(-explorer_goal_distance / target_spawn_radius)
              + w_goal     * (reached_exp1 + reached_exp2) / 2
              - w_proximity * mean(proximity_penalty over all rover pairs)
              - w_action    * mean(action_rate over all agents)
        """
        # ---- Explorer progress ----
        progress_terms = []
        distance_terms = {}
        distance_delta_terms = {}
        distance_scale = max(float(self.cfg.target_spawn_radius), 1.0e-6)
        for aid in self.cfg.explorer_agents:
            pos_w = self._robots[aid].data.root_link_pose_w[:, :3]
            dist2d = torch.norm(pos_w[:, :2] - self._goals[aid][:, :2], dim=-1)
            distance_terms[aid] = dist2d
            distance_delta_terms[aid] = self._prev_goal_distances[aid] - dist2d
            progress_terms.append(-dist2d / distance_scale)
            # Update reached flags
            self._explorer_reached[aid] = dist2d < self.cfg.goal_reached_threshold
            self._last_goal_distances[aid] = dist2d

        mean_progress = torch.stack(progress_terms, dim=-1).mean(dim=-1)  # (N,)
        mean_distance_delta = torch.stack(
            [distance_delta_terms[aid] for aid in self.cfg.explorer_agents], dim=-1
        ).mean(dim=-1)

        # ---- Goal-reached bonus (per explorer, averaged) ----
        reached_count = torch.zeros(self.num_envs, device=self.device)
        for aid in self.cfg.explorer_agents:
            reached_count += self._explorer_reached[aid].float()
        goal_bonus = reached_count / len(self.cfg.explorer_agents)  # (N,)

        # ---- Proximity penalty (over all 3 rover pairs) ----
        all_pos = [
            self._robots[aid].data.root_link_pose_w[:, :3]
            for aid in self.cfg.possible_agents
        ]
        pair_penalties = []
        for i in range(len(all_pos)):
            for j in range(i + 1, len(all_pos)):
                dist = torch.norm(all_pos[i] - all_pos[j], dim=-1)
                penalty = torch.clamp(1.0 - dist / self._safe_distance, min=0.0)
                pair_penalties.append(penalty)
        mean_proximity = torch.stack(pair_penalties, dim=-1).mean(dim=-1)  # (N,)

        # ---- Action-rate penalty (all agents, averaged) ----
        action_rates = []
        action_rate_terms = {}
        for aid in self.cfg.possible_agents:
            diff = (self._actions_dict[aid] - self._prev_actions_dict[aid]).square()
            action_rate = diff.mean(dim=-1)
            action_rate_terms[aid] = action_rate
            action_rates.append(action_rate)
        mean_action_rate = torch.stack(action_rates, dim=-1).mean(dim=-1)  # (N,)

        # ---- Shared reward ----
        shared_reward = (
            self.cfg.w_progress * mean_progress
            + self.cfg.w_goal * goal_bonus
            - self.cfg.w_proximity * mean_proximity
            - self.cfg.w_action * mean_action_rate
        )

        self._update_episode_debug_metrics(
            distance_terms=distance_terms,
            distance_delta_terms=distance_delta_terms,
            action_rate_terms=action_rate_terms,
            mean_progress=mean_progress,
            mean_distance_delta=mean_distance_delta,
            goal_bonus=goal_bonus,
            mean_proximity=mean_proximity,
            mean_action_rate=mean_action_rate,
            shared_reward=shared_reward,
        )

        if self.cfg.live_reward_debug:
            interval = max(1, int(self.cfg.live_reward_debug_interval))
            step = int(getattr(self, "common_step_counter", 0))
            if step % interval == 0 and self.num_envs > 0:
                env_idx = 0
                progress = self.cfg.w_progress * mean_progress[env_idx]
                goal = self.cfg.w_goal * goal_bonus[env_idx]
                proximity = -self.cfg.w_proximity * mean_proximity[env_idx]
                action = -self.cfg.w_action * mean_action_rate[env_idx]
                reached = {
                    aid: bool(self._explorer_reached[aid][env_idx].item())
                    for aid in self.cfg.explorer_agents
                }
                print(
                    "[reward "
                    f"step={step} env={env_idx}] "
                    f"total={shared_reward[env_idx].item():+.4f} "
                    f"progress={progress.item():+.4f} "
                    f"goal={goal.item():+.4f} "
                    f"proximity={proximity.item():+.4f} "
                    f"action={action.item():+.4f} "
                    f"reached={reached}",
                    flush=True,
                )

        # Dec-POMDP: every agent gets the same reward
        for aid in self.cfg.explorer_agents:
            self._prev_goal_distances[aid] = distance_terms[aid]

        return {aid: shared_reward for aid in self.cfg.possible_agents}

    def _update_episode_debug_metrics(
        self,
        *,
        distance_terms: dict[str, torch.Tensor],
        distance_delta_terms: dict[str, torch.Tensor],
        action_rate_terms: dict[str, torch.Tensor],
        mean_progress: torch.Tensor,
        mean_distance_delta: torch.Tensor,
        goal_bonus: torch.Tensor,
        mean_proximity: torch.Tensor,
        mean_action_rate: torch.Tensor,
        shared_reward: torch.Tensor,
    ) -> None:
        if not self.cfg.debug_metrics:
            self.extras.pop("episode", None)
            return

        metrics = {
            **self._debug_action_metrics,
            **self._debug_termination_metrics,
            "Debug / Goal distance mean / explorers": torch.stack(
                [distance_terms[aid] for aid in self.cfg.explorer_agents], dim=-1
            ).mean(),
            "Debug / Goal delta distance mean / explorers": mean_distance_delta.mean(),
            "Debug / Goal reached rate / explorers": goal_bonus.mean(),
            "RewardComponents / progress raw mean": mean_progress.mean(),
            "RewardComponents / progress weighted mean": (
                self.cfg.w_progress * mean_progress
            ).mean(),
            "RewardComponents / goal raw mean": goal_bonus.mean(),
            "RewardComponents / goal weighted mean": (
                self.cfg.w_goal * goal_bonus
            ).mean(),
            "RewardComponents / proximity raw mean": mean_proximity.mean(),
            "RewardComponents / proximity weighted mean": (
                -self.cfg.w_proximity * mean_proximity
            ).mean(),
            "RewardComponents / action raw mean": mean_action_rate.mean(),
            "RewardComponents / action weighted mean": (
                -self.cfg.w_action * mean_action_rate
            ).mean(),
            "RewardComponents / shared reward mean": shared_reward.mean(),
            "Episode / active length mean": self.episode_length_buf.float().mean(),
        }

        for aid in self.cfg.possible_agents:
            robot = self._robots[aid]
            speed_xy = torch.norm(robot.data.root_lin_vel_b[:, :2], dim=-1)
            metrics[f"Debug / Speed xy mean / {aid}"] = speed_xy.mean()
            metrics[f"Debug / Speed xy max / {aid}"] = speed_xy.max()
            metrics[f"Debug / Action rate reward source / {aid}"] = action_rate_terms[
                aid
            ].mean()

        for aid in self.cfg.explorer_agents:
            metrics[f"Debug / Goal distance mean / {aid}"] = distance_terms[aid].mean()
            metrics[f"Debug / Goal distance min / {aid}"] = distance_terms[aid].min()
            metrics[f"Debug / Goal delta distance mean / {aid}"] = distance_delta_terms[
                aid
            ].mean()
            metrics[f"Debug / Goal reached rate / {aid}"] = (
                self._explorer_reached[aid].float().mean()
            )

        self.extras["episode"] = {
            key: value.detach() if isinstance(value, torch.Tensor) else value
            for key, value in metrics.items()
        }

    # --------------------------------------------------------------------- #
    #  Termination / Truncation                                               #
    # --------------------------------------------------------------------- #

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Terminate on joint goal-reach or rollover.

        Rollover detection uses projected gravity from the IMU. The tilt
        must persist for ``rollover_debounce_steps`` consecutive timesteps
        to avoid termination on transient bounces.
        """
        # ---- Success: both explorers reached their targets ----
        both_reached = torch.ones(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        for aid in self.cfg.explorer_agents:
            both_reached &= self._explorer_reached[aid]

        # ---- Rollover: any rover tilted beyond threshold ----
        any_rolled = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        for aid in self.cfg.possible_agents:
            imu = self._imus[aid]
            grav_b = imu.data.projected_gravity_b  # (N, 3)
            # tilt_angle = angle between body Z-axis and world "down"
            # When upright: grav_b[:, 2] ≈ -1 → acos(1) = 0
            # When tilted:  grav_b[:, 2] → 0  → acos(0) ≈ π/2
            tilt = torch.acos(torch.clamp(-grav_b[:, 2], -1.0, 1.0))
            tilted = tilt > self.cfg.rollover_threshold_rad

            # Debounce: increment counter if tilted, reset if not
            self._rollover_count[aid] = torch.where(
                tilted,
                self._rollover_count[aid] + 1,
                torch.zeros_like(self._rollover_count[aid]),
            )
            any_rolled |= self._rollover_count[aid] >= self.cfg.rollover_debounce_steps

        # Combined termination
        terminated = both_reached | any_rolled

        # Shared signal — all agents end together
        termination = {aid: terminated for aid in self.cfg.possible_agents}

        # Time-based truncation (only if not already terminated)
        time_out = self.episode_length_buf >= self.max_episode_length
        time_out_not_terminated = time_out & ~terminated
        done = terminated | time_out_not_terminated
        done_count = done.float().sum()
        truncation = {aid: time_out_not_terminated for aid in self.cfg.possible_agents}

        self._debug_termination_metrics = {
            "Termination / done count": done_count,
            "Termination / success count": (both_reached & done).float().sum(),
            "Termination / rollover count": (any_rolled & done).float().sum(),
            "Termination / timeout count": time_out_not_terminated.float().sum(),
            "Termination / success rate current": (both_reached & done).float().sum()
            / torch.clamp(done_count, min=1.0),
            "Termination / rollover rate current": (any_rolled & done).float().sum()
            / torch.clamp(done_count, min=1.0),
            "Termination / timeout rate current": time_out_not_terminated.float().sum()
            / torch.clamp(done_count, min=1.0),
        }

        return termination, truncation
