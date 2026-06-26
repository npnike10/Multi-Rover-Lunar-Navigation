"""MARL Waypoint Navigation Task for Space Robotics Bench.

Multi-agent environment with 3 lunar rovers (1 supporter + 2 explorers) on
procedurally generated Moon terrain. Explorers navigate to individual waypoints;
the supporter assists by scouting terrain and maintaining communication proximity.

Global State (414 dims, CTDE centralized critic):
    [Pose_sup(9), Pose_exp1(9), Pose_exp2(9),
     Vel_sup(6),  Vel_exp1(6),  Vel_exp2(6),
     Target_exp1(3), Target_exp2(3),
     Terrain_sup(121), Terrain_exp1(121), Terrain_exp2(121)]

Local Observation (125 dims per agent, decentralized actor):
    Explorers: [rel_xy_to_target(2), dist(1), angle(1), terrain(121)]
    Supporter: [rel_xy_to_nearest_explorer(2), dist(1), angle(1), terrain(121)]

Actions (2 dims per agent): [linear_velocity, angular_velocity]

Termination: episode ends when BOTH explorers reach their targets.
Truncation:  episode ends when time limit is exceeded.
"""

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
from srb.utils.math import matrix_from_quat, subtract_frame_transforms, quat_to_rot6d


###############################################################################
# Configuration
###############################################################################


@configclass
class MarlWaypointTaskCfg(GroundMarlEnvCfg):
    """Configuration for the MARL Waypoint Navigation environment.

    Three lunar rovers (1 supporter + 2 explorers) operate on procedurally
    generated Moon terrain.

    Explorers:
        Each must navigate to an individual target waypoint. Rewarded for
        distance-based progress and a sparse bonus on arrival.

    Supporter:
        No waypoint target. Rewarded for (a) scouting terrain ahead of the
        explorers, and (b) maintaining communication proximity to at least
        one explorer.

    Episode ends when BOTH explorers reach their targets (success) or the
    time limit is exceeded (truncation).

    Attributes:
        goal_reached_threshold: Distance (m) within which an explorer is
            considered to have reached its target. Default 0.5 m.
        target_spawn_radius: Maximum distance (m) from env origin at which
            explorer targets are randomly spawned. Default 5.0 m.
        target_spawn_height: Height (m) at which target markers float above
            the terrain. Default 0.5 m.
        supporter_proximity_min: Inner boundary (m) of the supporter's
            useful communication range. Closer than this is the collision
            zone and receives no proximity reward. Default 1.5 m.
        supporter_proximity_max: Outer boundary (m) of the supporter's
            useful communication range. Farther than this receives no
            proximity reward. Default 8.0 m.
        w_progress: Reward weight for explorer shaped progress. Default 1.0.
        w_goal: Reward weight for explorer sparse goal-reached bonus.
            Default 10.0.
        w_action: Penalty weight for jerky action changes (all agents).
            Default 0.1.
        w_scout: Reward weight for supporter scouting behaviour. Default 0.5.
        w_proximity: Reward weight for supporter staying in communication
            range of explorers. Default 0.3.
    """

    # -- Scene ----------------------------------------------------------------
    scene: GroundMarlSceneCfg = GroundMarlSceneCfg()

    # -- Terrain --------------------------------------------------------------
    from srb.core.asset import Scenery
    scenery: Scenery | AssetVariant = AssetVariant.PROCEDURAL

    # -- Rovers ---------------------------------------------------------------
    robots = {
        "supporter": assets.LeoRover(),
        "explorer_1": assets.LeoRover(),
        "explorer_2": assets.LeoRover(),
    }

    # -- Agent role classification --------------------------------------------
    explorer_agents: list = ["explorer_1", "explorer_2"]
    supporter_agent: str = "supporter"

    # -- Visual markers (explorers only) --------------------------------------
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
                visual_material=PreviewSurfaceCfg(
                    emissive_color=(0.2, 0.2, 0.8)
                ),
            )
        },
    )

    # -- Episode --------------------------------------------------------------
    episode_length_s: float = 60.0
    is_finite_horizon: bool = False

    # -- Goal parameters ------------------------------------------------------
    goal_reached_threshold: float = 0.5   # meters
    target_spawn_radius: float = 5.0      # meters from env origin
    target_spawn_height: float = 0.5      # meters above ground

    # -- Supporter parameters -------------------------------------------------
    supporter_proximity_min: float = 1.5  # meters (inner boundary)
    supporter_proximity_max: float = 8.0  # meters (outer boundary)

    # -- Reward weights -------------------------------------------------------
    w_progress: float = 1.0    # explorer progress toward target
    w_goal: float = 10.0       # explorer goal-reached bonus
    w_action: float = 0.1      # action rate penalty (all agents)
    w_scout: float = 0.5       # supporter scouting reward
    w_proximity: float = 0.3   # supporter proximity reward

    # -- Delays ---------------------------------------------------------------
    action_delay_steps: int = 0
    observation_delay_steps: int = 0

    # -- Space metadata -------------------------------------------------------
    # Obs per agent : 2 (rel x,y) + 1 (dist) + 1 (angle) + 121 (raycaster) = 125
    # State (global): 414  (see env_design.md for full breakdown)
    # Act per agent : 2    (linear + angular velocity)
    observation_spaces: dict = None  # type: ignore[assignment]
    action_spaces: dict = None       # type: ignore[assignment]
    state_space: int = 0             # set in __post_init__

    def __post_init__(self):
        # Terrain must be stacked so the single /World/scenery mesh is
        # visible to all RayCasters.
        self.stack = True

        # Populate space metadata BEFORE super().__post_init__() because
        # DirectMARLEnvCfg.validate() checks these are not None.
        agent_ids = list(self.robots.keys())
        self.observation_spaces = {aid: 125 for aid in agent_ids}
        self.action_spaces = {aid: 2 for aid in agent_ids}
        self.possible_agents = agent_ids

        # Global state dimensionality:
        #   3 poses   × (3 pos + 6 rot6d) = 27
        #   3 vels    × (3 lin + 3 ang)   = 18
        #   2 targets × 3 pos             =  6
        #   3 terrain × 121 rays          = 363
        #   Total                          = 414
        self.state_space = 414

        super().__post_init__()

        # Dynamically inject a downward-facing RayCaster per rover so each
        # agent can sense local terrain elevation (craters, rocks, slopes).
        for agent_id in self.robots.keys():
            setattr(
                self.scene,
                f"raycaster_{agent_id}",
                RayCasterCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/robot_{agent_id}/chassis",
                    update_period=self.agent_rate,
                    offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.5)),
                    mesh_prim_paths=["/World/scenery"],
                    pattern_cfg=patterns.GridPatternCfg(
                        resolution=0.15,
                        size=[1.5, 1.5],
                        direction=(0.0, 0.0, -1.0),
                    ),
                    max_distance=2.0,
                    debug_vis=False,
                ),
            )


###############################################################################
# Task
###############################################################################


class MarlWaypointTask(GroundMarlEnv):
    """Multi-Agent Waypoint Navigation on procedural lunar terrain.

    MDP Components
    --------------
    * ``_get_observations``: 125-dim local obs per agent (see module docstring).
    * ``_get_states``:       414-dim privileged global state for CTDE critic.
    * ``_get_rewards``:      Explorer progress/goal + supporter scout/proximity.
    * ``_get_dones``:        Terminate when both explorers reach targets.
    """

    cfg: MarlWaypointTaskCfg

    def __init__(self, cfg: MarlWaypointTaskCfg, **kwargs):
        super().__init__(cfg, **kwargs)

        # -- Visual markers (explorer targets only) --
        self._target_marker = VisualizationMarkers(self.cfg.target_marker_cfg)

        # -- Action manager for heterogeneous rover mapping --
        self.action_manager = ActionManager(self.cfg.actions, env=self)

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
            aid: self.scene[f"raycaster_{aid}"]
            for aid in self.cfg.possible_agents
        }

        # -- Goal buffers (explorers only — supporter has no waypoint) --
        self._goals = {
            aid: torch.zeros(self.num_envs, 3, device=self.device)
            for aid in self.cfg.explorer_agents
        }
        for aid in self.cfg.explorer_agents:
            self._goals[aid][:, :3] = self.scene.env_origins[:, :3]
            self._goals[aid][:, 2] = self.cfg.target_spawn_height

        # -- Per-explorer "reached" flag (for joint termination check) --
        self._explorer_reached = {
            aid: torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            for aid in self.cfg.explorer_agents
        }

    # --------------------------------------------------------------------- #
    #  Reset                                                                  #
    # --------------------------------------------------------------------- #

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)

        n = len(env_ids)

        # -- Reset explorer targets --
        for aid in self.cfg.explorer_agents:
            # Random XY within target_spawn_radius of the env origin
            offset = torch.empty(n, 3, device=self.device)
            offset[:, :2] = (
                (torch.rand(n, 2, device=self.device) * 2.0 - 1.0)
                * self.cfg.target_spawn_radius
            )
            offset[:, 2] = 0.0

            self._goals[aid][env_ids] = self.scene.env_origins[env_ids] + offset
            self._goals[aid][env_ids, 2] = self.cfg.target_spawn_height

        # -- Reset reached flags --
        for aid in self.cfg.explorer_agents:
            self._explorer_reached[aid][env_ids] = False

        # -- Reset action buffers --
        for aid in self.cfg.possible_agents:
            self._actions_dict[aid][env_ids] = 0.0
            self._prev_actions_dict[aid][env_ids] = 0.0

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

    def _apply_action(self) -> None:
        self.action_manager.apply_action()

    # --------------------------------------------------------------------- #
    #  Terrain helper (shared by observations and state)                      #
    # --------------------------------------------------------------------- #

    def _get_terrain_features(self, agent_id: str) -> torch.Tensor:
        """Relative terrain heights from the RayCaster grid.

        Returns:
            ``(num_envs, 121)`` tensor.  Negative values = crater / below
            sensor, positive = hill / above sensor.  NaN/inf (no hit)
            replaced with ``-2.0``.
        """
        rc = self._raycasters[agent_id]
        sensor_z = rc.data.pos_w[:, 2:3]          # (num_envs, 1)
        hit_z = rc.data.ray_hits_w[..., 2]        # (num_envs, 121)
        heights = hit_z - sensor_z
        return torch.nan_to_num(heights, nan=-2.0, posinf=-2.0, neginf=-2.0)

    # --------------------------------------------------------------------- #
    #  Observations  (per-agent, decentralized)                               #
    # --------------------------------------------------------------------- #

    def _get_observations(self) -> dict[str, torch.Tensor]:
        obs = {}

        # ---- Explorer observations: target-relative + terrain ----
        all_marker_pos = []
        all_marker_quat = []
        for aid in self.cfg.explorer_agents:
            robot = self._robots[aid]
            pose = robot.data.root_link_pose_w           # (N, 7)
            goal = self._goals[aid]                       # (N, 3)

            # Identity quaternion for the goal (it's just a position)
            goal_quat = torch.zeros(self.num_envs, 4, device=self.device)
            goal_quat[:, 0] = 1.0

            tf_pos, _ = subtract_frame_transforms(
                t01=pose[:, :3], q01=pose[:, 3:7],
                t02=goal, q02=goal_quat,
            )

            pos2d = tf_pos[:, :2]                                         # 2
            dist = torch.norm(pos2d, dim=-1, keepdim=True)                # 1
            angle = torch.atan2(tf_pos[:, 1], tf_pos[:, 0]).unsqueeze(-1) # 1
            terrain = self._get_terrain_features(aid)                     # 121

            obs[aid] = torch.cat([pos2d, dist, angle, terrain], dim=-1)   # 125

            # Collect markers for batch visualisation
            all_marker_pos.append(goal)
            all_marker_quat.append(goal_quat)

        # Visualise all explorer target markers in one call
        self._target_marker.visualize(
            torch.cat(all_marker_pos, dim=0),
            torch.cat(all_marker_quat, dim=0),
        )

        # ---- Supporter observation: nearest-explorer-relative + terrain ----
        sup = self._robots[self.cfg.supporter_agent]
        sup_pose = sup.data.root_link_pose_w                      # (N, 7)

        # Stack explorer world positions → (N, num_explorers, 3)
        exp_pos = torch.stack(
            [self._robots[aid].data.root_link_pose_w[:, :3]
             for aid in self.cfg.explorer_agents],
            dim=1,
        )
        dists = torch.norm(exp_pos - sup_pose[:, :3].unsqueeze(1), dim=-1)  # (N, 2)
        nearest_idx = torch.argmin(dists, dim=-1)                           # (N,)

        nearest_pos = exp_pos[
            torch.arange(self.num_envs, device=self.device), nearest_idx
        ]  # (N, 3)

        nearest_quat = torch.zeros(self.num_envs, 4, device=self.device)
        nearest_quat[:, 0] = 1.0
        tf_pos, _ = subtract_frame_transforms(
            t01=sup_pose[:, :3], q01=sup_pose[:, 3:7],
            t02=nearest_pos, q02=nearest_quat,
        )

        pos2d = tf_pos[:, :2]
        dist = torch.norm(pos2d, dim=-1, keepdim=True)
        angle = torch.atan2(tf_pos[:, 1], tf_pos[:, 0]).unsqueeze(-1)
        terrain = self._get_terrain_features(self.cfg.supporter_agent)

        obs[self.cfg.supporter_agent] = torch.cat(
            [pos2d, dist, angle, terrain], dim=-1,
        )  # 125

        return obs

    # --------------------------------------------------------------------- #
    #  Global State  (CTDE centralized critic — 414 dims)                     #
    # --------------------------------------------------------------------- #

    def _get_states(self) -> torch.Tensor:
        """Build the 414-dim privileged global state vector.

        Layout (all in env-local frame, orientations as 6D continuous rep)::

            Pose_supporter   (9)  = pos(3) + rot6d(6)
            Pose_explorer_1  (9)
            Pose_explorer_2  (9)
            Vel_supporter    (6)  = lin_vel_w(3) + ang_vel_w(3)
            Vel_explorer_1   (6)
            Vel_explorer_2   (6)
            Target_explorer_1(3)  = position relative to env_origin
            Target_explorer_2(3)
            Terrain_supporter  (121)  = RayCaster relative heights
            Terrain_explorer_1 (121)
            Terrain_explorer_2 (121)
            ─────────────────────────
            Total              414
        """
        parts = []

        # -- Poses: position (3) + 6D orientation (6) = 9 per rover --
        for aid in self.cfg.possible_agents:
            robot = self._robots[aid]
            pos_w = robot.data.root_link_pose_w[:, :3]
            quat_w = robot.data.root_link_pose_w[:, 3:7]

            pos_local = pos_w - self.scene.env_origins     # (N, 3)
            rot6d = quat_to_rot6d(quat_w)                  # (N, 6)

            parts.append(pos_local)
            parts.append(rot6d)

        # -- Velocities: linear (3) + angular (3) = 6 per rover, world frame --
        for aid in self.cfg.possible_agents:
            robot = self._robots[aid]
            parts.append(robot.data.root_lin_vel_w)        # (N, 3)
            parts.append(robot.data.root_ang_vel_w)        # (N, 3)

        # -- Explorer target positions, relative to env origin --
        for aid in self.cfg.explorer_agents:
            target_local = self._goals[aid] - self.scene.env_origins  # (N, 3)
            parts.append(target_local)

        # -- Terrain features (121 rays per rover) --
        for aid in self.cfg.possible_agents:
            parts.append(self._get_terrain_features(aid))  # (N, 121)

        return torch.cat(parts, dim=-1)  # (N, 414)

    # --------------------------------------------------------------------- #
    #  Rewards                                                                #
    # --------------------------------------------------------------------- #

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        rew = {}

        # ---- Explorer rewards ----
        for aid in self.cfg.explorer_agents:
            robot = self._robots[aid]
            pos_w = robot.data.root_link_pose_w[:, :3]
            goal = self._goals[aid]

            dist2d = torch.norm(pos_w[:, :2] - goal[:, :2], dim=-1)

            # Shaped progress reward (inverse distance)
            progress = 1.0 / (1.0 + dist2d)

            # Sparse goal-reached bonus
            reached = (dist2d < self.cfg.goal_reached_threshold).float()
            self._explorer_reached[aid] = dist2d < self.cfg.goal_reached_threshold

            # Action-rate penalty
            act_diff = torch.mean(
                (self._actions_dict[aid] - self._prev_actions_dict[aid]).square(),
                dim=-1,
            )

            rew[aid] = (
                self.cfg.w_progress * progress
                + self.cfg.w_goal * reached
                - self.cfg.w_action * act_diff
            )

        # ---- Supporter reward ----
        sup_pos = self._robots[self.cfg.supporter_agent].data.root_link_pose_w[:, :3]

        # Distances to each explorer
        exp_pos = torch.stack(
            [self._robots[aid].data.root_link_pose_w[:, :3]
             for aid in self.cfg.explorer_agents],
            dim=1,
        )  # (N, 2, 3)
        dists = torch.norm(exp_pos - sup_pos.unsqueeze(1), dim=-1)  # (N, 2)

        # Scout reward: reward the supporter for being closer to an
        # explorer's target than the explorer itself (i.e. "ahead" on the
        # path).  Averaged over both explorers and clamped to [0, 1].
        scout_reward = torch.zeros(self.num_envs, device=self.device)
        for i, aid in enumerate(self.cfg.explorer_agents):
            exp_dist_to_target = torch.norm(
                self._robots[aid].data.root_link_pose_w[:, :2] - self._goals[aid][:, :2],
                dim=-1,
            )
            sup_dist_to_target = torch.norm(
                sup_pos[:, :2] - self._goals[aid][:, :2], dim=-1,
            )
            # Positive when supporter is closer to the target than the explorer
            ahead = (exp_dist_to_target - sup_dist_to_target) / (
                self.cfg.target_spawn_radius + 1e-6
            )
            scout_reward += torch.clamp(ahead, min=0.0, max=1.0)
        scout_reward /= len(self.cfg.explorer_agents)

        # Proximity reward: Gaussian centred at the midpoint of [min, max].
        nearest_dist = torch.min(dists, dim=-1).values
        centre = (
            self.cfg.supporter_proximity_min + self.cfg.supporter_proximity_max
        ) / 2.0
        sigma = (
            self.cfg.supporter_proximity_max - self.cfg.supporter_proximity_min
        ) / 4.0
        proximity = torch.exp(-0.5 * ((nearest_dist - centre) / sigma) ** 2)

        # Action-rate penalty
        act_diff = torch.mean(
            (self._actions_dict[self.cfg.supporter_agent]
             - self._prev_actions_dict[self.cfg.supporter_agent]).square(),
            dim=-1,
        )

        rew[self.cfg.supporter_agent] = (
            self.cfg.w_scout * scout_reward
            + self.cfg.w_proximity * proximity
            - self.cfg.w_action * act_diff
        )

        return rew

    # --------------------------------------------------------------------- #
    #  Termination / Truncation                                               #
    # --------------------------------------------------------------------- #

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Terminate when BOTH explorers reach their targets.

        Time-based truncation fires when ``episode_length_buf`` exceeds
        ``max_episode_length`` (derived from ``episode_length_s``).
        """
        # Joint success: all explorers at their goals simultaneously
        both_reached = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device,
        )
        for aid in self.cfg.explorer_agents:
            both_reached &= self._explorer_reached[aid]

        # Shared termination signal (all agents end together)
        termination = {
            aid: both_reached for aid in self.cfg.possible_agents
        }

        # Time-based truncation (only if episode hasn't already terminated)
        time_out = self.episode_length_buf >= self.max_episode_length
        truncation = {
            aid: time_out & ~both_reached for aid in self.cfg.possible_agents
        }

        return termination, truncation
