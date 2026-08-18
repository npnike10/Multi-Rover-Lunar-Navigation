"""SRB-compatible multi-agent waypoint navigation task.

This task is a Dec-POMDP extension of SRB's state-based single-rover waypoint
navigation task. Every rover owns one moving planar waypoint. Rewards can be
either the rover's own original-SRB reward or the mean team reward; an optional
proximity penalty is always computed per rover before this aggregation.
Configuring one rover reproduces the original task's task signals.

Each decentralized observation contains the rover's noisy target-relative XY
position, noisy target-relative yaw encoded as sine/cosine, and noisy relative
XY positions for all other rovers.  The task deliberately has no RayCaster,
IMU, terrain observation, goal-completion, or rollover signal.
"""

import math
from typing import Literal, Sequence

import gymnasium
import torch

from srb import assets
from srb.core.asset import AssetVariant, Scenery
from srb.core.env.mobile.ground.marl_env import (
    GroundMarlEnv,
    GroundMarlEnvCfg,
    GroundMarlEventCfg,
    GroundMarlSceneCfg,
)
from srb.core.manager import ActionManager, EventTermCfg
from srb.core.marker import VisualizationMarkers, VisualizationMarkersCfg
from srb.core.mdp import offset_pose_natural
from srb.core.sim import PreviewSurfaceCfg
from srb.core.sim.spawners.shapes.extras.cfg import PinnedArrowCfg
from srb.utils.cfg import configclass
from srb.utils.math import matrix_from_quat, subtract_frame_transforms

###############################################################################
# Scene and event configuration
###############################################################################


@configclass
class MarlWaypointSceneCfg(GroundMarlSceneCfg):
    """Scene configuration for the procedural lunar waypoint task."""


@configclass
class MarlWaypointEventCfg(GroundMarlEventCfg):
    """Holds reset events and one natural-motion event per rover target."""


###############################################################################
# Task configuration
###############################################################################


@configclass
class MarlWaypointTaskCfg(GroundMarlEnvCfg):
    """Configuration for SRB-compatible multi-agent waypoint navigation.

    The default configuration has three identical Leo rovers.  Set
    ``num_rovers=1`` to create the one-rover SRB-replication configuration;
    rover identities are then generated as ``rover_1`` through
    ``rover_<num_rovers>``.

    The procedural scenery is intentionally left as ``AssetVariant.PROCEDURAL``.
    With the Moon domain and 32 m spacing, SRB resolves it to its default
    ``MoonSurface`` configuration rather than the former MARL-specific terrain
    override.
    """

    # -- Scene and world -----------------------------------------------------
    scene: MarlWaypointSceneCfg = MarlWaypointSceneCfg(env_spacing=32.0)
    events: MarlWaypointEventCfg = MarlWaypointEventCfg()
    stack: bool = True
    scenery: Scenery | AssetVariant = AssetVariant.PROCEDURAL

    # -- Homogeneous rover team ---------------------------------------------
    # LeoRover maps normalized [linear, angular] actions to 0.4 m/s and
    # 60 deg/s respectively.  Use this robot for replication of the requested
    # action bounds and scaling.
    num_rovers: int = 3
    robots = {
        "rover_1": assets.LeoRover(),
        "rover_2": assets.LeoRover(),
        "rover_3": assets.LeoRover(),
    }

    # -- Time ---------------------------------------------------------------
    episode_length_s: float = 60.0
    is_finite_horizon: bool = False

    # -- Action and observation latency ------------------------------------
    # Match the state-based SRB waypoint task.  Delays are measured in agent
    # steps (the default agent rate is 25 Hz).  A range is sampled separately
    # for every rover in every parallel environment at reset, then may drift
    # by one step at the configured interval.  Set both values to ``0`` for
    # the previous zero-latency MARL behavior.
    action_delay_steps: int | tuple[int, int] = (0, 3)
    action_delay_on_step_change_freq: float = 1.0
    action_delay_on_step_change_prob: float = 0.01
    observation_delay_steps: int | tuple[int, int] = (0, 1)
    observation_delay_on_step_change_freq: float = 1.0
    observation_delay_on_step_change_prob: float = 0.01

    # -- Moving waypoint event (matches the single-rover SRB task) ----------
    target_event_interval_s: float = 0.05
    target_pos_step_range: tuple[float, float] = (0.005, 0.01)
    target_pos_smoothness: float = 0.99
    target_pos_step_smoothness: float = 0.8
    target_orient_smoothness: float = 0.8
    target_pos_range_ratio: float = 0.9
    multi_rover_target_motion_half_range: float = 1
    """Half-width (m) of each multi-rover target's persistent XY region.

    Each target region is centered on the corresponding rover's reset-window
    center. This prevents the three targets from initially coinciding at the
    environment origin while retaining the original broad SRB target region
    when ``num_rovers=1``.
    """

    # -- Observation noise (matches the single-rover SRB task) -------------
    # Noise is per coordinate.  Each directed observer-to-entity XY relation
    # has an independent episodic offset and independent per-observation noise.
    episodic_xy_noise_std: float = 0.01
    per_step_xy_noise_std: float = 0.0025
    episodic_yaw_noise_std: float = math.radians(2.5)
    per_step_yaw_noise_std: float = math.radians(0.5)

    # -- Multi-agent reward and safety terms --------------------------------
    reward_mode: Literal["individual", "team"] = "individual"
    """``individual`` returns each rover's own reward; ``team`` returns their mean."""

    w_proximity: float = 1.0
    # Leo's longitudinal wheelbase is 0.3587 m and serves as one rover length.
    proximity_safe_distance: float = 0.3587

    # -- Visuals -------------------------------------------------------------
    target_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/marl_waypoint_targets",
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

    # -- Space metadata ------------------------------------------------------
    # Local observation: target XY (2), target yaw sin/cos (2), and XY for
    # every other rover (2 each): 4 + 2 * (N - 1).
    # Centralized state: rover XY/yaw sin-cos (4 each) followed by target
    # XY/yaw sin-cos (4 each): 8 * N.
    observation_spaces: dict = None  # type: ignore[assignment]
    action_spaces: dict = None  # type: ignore[assignment]
    state_space: gymnasium.Space = gymnasium.spaces.Box(
        low=-math.inf, high=math.inf, shape=(1,)
    )

    @property
    def local_observation_dim(self) -> int:
        """Dimension of one rover's decentralized observation."""
        return 4 + 2 * (len(self.robots) - 1)

    @property
    def global_state_dim(self) -> int:
        """Dimension of the compact, unnoised centralized-critic state."""
        return 8 * len(self.robots)

    def __post_init__(self):
        if self.num_rovers < 1:
            raise ValueError(
                f"num_rovers must be at least 1. Received: {self.num_rovers}."
            )
        if self.reward_mode not in {"individual", "team"}:
            raise ValueError(
                "reward_mode must be either 'individual' or 'team'. "
                f"Received: {self.reward_mode!r}."
            )
        if self.proximity_safe_distance <= 0.0:
            raise ValueError("proximity_safe_distance must be positive.")
        if self.multi_rover_target_motion_half_range <= 0.0:
            raise ValueError("multi_rover_target_motion_half_range must be positive.")
        self._validate_delay_config(
            "action_delay_steps",
            self.action_delay_steps,
            self.action_delay_on_step_change_freq,
            self.action_delay_on_step_change_prob,
        )
        self._validate_delay_config(
            "observation_delay_steps",
            self.observation_delay_steps,
            self.observation_delay_on_step_change_freq,
            self.observation_delay_on_step_change_prob,
        )

        # All agents in this task are homogeneous Leo waypoint navigators.
        # Rebuild the mapping from the public ``num_rovers`` override before
        # the base configuration creates scene assets and action terms.
        self.robots = {
            f"rover_{index + 1}": assets.LeoRover() for index in range(self.num_rovers)
        }
        agent_ids = list(self.robots.keys())

        # The target events must exist before the base configuration builds its
        # event manager.  ``spacing`` may be supplied explicitly; otherwise it
        # is inherited from the 32 m scene configuration.
        single_rover_target_bound = (
            0.5
            * self.target_pos_range_ratio
            * (self.spacing if self.spacing is not None else self.scene.env_spacing)
        )
        for index, agent_id in enumerate(agent_ids):
            target_center_x, target_center_y = self._target_center(
                index, len(agent_ids)
            )
            target_bound = (
                single_rover_target_bound
                if len(agent_ids) == 1
                else self.multi_rover_target_motion_half_range
            )
            setattr(
                self.events,
                f"target_{agent_id}_pose_evolution",
                EventTermCfg(
                    func=offset_pose_natural,
                    mode="interval",
                    interval_range_s=(
                        self.target_event_interval_s,
                        self.target_event_interval_s,
                    ),
                    is_global_time=True,
                    params={
                        "env_attr_name": f"_target_goal_{agent_id}",
                        "pos_axes": ("x", "y"),
                        "pos_step_range": self.target_pos_step_range,
                        "pos_smoothness": self.target_pos_smoothness,
                        "pos_step_smoothness": self.target_pos_step_smoothness,
                        "pos_bounds": {
                            "x": (
                                target_center_x - target_bound,
                                target_center_x + target_bound,
                            ),
                            "y": (
                                target_center_y - target_bound,
                                target_center_y + target_bound,
                            ),
                        },
                        "orient_yaw_only": True,
                        "orient_smoothness": self.target_orient_smoothness,
                    },
                ),
            )

        self.possible_agents = agent_ids
        self.observation_spaces = {
            agent_id: gymnasium.spaces.Box(
                low=-math.inf,
                high=math.inf,
                shape=(self.local_observation_dim,),
            )
            for agent_id in agent_ids
        }
        self.action_spaces = {
            agent_id: gymnasium.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(2,),
            )
            for agent_id in agent_ids
        }
        self.state_space = gymnasium.spaces.Box(
            low=-math.inf,
            high=math.inf,
            shape=(self.global_state_dim,),
        )

        super().__post_init__()

        # Reproduce the original reset ranges for one rover.  For multi-rover
        # runs, only the XY centers are adapted to keep the rovers separated;
        # yaw, height, and velocity distributions remain those of SRB's task.
        for index, agent_id in enumerate(agent_ids):
            event_cfg = getattr(self.events, f"randomize_{agent_id}_state")
            x_center, y_center, xy_half_range = self._spawn_window(
                index, len(agent_ids)
            )
            event_cfg.params["pose_range"]["x"] = (
                x_center - xy_half_range,
                x_center + xy_half_range,
            )
            event_cfg.params["pose_range"]["y"] = (
                y_center - xy_half_range,
                y_center + xy_half_range,
            )
            event_cfg.params["pose_range"]["z"] = (0.4, 0.6)
            event_cfg.params["pose_range"]["yaw"] = (-torch.pi, torch.pi)
            event_cfg.params["velocity_range"].update(
                {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (0.0, 0.5),
                    "roll": (-math.radians(5.0), math.radians(5.0)),
                    "pitch": (-math.radians(5.0), math.radians(5.0)),
                    "yaw": (-math.radians(15.0), math.radians(15.0)),
                }
            )

    @staticmethod
    def _validate_delay_config(
        name: str,
        delay: int | tuple[int, int],
        change_frequency: float,
        change_probability: float,
    ) -> None:
        """Validate an SRB-style integer delay configuration."""
        lower, upper = (delay, delay) if isinstance(delay, int) else delay
        if lower < 0 or upper < lower:
            raise ValueError(
                f"{name} must be a non-negative integer or an ordered "
                f"(min, max) pair. Received: {delay!r}."
            )
        if change_frequency <= 0.0:
            raise ValueError(f"{name} change frequency must be positive.")
        if not 0.0 <= change_probability <= 1.0:
            raise ValueError(
                f"{name} change probability must be in [0, 1]. Received: "
                f"{change_probability}."
            )

    @staticmethod
    def _spawn_window(index: int, count: int) -> tuple[float, float, float]:
        """Return a separated SRB-style reset window for one rover.

        A single rover uses the original ``[-0.5, 0.5]`` XY ranges.  The
        three-rover default uses the former task's proven non-overlapping
        centers with a 0.2 m jitter.  Other team sizes use evenly spaced points
        on a 1 m radius circle with the same jitter.
        """
        if count == 1:
            return 0.0, 0.0, 0.5
        if count == 2:
            return (-0.5, 0.0, 0.2) if index == 0 else (0.5, 0.0, 0.2)
        if count == 3:
            return ((-0.5, 0.0, 0.2), (0.5, -0.5, 0.2), (0.5, 0.5, 0.2))[index]

        angle = 2.0 * math.pi * index / count
        return math.cos(angle), math.sin(angle), 0.2

    def _target_center(self, index: int, count: int) -> tuple[float, float]:
        """Return the persistent center of one rover's target-motion region.

        Multi-rover target centers coincide with the separated rover reset
        centers, keeping each target near the rover's own local navigation
        region. The one-rover center remains the original environment origin.
        """
        x_center, y_center, _ = self._spawn_window(index, count)
        return x_center, y_center


###############################################################################
# Task
###############################################################################


class MarlWaypointTask(GroundMarlEnv):
    """Configurable-reward Dec-POMDP extension of SRB waypoint navigation.

    No physical sensor object is created for this task.  Target and rover
    relative positions are task-state measurements; configurable Gaussian noise
    is applied directly when decentralized observations are constructed.
    """

    cfg: MarlWaypointTaskCfg

    def __init__(self, cfg: MarlWaypointTaskCfg, **kwargs):
        super().__init__(cfg, **kwargs)

        self.action_manager = ActionManager(self.cfg.actions, env=self)
        self._actions = {
            agent_id: torch.zeros(self.num_envs, 2, device=self.device)
            for agent_id in self.cfg.possible_agents
        }
        self._previous_actions = {
            agent_id: torch.zeros(self.num_envs, 2, device=self.device)
            for agent_id in self.cfg.possible_agents
        }

        # Goals are virtual 7D poses.  The dynamic events mutate each named
        # tensor in place; the dictionary provides convenient agent lookup.
        self._goal_attr_names = {
            agent_id: f"_target_goal_{agent_id}"
            for agent_id in self.cfg.possible_agents
        }
        self._goals: dict[str, torch.Tensor] = {}
        for agent_id, attr_name in self._goal_attr_names.items():
            goal = torch.zeros(self.num_envs, 7, device=self.device)
            goal[:, :3] = self.scene.env_origins
            goal[:, 3] = 1.0
            setattr(self, attr_name, goal)
            self._goals[agent_id] = goal

        # XY offsets are independent for each ordered observer/entity pair.
        # ``target`` denotes the observer's own target; every other key denotes
        # a physical rover visible to that observer.
        self._episodic_xy_noise = {
            (observer, entity): torch.zeros(self.num_envs, 2, device=self.device)
            for observer in self.cfg.possible_agents
            for entity in [
                "target",
                *[other for other in self.cfg.possible_agents if other != observer],
            ]
        }
        self._episodic_yaw_noise = {
            agent_id: torch.zeros(self.num_envs, device=self.device)
            for agent_id in self.cfg.possible_agents
        }
        self._other_agents = {
            agent_id: [other for other in self.cfg.possible_agents if other != agent_id]
            for agent_id in self.cfg.possible_agents
        }
        self._target_marker = VisualizationMarkers(self.cfg.target_marker_cfg)

        self._proximity_safe_distance = self.cfg.proximity_safe_distance

        # DirectMARLEnv does not provide DirectEnv's delay machinery.  Keep a
        # separate delay process and history per rover, matching what each
        # rover would experience when running the original single-agent task.
        self._init_delay_buffers()

    # ------------------------------------------------------------------ #
    # Action and observation delay
    # ------------------------------------------------------------------ #

    @staticmethod
    def _delay_bounds(delay: int | tuple[int, int]) -> tuple[int, int]:
        """Normalize a fixed delay or inclusive delay range."""
        return (delay, delay) if isinstance(delay, int) else delay

    def _init_delay_buffers(self) -> None:
        """Allocate zero-filled, per-rover delay buffers.

        A history has ``maximum_delay + 1`` entries so that delay zero reads
        the command/measurement just written while the largest delay reads a
        genuinely older entry.  The additional slot also preserves the
        intended zero-action/zero-observation warm-up after reset.
        """
        self._min_action_delay_steps, self._max_action_delay_steps = (
            self._delay_bounds(self.cfg.action_delay_steps)
        )
        self._min_observation_delay_steps, self._max_observation_delay_steps = (
            self._delay_bounds(self.cfg.observation_delay_steps)
        )

        self._action_delay_steps = {
            agent_id: torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            for agent_id in self.cfg.possible_agents
        }
        self._observation_delay_steps = {
            agent_id: torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            for agent_id in self.cfg.possible_agents
        }

        self._action_history_buffer: dict[str, torch.Tensor] | None = None
        self._action_history_buffer_ptr = 0
        if self._max_action_delay_steps > 0:
            self._action_history_buffer = {
                agent_id: torch.zeros(
                    self._max_action_delay_steps + 1,
                    self.num_envs,
                    2,
                    device=self.device,
                )
                for agent_id in self.cfg.possible_agents
            }

        self._observation_history_buffer: dict[str, torch.Tensor] | None = None
        self._observation_history_buffer_ptr = 0
        if self._max_observation_delay_steps > 0:
            self._observation_history_buffer = {
                agent_id: torch.zeros(
                    self._max_observation_delay_steps + 1,
                    self.num_envs,
                    self.cfg.local_observation_dim,
                    device=self.device,
                )
                for agent_id in self.cfg.possible_agents
            }

    def _reset_delay_buffers(self, env_ids: Sequence[int]) -> None:
        """Sample fresh delays and clear history for reset environments."""
        num_reset_envs = len(env_ids)
        for agent_id in self.cfg.possible_agents:
            self._action_delay_steps[agent_id][env_ids] = torch.randint(
                self._min_action_delay_steps,
                self._max_action_delay_steps + 1,
                (num_reset_envs,),
                device=self.device,
            )
            self._observation_delay_steps[agent_id][env_ids] = torch.randint(
                self._min_observation_delay_steps,
                self._max_observation_delay_steps + 1,
                (num_reset_envs,),
                device=self.device,
            )
            if self._action_history_buffer is not None:
                self._action_history_buffer[agent_id][:, env_ids] = 0.0
            if self._observation_history_buffer is not None:
                self._observation_history_buffer[agent_id][:, env_ids] = 0.0

    def _advance_delay_process(
        self,
        delays: dict[str, torch.Tensor],
        minimum: int,
        maximum: int,
        change_frequency: float,
        change_probability: float,
    ) -> None:
        """Occasionally move each rover's delay by one bounded agent step."""
        if (
            change_probability <= 0.0
            or minimum == maximum
            or (self.sim.current_time % change_frequency) >= self.cfg.agent_rate
        ):
            return

        for agent_delays in delays.values():
            random_values = torch.rand(self.num_envs, device=self.device)
            decrease = (agent_delays > minimum) & (
                random_values < change_probability
            )
            increase = (agent_delays < maximum) & (
                random_values > (1.0 - change_probability)
            )
            agent_delays[decrease] -= 1
            agent_delays[increase] += 1

    def _apply_observation_delay(
        self, observations: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Store current observations and return each rover's delayed view."""
        if self._observation_history_buffer is None:
            return observations

        history_length = self._max_observation_delay_steps + 1
        env_indices = torch.arange(self.num_envs, device=self.device)
        delayed_observations = {}
        for agent_id, observation in observations.items():
            history = self._observation_history_buffer[agent_id]
            history[self._observation_history_buffer_ptr] = observation
            read_indices = (
                self._observation_history_buffer_ptr
                - self._observation_delay_steps[agent_id]
            ) % history_length
            delayed_observations[agent_id] = history[read_indices, env_indices]

        self._observation_history_buffer_ptr = (
            self._observation_history_buffer_ptr + 1
        ) % history_length
        self._advance_delay_process(
            self._observation_delay_steps,
            self._min_observation_delay_steps,
            self._max_observation_delay_steps,
            self.cfg.observation_delay_on_step_change_freq,
            self.cfg.observation_delay_on_step_change_prob,
        )
        return delayed_observations

    # ------------------------------------------------------------------ #
    # Reset and actions
    # ------------------------------------------------------------------ #

    def _reset_idx(self, env_ids: Sequence[int]):
        """Reset rover-local buffers, targets, and episodic observation noise."""
        super()._reset_idx(env_ids)

        self._reset_delay_buffers(env_ids)

        origins = self.scene.env_origins[env_ids]
        for index, agent_id in enumerate(self.cfg.possible_agents):
            goal = self._goals[agent_id]
            goal[env_ids, :3] = origins
            target_center_x, target_center_y = self.cfg._target_center(
                index, len(self.cfg.possible_agents)
            )
            goal[env_ids, 0] += target_center_x
            goal[env_ids, 1] += target_center_y
            goal[env_ids, 3:7] = 0.0
            goal[env_ids, 3] = 1.0

            self._actions[agent_id][env_ids] = 0.0
            self._previous_actions[agent_id][env_ids] = 0.0
            self._episodic_yaw_noise[agent_id][env_ids] = (
                torch.randn(len(env_ids), device=self.device)
                * self.cfg.episodic_yaw_noise_std
            )

        for noise in self._episodic_xy_noise.values():
            noise[env_ids] = (
                torch.randn(len(env_ids), 2, device=self.device)
                * self.cfg.episodic_xy_noise_std
            )

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        """Queue normalized commands and send the per-rover delayed commands."""
        flat_actions = []
        env_indices = torch.arange(self.num_envs, device=self.device)
        history_length = self._max_action_delay_steps + 1
        for agent_id in self.cfg.possible_agents:
            action = actions[agent_id].to(self.device).clamp(-1.0, 1.0)

            if self._action_history_buffer is None:
                applied_action = action
            else:
                history = self._action_history_buffer[agent_id]
                history[self._action_history_buffer_ptr] = action
                read_indices = (
                    self._action_history_buffer_ptr
                    - self._action_delay_steps[agent_id]
                ) % history_length
                applied_action = history[read_indices, env_indices]

            # Rewards use the change in commands that actually reach the
            # rover, consistent with the delayed action-manager path in the
            # original waypoint task.
            self._previous_actions[agent_id][:] = self._actions[agent_id]
            self._actions[agent_id][:] = applied_action
            flat_actions.append(applied_action)

        if self._action_history_buffer is not None:
            self._action_history_buffer_ptr = (
                self._action_history_buffer_ptr + 1
            ) % history_length
            self._advance_delay_process(
                self._action_delay_steps,
                self._min_action_delay_steps,
                self._max_action_delay_steps,
                self.cfg.action_delay_on_step_change_freq,
                self.cfg.action_delay_on_step_change_prob,
            )

        self.action_manager.process_action(torch.cat(flat_actions, dim=-1))

    def _apply_action(self) -> None:
        """Apply the Leo Rover linear and angular velocity commands."""
        self.action_manager.apply_action()

    # ------------------------------------------------------------------ #
    # Observations and centralized state
    # ------------------------------------------------------------------ #

    @staticmethod
    def _yaw_from_quat(quat: torch.Tensor) -> torch.Tensor:
        """Return yaw from a wxyz quaternion tensor."""
        rotmat = matrix_from_quat(quat)
        return torch.atan2(rotmat[..., 1, 0], rotmat[..., 0, 0])

    @staticmethod
    def _yaw_sin_cos(yaw: torch.Tensor) -> torch.Tensor:
        """Encode a heading without the discontinuity at ±pi."""
        return torch.stack((torch.sin(yaw), torch.cos(yaw)), dim=-1)

    def _noisy_xy(
        self, observer: str, entity: str, relative_xy: torch.Tensor
    ) -> torch.Tensor:
        """Add the configured episodic offset and fresh per-step XY noise."""
        return (
            relative_xy
            + self._episodic_xy_noise[(observer, entity)]
            + torch.randn_like(relative_xy) * self.cfg.per_step_xy_noise_std
        )

    def _visualize_targets(self) -> None:
        """Display one virtual target marker per rover when rendering is active."""
        target_poses = torch.cat(
            [self._goals[agent_id] for agent_id in self.cfg.possible_agents], dim=0
        )
        self._target_marker.visualize(target_poses[:, :3], target_poses[:, 3:7])

    def _get_observations(self) -> dict[str, torch.Tensor]:
        """Return each rover's decentralized noisy planar observation.

        The target-relative XY and yaw use the original SRB noise scales.  The
        multi-agent extension applies independent XY noise to each directed
        observer-to-other-rover measurement as well.
        """
        self._visualize_targets()
        observations = {}
        rover_positions = {
            agent_id: self._robots[agent_id].data.root_link_pose_w[:, :3]
            for agent_id in self.cfg.possible_agents
        }

        for agent_id in self.cfg.possible_agents:
            pose = self._robots[agent_id].data.root_link_pose_w
            rover_pos, rover_quat = pose[:, :3], pose[:, 3:7]
            goal = self._goals[agent_id]
            target_pos_b, target_quat_b = subtract_frame_transforms(
                t01=rover_pos,
                q01=rover_quat,
                t02=goal[:, :3],
                q02=goal[:, 3:7],
            )
            noisy_target_xy = self._noisy_xy(agent_id, "target", target_pos_b[:, :2])
            target_yaw_b = self._yaw_from_quat(target_quat_b)
            noisy_target_yaw = (
                target_yaw_b
                + self._episodic_yaw_noise[agent_id]
                + torch.randn_like(target_yaw_b) * self.cfg.per_step_yaw_noise_std
            )
            parts = [noisy_target_xy, self._yaw_sin_cos(noisy_target_yaw)]

            for other_id in self._other_agents[agent_id]:
                other_pos_b, _ = subtract_frame_transforms(
                    t01=rover_pos,
                    q01=rover_quat,
                    t02=rover_positions[other_id],
                )
                parts.append(self._noisy_xy(agent_id, other_id, other_pos_b[:, :2]))

            observations[agent_id] = torch.cat(parts, dim=-1)

        return self._apply_observation_delay(observations)

    def _get_states(self) -> torch.Tensor:
        """Return the compact unnoised planar CTDE state.

        The state contains rover XY and heading sin/cos for every rover,
        followed by the corresponding target XY and heading sin/cos.  It is a
        privileged task state for a centralized critic, not the full physical
        simulator state (which also includes velocities and target dynamics).
        """
        parts = []
        for agent_id in self.cfg.possible_agents:
            pose = self._robots[agent_id].data.root_link_pose_w
            parts.extend(
                (
                    pose[:, :2] - self.scene.env_origins[:, :2],
                    self._yaw_sin_cos(self._yaw_from_quat(pose[:, 3:7])),
                )
            )
        for agent_id in self.cfg.possible_agents:
            goal = self._goals[agent_id]
            parts.extend(
                (
                    goal[:, :2] - self.scene.env_origins[:, :2],
                    self._yaw_sin_cos(self._yaw_from_quat(goal[:, 3:7])),
                )
            )
        return torch.cat(parts, dim=-1)

    # ------------------------------------------------------------------ #
    # Per-rover reward
    # ------------------------------------------------------------------ #

    def _srb_reward(self, agent_id: str) -> torch.Tensor:
        """Return one rover's unmodified six-term SRB reward.

        This exactly preserves the constants and equations in the original
        state-based waypoint task.
        """
        pose = self._robots[agent_id].data.root_link_pose_w
        target_pos_b, target_quat_b = subtract_frame_transforms(
            t01=pose[:, :3],
            q01=pose[:, 3:7],
            t02=self._goals[agent_id][:, :3],
            q02=self._goals[agent_id][:, 3:7],
        )
        distance = torch.norm(target_pos_b[:, :2], dim=-1)
        angle_to_target = torch.atan2(target_pos_b[:, 1], target_pos_b[:, 0])
        target_yaw_b = self._yaw_from_quat(target_quat_b)
        action_rate = (
            (self._actions[agent_id] - self._previous_actions[agent_id])
            .square()
            .mean(dim=-1)
        )

        penalty_action_rate = -0.5 * action_rate
        penalty_position_tracking = -torch.square(distance)
        reward_point_towards_target = 1.0 - torch.tanh(
            torch.abs(angle_to_target) / 0.7854
        )
        position_precision = 1.0 - torch.tanh(distance / 0.05)
        reward_position_tracking_precision = 4.0 * position_precision
        orientation_precision = position_precision * (
            1.0 - torch.tanh(torch.abs(target_yaw_b) / 0.2618)
        )
        reward_orientation_tracking = 8.0 * orientation_precision
        reward_action_rate_at_target = (
            32.0 * orientation_precision * (1.0 - torch.tanh(action_rate / 0.1))
        )

        reward = (
            penalty_action_rate
            + penalty_position_tracking
            + reward_point_towards_target
            + reward_position_tracking_precision
            + reward_orientation_tracking
            + reward_action_rate_at_target
        )
        return reward

    def _individual_proximity_penalties(self) -> dict[str, torch.Tensor]:
        """Return each rover's mean close-neighbor XY penalty.

        A pair contributes the same normalized penalty to both involved
        rovers. Averaging each rover's pair contributions keeps the scale
        independent of the team size. ``proximity_safe_distance`` defaults to
        one Leo rover length (0.3587 m).
        """
        penalties = {agent_id: [] for agent_id in self.cfg.possible_agents}
        if len(self.cfg.possible_agents) < 2:
            return {
                agent_id: torch.zeros(self.num_envs, device=self.device)
                for agent_id in self.cfg.possible_agents
            }

        for first, first_id in enumerate(self.cfg.possible_agents):
            first_position = self._robots[first_id].data.root_link_pose_w[:, :2]
            for second_id in self.cfg.possible_agents[first + 1 :]:
                second_position = self._robots[second_id].data.root_link_pose_w[:, :2]
                distance = torch.norm(first_position - second_position, dim=-1)
                penalty = torch.clamp(
                    1.0 - distance / self._proximity_safe_distance, min=0.0
                )
                penalties[first_id].append(penalty)
                penalties[second_id].append(penalty)
        return {
            agent_id: torch.stack(agent_penalties, dim=-1).mean(dim=-1)
            for agent_id, agent_penalties in penalties.items()
        }

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        """Return individual rewards or their configurable team mean.

        Every rover first receives its own unmodified SRB reward minus its own
        mean close-neighbor proximity penalty. ``reward_mode='team'`` then
        gives all rovers the mean of these already-penalized individual terms.
        """
        proximity_penalties = self._individual_proximity_penalties()
        individual_rewards = {
            agent_id: self._srb_reward(agent_id)
            - self.cfg.w_proximity * proximity_penalties[agent_id]
            for agent_id in self.cfg.possible_agents
        }
        if self.cfg.reward_mode == "individual":
            return individual_rewards

        team_reward = torch.stack(
            [individual_rewards[agent_id] for agent_id in self.cfg.possible_agents],
            dim=-1,
        ).mean(dim=-1)
        return {agent_id: team_reward for agent_id in self.cfg.possible_agents}

    # ------------------------------------------------------------------ #
    # Termination
    # ------------------------------------------------------------------ #

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Terminate only through the SRB task's configured time limit."""
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        time_out = self.episode_length_buf >= self.max_episode_length
        return (
            {agent_id: terminated for agent_id in self.cfg.possible_agents},
            {agent_id: time_out for agent_id in self.cfg.possible_agents},
        )
