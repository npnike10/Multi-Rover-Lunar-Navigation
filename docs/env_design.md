# MARL Waypoint Navigation — Environment Design Document (v3 — Final)

## Overview

A **Dec-POMDP** multi-agent reinforcement learning environment with 3 lunar
rovers (1 supporter + 2 explorers) on procedurally generated Moon terrain.
All agents receive the **exact same reward** at each timestep. Explorers
navigate to individual waypoints; the supporter has no target and must learn
how to assist the team purely through the shared reward signal.

---

## Agent Roles

### Explorers (`explorer_1`, `explorer_2`)
- **Goal:** Navigate to their individual target waypoint.
- **Reward contribution:** Progress toward target + goal-reached bonus.

### Supporter (`supporter`)
- **Goal:** No waypoint target. The shared reward incentivizes the
  supporter to learn whatever behavior helps the team succeed.
- **Reward contribution:** None specific — benefits from the team reward.

---

## Dec-POMDP Shared Reward

All agents receive the **exact same scalar** at each timestep:

```
R = w_progress  * mean(explorer_progress)
  + w_goal      * (reached_exp1 + reached_exp2) / 2
  - w_proximity * mean(proximity_penalty over all 3 rover pairs)
  - w_action    * mean(action_rate_penalty over all 3 agents)
```

| Term | Formula | Range |
|:---|:---|:---|
| `progress_i` | `1 / (1 + dist_explorer_i_to_target)` | [0, 1] |
| `reached_i` | `1.0 if dist < goal_reached_threshold` | {0, 1} |
| `proximity(a,b)` | `max(0, 1 - dist(a,b) / safe_distance)` | [0, 1] |
| `action_rate(a)` | `mean((a_t - a_{t-1})^2)` | [0, ~4] |

| Weight | Default | Rationale |
|:---|:---|:---|
| `w_progress` | `1.0` | Dense shaped signal |
| `w_goal` | `5.0` | Per-explorer sparse milestone bonus |
| `w_proximity` | `0.5` | Discourages dangerous proximity |
| `w_action` | `0.1` | Light smoothness regularization |

---

## Observation Space (139 dims per agent)

All agents share the same 139-dimensional observation. The first 2 dims
differ semantically: explorers see relative XY to their target; the
supporter receives zeros (no target).

| Component | Dims | Source | Notes |
|:---|:---|:---|:---|
| Task target XY (explorers) / zeros (supporter) | 2 | `subtract_frame_transforms` | Body frame |
| Other rover 1 relative XY | 2 | `subtract_frame_transforms` | Body frame |
| Other rover 2 relative XY | 2 | `subtract_frame_transforms` | Body frame |
| Body-frame linear velocity | 3 | `root_lin_vel_b` | Deployment: sensor fusion (wheel odom + IMU) |
| IMU linear acceleration | 3 | `ImuData.lin_acc_b` | Body frame; captures terrain forces |
| IMU angular velocity | 3 | `ImuData.ang_vel_b` | Body frame; gyroscope reading |
| Projected gravity | 3 | `ImuData.projected_gravity_b` | (0,0,-1) when upright; detects slope and tilt |
| RayCaster terrain heights | 121 | `RayCasterData` | 11x11 grid, relative to sensor Z |
| **Total** | **139** | | |

### Design Rationale
- **Relative XY only** (no distance + heading): XY is sufficient; distance
  and heading are derivable and add no information.
- **No relative velocity of other rovers**: Preserves partial observability
  (Dec-POMDP). A recurrent policy can infer velocity from consecutive
  position observations.
- **Body-frame velocity**: In deployment, estimated from sensor fusion.
  In simulation, ground truth from PhysX (SRB convention).
- **IMU acceleration**: Captures instantaneous terrain forces; useful for
  low-level control even though velocity is also provided.
- **Projected gravity**: NOT the gravitational constant (fixed on Moon).
  It's the *direction* of gravity in body frame — shifts on slopes, crucial
  for rollover awareness and slope-aware control.

---

## Global State (414 dims, CTDE Centralized Critic)

All in the **env-local frame** (world-aligned, origin at `env_origin`).
Uses ground-truth data not available to the actor.

| Component | Dims | Notes |
|:---|:---|:---|
| Pose_supporter | 9 | position(3) + 6D rotation(6) |
| Pose_explorer_1 | 9 | |
| Pose_explorer_2 | 9 | |
| Velocity_supporter | 6 | lin_vel_w(3) + ang_vel_w(3), world frame |
| Velocity_explorer_1 | 6 | |
| Velocity_explorer_2 | 6 | |
| Target_explorer_1 | 3 | Position relative to env_origin |
| Target_explorer_2 | 3 | |
| Terrain_supporter | 121 | RayCaster heights |
| Terrain_explorer_1 | 121 | |
| Terrain_explorer_2 | 121 | |
| **Total** | **414** | |

### Key Differences: State vs Observation
| Aspect | Observation (Actor) | State (Critic) |
|:---|:---|:---|
| Velocity frame | Body frame | World frame |
| Velocity source | `root_lin_vel_b` (local) | `root_lin_vel_w` (global, comparable) |
| Orientation | Implicit (via projected_gravity) | Explicit 6D rotation |
| Other rovers | Relative XY only | Full absolute poses |
| Targets | Own target only | All targets |

---

## Termination Conditions

| Condition | Type | Details |
|:---|:---|:---|
| Both explorers reach targets | Termination (success) | `dist < goal_reached_threshold` for both simultaneously |
| Rollover | Termination (failure) | Tilt > 75 degrees for 5 consecutive steps (0.2s debounce) |
| Episode timeout | Truncation | `episode_length_s` exceeded |

### Rollover Detection
Uses `projected_gravity_b` from IMU (mathematically exact in sim):
```python
tilt = acos(clamp(-projected_gravity_b[:, 2], -1, 1))
tilted = tilt > 1.31  # ~75 degrees
# Must persist for 5 consecutive steps to trigger
```

---

## Sensors (per rover)

| Sensor | Config | Data Used |
|:---|:---|:---|
| **IMU** | `ImuCfg` on chassis, `gravity_bias=(0,0,0)` | `lin_acc_b(3)`, `ang_vel_b(3)`, `projected_gravity_b(3)` |
| **RayCaster** | 11x11 grid, 0.15m spacing, 2.0m max, downward | Relative terrain heights (121) |

---

## Configurable Parameters

| Parameter | Default | Description |
|:---|:---|:---|
| `episode_length_s` | `60.0` | Max episode duration (seconds) |
| `goal_reached_threshold` | `0.5` | Distance (m) for explorer goal completion |
| `target_spawn_radius` | `5.0` | Max distance (m) for random target placement |
| `target_spawn_height` | `0.5` | Target marker height (m) |
| `safe_distance` | `None` (auto) | Min inter-rover distance (m). Auto = longest rover length |
| `rollover_threshold_rad` | `1.31` | ~75 deg tilt for rollover termination |
| `rollover_debounce_steps` | `5` | Consecutive steps before rollover triggers (0.2s) |
| `w_progress` | `1.0` | Explorer progress reward weight |
| `w_goal` | `5.0` | Per-explorer goal-reached weight |
| `w_proximity` | `0.5` | Inter-rover proximity penalty weight |
| `w_action` | `0.1` | Action-rate penalty weight |
| `env_rate` | `1/50` | Physics rate (50 Hz) |
| `agent_rate` | `1/25` | Decision rate (25 Hz, decimation=2) |
