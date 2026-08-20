# MARL Waypoint Navigation — SRB-Compatible Design

## Overview

`marl_waypoint_navigation` is a homogeneous-rover **Dec-POMDP** extension of
SRB's state-based single-rover waypoint-navigation task. Every rover has an
individual moving planar waypoint and a decentralized noisy observation. The
task can return either each rover's own reward or a common mean team reward.
It also reproduces the original task's action and observation latency model.

`num_rovers` defines the homogeneous Leo team.  Its default is `3`, producing
`rover_1`, `rover_2`, and `rover_3`; set `env.num_rovers=1` to create the
single-agent replication configuration.  There are no supporter/explorer
roles, RayCaster terrain features, IMU features, rollover logic, or
goal-completion termination.

## World, reset, and target dynamics

- The task uses SRB's Moon-domain `AssetVariant.PROCEDURAL` scenery.  At the
  default 32 m spacing, this resolves to the default procedural `MoonSurface`.
- A one-rover run uses SRB's original reset ranges: XY in `[-0.5, 0.5]` m,
  root height in `[0.4, 0.6]` m, yaw in `[-π, π]`, and the original initial
  velocity ranges.  Multi-rover runs retain those height, yaw, and velocity
  distributions but shift their XY window centers to prevent initial overlap.
- A one-rover target starts at its environment origin with identity
  orientation and uses SRB's XY bounds `±0.45 × environment spacing` (±14.4 m
  at the default spacing). For two or more rovers, each target instead starts
  at, and remains inside, a distinct persistent region centered on that
  rover's compact reset-window center. Each target starts at that center, then
  moves through its own disjoint outward corridor with `6.0 m` extent
  (`env.multi_rover_target_motion_half_range`) and `3.0 m` lateral half-width
  (`env.multi_rover_target_lateral_half_range`). The three rover reset centers
  are spaced on a `1.0 m` radius circle (`env.multi_rover_spawn_radius`), so
  close initial interactions remain possible while targets eventually move
  apart. This prevents all
  rovers from initially pursuing the same origin waypoint. Targets evolve
  every 0.05 s via the same `offset_pose_natural` event: XY step size
  `[0.005, 0.01]` m, position smoothness `0.99`, step-size smoothness `0.8`,
  and yaw-only orientation with smoothness `0.8`.
- Episodes end only at the 60 s time limit.

## Dec-POMDP definition

For `N` rovers, all positions are two-dimensional task positions.

| Element | Definition |
|:---|:---|
| Agents | `I = {1, …, N}`; every rover owns target `g_i`. |
| Actions | `a_i = (a_linear, a_angular) ∈ [-1, 1]^2`. The default Leo drive maps them to `v_linear = 0.4 a_linear` m/s and `v_angular = (π/3) a_angular` rad/s (60°/s). |
| Local observation | A delayed noisy `[relative XY to g_i (2), relative target yaw as sin/cos (2), relative XY to each rover j ≠ i (2(N-1))]`; dimension `4 + 2(N-1)`. |
| Critic state | Unnoised environment-relative rover `[XY, sin(yaw), cos(yaw)]` for all rovers, followed by the same fields for all targets; dimension `8N`. |
| Reward | `env.reward_mode=individual` (default): each rover gets its own SRB reward minus its own optional proximity penalty. `team`: every rover gets the mean of those individual rewards. |
| Termination | No success or rollover termination; timeout only. |

The centralized-critic state is intentionally a compact planar task state.  It
does not claim to be the complete simulator-Markov state: simulator velocities
and latent target-motion state remain privileged physical information outside
this vector.

## Action and observation latency

The task maintains independent action and observation delay buffers for every
rover in every parallel environment. By default, an action delay is sampled
uniformly from `0..3` agent steps and an observation delay from `0..1` agent
steps at reset. At the default 25 Hz agent rate, these correspond to up to
120 ms and 40 ms respectively. Buffers are zero-filled at reset, so the
initial delayed commands or observations are valid zero tensors. With current
delays (d^a_{i,t}) and (d^o_{i,t}), the applied command and policy input
are

```text
applied_action[i, t] = policy_action[i, t - d^a[i, t]]
policy_observation[i, t] = noisy_observation[i, t - d^o[i, t]]
```

`env.action_delay_on_step_change_prob` and
`env.observation_delay_on_step_change_prob` default to `0.01`; at their
default one-second check interval, they can move each delay by one bounded
step. Set both delay ranges to `0` to disable latency.

## Observation noise

The task follows the implementation of SRB's single-rover waypoint task:

```text
relative XY measurement = true body-frame relative XY
                        + episodic offset + per-step noise

episodic XY offset ~ Normal(0, 0.01² I₂)
per-step XY noise  ~ Normal(0, 0.0025² I₂)

episodic target-yaw offset ~ Normal(0, (2.5°)²)
per-step target-yaw noise  ~ Normal(0, (0.5°)²)
```

The per-step term is noise, not a persistent bias.  Each directed
observer-to-entity XY relation has an independent episodic offset and fresh
per-step sample.  Thus rover `i`'s measurement of rover `j` is independently
noised from rover `j`'s measurement of rover `i`.  Target yaw follows the
single-rover SRB convention and is encoded as `[sin(yaw), cos(yaw)]` after
noise is applied.

This corrects the earlier proposed XY values of `0.012` and `0.00252`: SRB's
current waypoint-task implementation uses `0.01` and `0.0025`, respectively.
Target yaw must remain observable because it is directly rewarded.

## Configurable reward

For each rover, the task computes SRB's original state-based components using
its true target-relative pose and normalized action change:

```text
action_rate                    = mean((a_t - a_{t-1})²)
distance                       = ||relative_target_XY||
position_precision             = 1 - tanh(distance / 0.05)
orientation_precision          = position_precision
                               * (1 - tanh(|relative_target_yaw| / 0.2618))

r_i = -0.5 * action_rate
    - distance²
    + (1 - tanh(|heading_to_target| / 0.7854))
    + 4  * position_precision
    + 8  * orientation_precision
    + 32 * orientation_precision * (1 - tanh(action_rate / 0.1))

p_i = mean_{j \neq i}(max(0, 1 - d_{ij} / d_safe))
R_i = r_i - w_proximity * p_i
R_team = mean_i(R_i)
```

`env.reward_mode` selects the returned reward: `individual` (the default)
returns `R_i` to rover `i`; `team` returns `R_team` to every rover. The
proximity term is therefore always assigned to the rovers in the close pair,
even in team mode before averaging. `w_proximity` defaults to `1.0`.
`d_safe` defaults to one Leo rover length, `0.3587 m`, and can be overridden
with `env.proximity_safe_distance`. The pair term is zero for a single rover.
Consequently, one rover with the default proximity weight has the original SRB
reward exactly.

## Compatibility notes

- The literal single-rover task class defaults to the Perseverance rover
  (0.7 m/s and 75°/s).  This MARL task intentionally uses Leo Rover so that its
  requested 0.4 m/s and 60°/s action mapping is preserved.
- The target-relative target yaw is an orientation error, not the heading from
  rover to target.  Both are used separately by the SRB reward.
- Changing the number of rovers changes the agent set and the local/state
  dimensions.  Use `env.num_rovers=<N>` to change it; policies and critics
  trained with one team size are not shape-compatible with another.

## Three-rover MAPPO baseline

Use `skrl_mappo` with the default `env.num_rovers=3`.  SRB's `MAPPO_RNN`
extension shares one decentralized policy and one centralized critic across
the three homogeneous rovers. Both networks receive an internal one-hot rover
ID: the policy receives its 8-D local observation plus that ID, and the critic
receives the common `24`-D state plus that ID. Each rover keeps a separate
recurrent rollout state and memory buffer, while one optimizer aggregates all
three rovers' losses into shared parameters. Its configuration matches the
single-rover PPO-RNN baseline wherever
the algorithms overlap: 128-step rollouts, 1024 samples per-rover minibatches,
16 PPO epochs, `γ=0.997`, `λ=0.95`, linear `1e-4 → 0` learning rate, clip
range `0.2`, entropy coefficient `0.01`, gradient norm `0.5`, and `[384, 384]`
ELU MLP heads and a 384-unit LSTM. Stock SKRL MAPPO is feed-forward and does
not implement cross-agent parameter sharing; this task uses SRB's extension.

Because SKRL owns one 128-step rollout buffer per rover, the 1024 minibatch is
defined per rover rather than across all three buffers.  Choose
`env.num_envs=8` or another multiple of eight.  For example:

```bash
srb agent train --headless --algo skrl_mappo -e marl_waypoint_navigation \
  env.num_rovers=3 env.num_envs=8
```
