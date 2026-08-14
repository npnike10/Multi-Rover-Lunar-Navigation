"""Simulator smoke tests for SRB-compatible MARL waypoint navigation.

Run with: ``$ISAAC_SIM_PYTHON tests/test_env.py``.

The checks cover the three-rover interface, Dec-POMDP reward symmetry, the
unmodified SRB per-rover reward aggregation, observation-noise reset buffers,
and short-horizon simulator stability.
"""

from srb.core.app import AppLauncher

launcher = AppLauncher(enable_cameras=False, headless=True)
app = launcher.app

import gymnasium as gym
import torch

import srb.tasks
from srb.tasks.mobile.marl_waypoint_navigation.task import MarlWaypointTaskCfg


ENV_NAME = "srb/marl_waypoint_navigation"
NUM_ROVERS = 3
OBS_DIM = 4 + 2 * (NUM_ROVERS - 1)
STATE_DIM = 8 * NUM_ROVERS


def make_env():
    """Create the default three-rover task with a predictable safety radius."""
    cfg = MarlWaypointTaskCfg()
    cfg.proximity_safe_distance = 1.0
    env = gym.make(ENV_NAME, cfg=cfg)
    observation, _ = env.reset()
    return env, observation


def test_one_rover_configuration():
    """Verify the public one-rover override without launching another scene."""
    print("\n" + "=" * 60)
    print("TEST 1: One-Rover Configuration")
    print("=" * 60)

    config = MarlWaypointTaskCfg(num_rovers=1)
    passed = (
        config.possible_agents == ["rover_1"]
        and list(config.robots) == ["rover_1"]
        and config.local_observation_dim == 4
        and config.global_state_dim == 8
        and hasattr(config.events, "target_rover_1_pose_evolution")
    )
    print(f"  agents={config.possible_agents}")
    print(
        "  dimensions: "
        f"observation={config.local_observation_dim}, state={config.global_state_dim}"
    )
    print(f"  RESULT: {'PASSED' if passed else 'FAILED'}")
    return passed


def test_spaces(env, observation):
    """Verify the dynamic observation, state, and bounded action spaces."""
    print("\n" + "=" * 60)
    print("TEST 2: Space Verification")
    print("=" * 60)

    unwrapped = env.unwrapped
    passed = True
    for agent_id in unwrapped.possible_agents:
        action_space = unwrapped.action_spaces[agent_id]
        correct_observation = observation[agent_id].shape[-1] == OBS_DIM
        correct_action = action_space.shape == (2,)
        bounded_action = bool(
            torch.all(torch.as_tensor(action_space.low) == -1.0)
            and torch.all(torch.as_tensor(action_space.high) == 1.0)
        )
        print(
            f"  {agent_id}: obs={observation[agent_id].shape[-1]} "
            f"(expect {OBS_DIM}), action={action_space.shape}, "
            f"bounded={bounded_action}"
        )
        passed &= correct_observation and correct_action and bounded_action

    state = unwrapped.state() if callable(unwrapped.state) else unwrapped.state
    correct_state = state is not None and state.shape[-1] == STATE_DIM
    print(f"  state: dim={None if state is None else state.shape[-1]} (expect {STATE_DIM})")
    passed &= correct_state
    print(f"  RESULT: {'PASSED' if passed else 'FAILED'}")
    return passed


def test_observation_state_finiteness(env, observation):
    """Check that noisy observations and unnoised state are finite."""
    print("\n" + "=" * 60)
    print("TEST 3: Observation and State Finiteness")
    print("=" * 60)

    unwrapped = env.unwrapped
    passed = True
    for agent_id in unwrapped.possible_agents:
        value = observation[agent_id]
        finite = bool(torch.isfinite(value).all())
        print(f"  {agent_id}: finite={finite}, shape={tuple(value.shape)}")
        passed &= finite

    state = unwrapped.state() if callable(unwrapped.state) else unwrapped.state
    finite_state = state is not None and bool(torch.isfinite(state).all())
    print(f"  state: finite={finite_state}")
    passed &= finite_state
    print(f"  RESULT: {'PASSED' if passed else 'FAILED'}")
    return passed


def zero_actions(unwrapped):
    """Build a correctly shaped zero command for every active rover."""
    return {
        agent_id: torch.zeros(1, 2, device=unwrapped.device)
        for agent_id in unwrapped.possible_agents
    }


def test_shared_reward_aggregation(env):
    """Verify all agents receive the mean of the unmodified SRB rewards."""
    print("\n" + "=" * 60)
    print("TEST 4: Shared SRB Reward Aggregation")
    print("=" * 60)

    unwrapped = env.unwrapped
    env.reset()
    _, rewards, _, _, _ = env.step(zero_actions(unwrapped))
    individual = torch.stack(
        [unwrapped._srb_reward(agent_id) for agent_id in unwrapped.possible_agents],
        dim=-1,
    ).mean(dim=-1)
    passed = True
    for agent_id in unwrapped.possible_agents:
        matches = bool(torch.allclose(rewards[agent_id], individual, atol=1.0e-6))
        print(f"  {agent_id}: shared reward matches mean SRB reward={matches}")
        passed &= matches

    values = [rewards[agent_id] for agent_id in unwrapped.possible_agents]
    symmetric = all(torch.allclose(value, values[0]) for value in values[1:])
    print(f"  Dec-POMDP reward symmetry={symmetric}")
    passed &= symmetric
    print(f"  RESULT: {'PASSED' if passed else 'FAILED'}")
    return passed


def test_noise_and_proximity_configuration(env):
    """Validate reset-persistent noise buffers and configurable safe distance."""
    print("\n" + "=" * 60)
    print("TEST 5: Noise and Proximity Configuration")
    print("=" * 60)

    unwrapped = env.unwrapped
    initial_noise = {
        key: value.clone() for key, value in unwrapped._episodic_xy_noise.items()
    }
    env.reset()
    changed_noise = any(
        not torch.equal(initial_noise[key], value)
        for key, value in unwrapped._episodic_xy_noise.items()
    )
    target_relation_count = len(unwrapped._episodic_xy_noise)
    expected_relation_count = NUM_ROVERS * NUM_ROVERS
    safe_distance_ok = abs(unwrapped._proximity_safe_distance - 1.0) < 1.0e-6
    print(f"  directed XY relations={target_relation_count} (expect {expected_relation_count})")
    print(f"  episodic XY samples refresh at reset={changed_noise}")
    print(f"  configured proximity safe distance={safe_distance_ok}")
    passed = (
        target_relation_count == expected_relation_count
        and changed_noise
        and safe_distance_ok
    )
    print(f"  RESULT: {'PASSED' if passed else 'FAILED'}")
    return passed


def test_target_motion_and_bounds(env):
    """Verify that every virtual target evolves within the configured bounds."""
    print("\n" + "=" * 60)
    print("TEST 6: Target Motion and Bounds")
    print("=" * 60)

    unwrapped = env.unwrapped
    env.reset()
    initial_goals = {
        agent_id: goal.clone() for agent_id, goal in unwrapped._goals.items()
    }
    for _ in range(5):
        env.step(zero_actions(unwrapped))

    moved = any(
        not torch.allclose(initial_goals[agent_id], unwrapped._goals[agent_id])
        for agent_id in unwrapped.possible_agents
    )
    bound = 0.5 * unwrapped.cfg.target_pos_range_ratio * unwrapped.cfg.spacing
    in_bounds = all(
        bool(
            torch.all(
                torch.abs(goal[:, :2] - unwrapped.scene.env_origins[:, :2])
                <= bound + 1.0e-6
            )
        )
        for goal in unwrapped._goals.values()
    )
    print(f"  at least one target moved={moved}")
    print(f"  all target XY values in ±{bound:.1f} m bounds={in_bounds}")
    passed = moved and in_bounds
    print(f"  RESULT: {'PASSED' if passed else 'FAILED'}")
    return passed


def test_multi_step_stability(env):
    """Run a short rollout and reject non-finite task outputs."""
    print("\n" + "=" * 60)
    print("TEST 7: Multi-Step Stability")
    print("=" * 60)

    unwrapped = env.unwrapped
    env.reset()
    passed = True
    for step in range(50):
        actions = {
            agent_id: torch.rand(1, 2, device=unwrapped.device) * 2.0 - 1.0
            for agent_id in unwrapped.possible_agents
        }
        observation, rewards, _, _, _ = env.step(actions)
        values = [*observation.values(), *rewards.values()]
        if not all(bool(torch.isfinite(value).all()) for value in values):
            print(f"  step {step}: non-finite observation or reward")
            passed = False
            break

    print(f"  RESULT: {'PASSED' if passed else 'FAILED'}")
    return passed


print("\n" + "#" * 60)
print("# SRB-Compatible MARL Waypoint Navigation — Smoke Tests")
print("#" * 60)

environment, initial_observation = make_env()
results = {
    "one_rover_config": test_one_rover_configuration(),
    "spaces": test_spaces(environment, initial_observation),
    "finiteness": test_observation_state_finiteness(environment, initial_observation),
    "reward": test_shared_reward_aggregation(environment),
    "noise_proximity": test_noise_and_proximity_configuration(environment),
    "target_motion": test_target_motion_and_bounds(environment),
    "stability": test_multi_step_stability(environment),
}

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
all_passed = True
for name, passed in results.items():
    print(f"  {name:20s} {'PASSED' if passed else 'FAILED'}")
    all_passed &= passed
print(f"\n  {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
print("=" * 60)

environment.close()
app.close()
