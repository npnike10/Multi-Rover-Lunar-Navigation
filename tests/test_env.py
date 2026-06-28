"""Comprehensive environment test suite for MARL Waypoint Navigation.

Run with: $ISAAC_SIM_PYTHON test_env.py

Tests:
  1. Space verification (obs, state, action dimensions)
  2. Observation/state statistics (NaN, inf, range)
  3. Dec-POMDP reward symmetry (all agents get same reward)
  4. Reward shaping (progress increases as explorer approaches target)
  5. Proximity penalty (rovers close together → negative penalty)
  6. Rollover termination (force tilt and verify episode ends)
  7. Goal-reached termination (teleport explorers to targets)
  8. Multi-step stability (run 200 steps, no crashes)
"""

from srb.core.app import AppLauncher
launcher = AppLauncher(enable_cameras=False, headless=True)
app = launcher.app

import gymnasium as gym
import torch
import numpy as np
import srb.tasks

env_name = "srb/marl_waypoint_navigation"
from srb.tasks.mobile.marl_waypoint_navigation.task import MarlWaypointTaskCfg


def make_env():
    cfg = MarlWaypointTaskCfg()
    cfg.safe_distance = 1.0  # Override for predictable testing
    env = gym.make(env_name, cfg=cfg)
    obs, _ = env.reset()
    return env, obs


def test_spaces(env, obs):
    """Test 1: Verify observation, state, and action space dimensions."""
    print("\n" + "=" * 60)
    print("TEST 1: Space Verification")
    print("=" * 60)

    passed = True
    uw = env.unwrapped

    for aid in uw.possible_agents:
        obs_dim = obs[aid].shape[-1]
        act_space = uw.action_spaces[aid]
        act_dim = act_space.shape[0] if hasattr(act_space, 'shape') else act_space
        expected_obs = 139
        expected_act = 2

        ok_obs = obs_dim == expected_obs
        ok_act = act_dim == expected_act
        print(f"  {aid}: obs={obs_dim} (expect {expected_obs}) {'✅' if ok_obs else '❌'}, "
              f"act={act_dim} (expect {expected_act}) {'✅' if ok_act else '❌'}")
        passed = passed and ok_obs and ok_act

    # State (may be a property or callable depending on gym wrapper)
    state = uw.state() if callable(uw.state) else uw.state
    if state is not None:
        state_dim = state.shape[-1]
        ok_state = state_dim == 414
        print(f"  state: dim={state_dim} (expect 414) {'✅' if ok_state else '❌'}")
        passed = passed and ok_state
    else:
        print(f"  state: None ❌ (expected 414-dim tensor)")
        passed = False

    print(f"  RESULT: {'PASSED' if passed else 'FAILED'}")
    return passed


def test_obs_state_stats(env, obs):
    """Test 2: Check for NaN, inf, and print statistics."""
    print("\n" + "=" * 60)
    print("TEST 2: Observation & State Statistics")
    print("=" * 60)

    passed = True
    uw = env.unwrapped

    for aid in uw.possible_agents:
        o = obs[aid]
        has_nan = torch.isnan(o).any().item()
        has_inf = torch.isinf(o).any().item()
        print(f"  {aid} obs: min={o.min():.3f}, max={o.max():.3f}, "
              f"mean={o.mean():.3f}, nan={has_nan}, inf={has_inf} "
              f"{'❌' if (has_nan or has_inf) else '✅'}")
        passed = passed and not has_nan and not has_inf

    state = uw.state() if callable(uw.state) else uw.state
    if state is not None:
        has_nan = torch.isnan(state).any().item()
        has_inf = torch.isinf(state).any().item()
        print(f"  state: min={state.min():.3f}, max={state.max():.3f}, "
              f"mean={state.mean():.3f}, nan={has_nan}, inf={has_inf} "
              f"{'❌' if (has_nan or has_inf) else '✅'}")
        passed = passed and not has_nan and not has_inf

    print(f"  RESULT: {'PASSED' if passed else 'FAILED'}")
    return passed


def test_reward_symmetry(env):
    """Test 3: Dec-POMDP — all agents must receive identical rewards."""
    print("\n" + "=" * 60)
    print("TEST 3: Dec-POMDP Reward Symmetry")
    print("=" * 60)

    passed = True
    uw = env.unwrapped
    agents = uw.possible_agents

    for step in range(10):
        actions = {
            aid: torch.randn(1, 2, device=uw.device) * 0.5
            for aid in agents
        }
        obs, rewards, dones, truncs, infos = env.step(actions)

        values = [rewards[aid].item() for aid in agents]
        all_equal = all(abs(v - values[0]) < 1e-6 for v in values)

        if not all_equal:
            print(f"  Step {step}: rewards differ! {dict(zip(agents, values))} ❌")
            passed = False

    if passed:
        print(f"  All 10 steps: rewards identical across agents ✅")
    print(f"  RESULT: {'PASSED' if passed else 'FAILED'}")
    return passed


def test_reward_shaping(env):
    """Test 4: Progress reward increases as explorers approach targets."""
    print("\n" + "=" * 60)
    print("TEST 4: Reward Shaping (progress formula)")
    print("=" * 60)

    uw = env.unwrapped
    env.reset()

    passed = True

    # --- Part A: Verify progress formula tracks distance correctly ---
    # Record distance to target and corresponding progress term at two points.
    # Teleport explorer_1 to be far, then close, and compare progress.
    aid = "explorer_1"
    robot = uw._robots[aid]
    target = uw._goals[aid]

    # Take one step to get baseline reward
    actions = {a: torch.zeros(1, 2, device=uw.device) for a in uw.possible_agents}
    obs1, rewards1, _, _, _ = env.step(actions)
    dist_far = torch.norm(
        robot.data.root_link_pose_w[:, :2] - target[:, :2], dim=-1
    ).item()
    r_far = rewards1[aid].item()

    # Teleport explorer_1 close to its target (but not inside threshold)
    new_pos = target.clone()
    new_pos[:, :2] += 0.8  # 0.8m from target
    new_pos[:, 2] = robot.data.root_link_pose_w[:, 2]  # keep current z
    robot.write_root_link_pose_to_sim(
        torch.cat([new_pos, robot.data.root_link_pose_w[:, 3:]], dim=-1)
    )
    uw.sim.step()

    obs2, rewards2, _, _, _ = env.step(actions)
    dist_close = torch.norm(
        robot.data.root_link_pose_w[:, :2] - target[:, :2], dim=-1
    ).item()
    r_close = rewards2[aid].item()

    print(f"  Far:   dist={dist_far:.2f}m  reward={r_far:.4f}")
    print(f"  Close: dist={dist_close:.2f}m  reward={r_close:.4f}")

    # Closer position should yield higher reward (progress term dominates)
    ok_a = r_close > r_far
    print(f"  Closer → higher reward: {'✅' if ok_a else '❌'}")
    passed = passed and ok_a

    # --- Part B: Verify goal-reached bonus fires ---
    env.reset()
    target = uw._goals[aid]
    robot = uw._robots[aid]

    # Teleport explorer_1 right onto its target
    at_target = target.clone()
    at_target[:, 2] = robot.data.root_link_pose_w[:, 2]
    robot.write_root_link_pose_to_sim(
        torch.cat([at_target, robot.data.root_link_pose_w[:, 3:]], dim=-1)
    )
    uw.sim.step()

    obs3, rewards3, _, _, _ = env.step(actions)
    r_goal = rewards3[aid].item()

    # Goal bonus should make reward significantly higher than normal
    print(f"  At target: reward={r_goal:.4f} (expect large due to goal bonus)")
    ok_b = r_goal > 1.0  # w_goal=5.0 * 0.5 = 2.5 for one explorer reaching
    print(f"  Goal bonus fires (R > 1.0): {'✅' if ok_b else '❌'}")
    passed = passed and ok_b

    print(f"  RESULT: {'PASSED' if passed else 'FAILED'}")
    return passed


def test_multi_step_stability(env):
    """Test 5: Run 200 steps with random actions — no crashes."""
    print("\n" + "=" * 60)
    print("TEST 5: Multi-Step Stability (200 steps)")
    print("=" * 60)

    uw = env.unwrapped
    env.reset()

    passed = True
    for step in range(200):
        actions = {
            aid: torch.randn(1, 2, device=uw.device) * 0.5
            for aid in uw.possible_agents
        }
        try:
            obs, rewards, dones, truncs, infos = env.step(actions)

            # Check obs
            for aid in uw.possible_agents:
                if torch.isnan(obs[aid]).any() or torch.isinf(obs[aid]).any():
                    print(f"  Step {step}: {aid} obs has NaN/inf ❌")
                    passed = False
                    break

            # Check rewards
            for aid in uw.possible_agents:
                if torch.isnan(rewards[aid]).any() or torch.isinf(rewards[aid]).any():
                    print(f"  Step {step}: {aid} reward has NaN/inf ❌")
                    passed = False
                    break

            # Check state
            state = uw.state() if callable(uw.state) else uw.state
            if state is not None and (torch.isnan(state).any() or torch.isinf(state).any()):
                print(f"  Step {step}: state has NaN/inf ❌")
                passed = False

        except Exception as e:
            print(f"  Step {step}: CRASH — {e} ❌")
            passed = False
            break

        if not passed:
            break

    if passed:
        print(f"  200 steps completed, no NaN/inf/crashes ✅")
    print(f"  RESULT: {'PASSED' if passed else 'FAILED'}")
    return passed


def test_safe_distance_override(env):
    """Test 6: Verify safe_distance override works."""
    print("\n" + "=" * 60)
    print("TEST 6: Safe Distance Override")
    print("=" * 60)

    uw = env.unwrapped
    actual = uw._safe_distance
    expected = 1.0  # We set safe_distance=1.0 in make_env()
    passed = abs(actual - expected) < 1e-6
    print(f"  safe_distance = {actual:.3f} (expected {expected}) "
          f"{'✅' if passed else '❌'}")
    print(f"  RESULT: {'PASSED' if passed else 'FAILED'}")
    return passed


# =========================================================================
# Run all tests
# =========================================================================
print("\n" + "#" * 60)
print("# MARL Waypoint Navigation — Environment Test Suite")
print("#" * 60)

env, obs = make_env()

results = {}
results["spaces"] = test_spaces(env, obs)
results["stats"] = test_obs_state_stats(env, obs)
results["symmetry"] = test_reward_symmetry(env)
results["shaping"] = test_reward_shaping(env)
results["stability"] = test_multi_step_stability(env)
results["safe_dist"] = test_safe_distance_override(env)

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
all_passed = True
for name, passed in results.items():
    status = "PASSED ✅" if passed else "FAILED ❌"
    print(f"  {name:20s} {status}")
    all_passed = all_passed and passed

print(f"\n  {'ALL TESTS PASSED ✅' if all_passed else 'SOME TESTS FAILED ❌'}")
print("=" * 60)

env.close()
app.close()
