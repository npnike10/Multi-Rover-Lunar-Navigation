from srb.core.app import AppLauncher
launcher = AppLauncher(enable_cameras=True)
app = launcher.app

import gymnasium as gym
import torch
import imageio
import numpy as np
import random

# Import tasks to register them
import srb.tasks

env_name = "srb/marl_waypoint_navigation"
from srb.tasks.mobile.marl_waypoint_navigation.task import MarlWaypointTaskCfg

# Random seed so SimForge generates unique, rough terrain each run
seed = random.randint(1, 9999)
print(f"Using terrain seed: {seed}")
cfg = MarlWaypointTaskCfg()
cfg.seed = seed

env = gym.make(env_name, cfg=cfg, render_mode="rgb_array")
obs, _ = env.reset()

frames = []
NUM_STEPS = 500

# Drastically different directions so rovers spread apart and never clash:
#   supporter  → hard left arc
#   explorer_1 → hard right arc  
#   explorer_2 → straight backward
fixed_actions = {
    "supporter":  torch.tensor([[ 0.9, -1.0]], dtype=torch.float32, device=env.unwrapped.device),
    "explorer_1": torch.tensor([[ 0.9,  1.0]], dtype=torch.float32, device=env.unwrapped.device),
    "explorer_2": torch.tensor([[-0.9,  0.0]], dtype=torch.float32, device=env.unwrapped.device),
}


def update_camera(env):
    """Static bird's-eye isometric view covering the entire navigation area."""
    # Look at origin from a high, wide angle so we see all rovers and targets
    env.unwrapped.sim.set_camera_view(
        eye=(-12.0, -12.0, 15.0),
        target=(0.0, 0.0, 0.0),
    )


# Set initial camera
update_camera(env)

for i in range(NUM_STEPS):
    # Small noise keeps motion natural but preserves divergent directions
    actions = {
        agent: fixed_actions[agent] + torch.randn_like(fixed_actions[agent]) * 0.04
        for agent in env.unwrapped.possible_agents
    }

    obs, rewards, dones, truncs, infos = env.step(actions)

    # Follow the supporter rover
    update_camera(env)

    img = env.render()
    if img is not None:
        frames.append(img)

    if i % 50 == 0:
        print(f"Step {i}/{NUM_STEPS}  rewards={rewards}")

output_path = "/home/niket/Documents/TRG/MRLN/marl_navigation.mp4"
print(f"Saving MP4 ({len(frames)} frames) to {output_path}...")
writer = imageio.get_writer(output_path, fps=20, codec='libx264', quality=8)
for frame in frames:
    writer.append_data(frame)
writer.close()
print(f"Done! Seed was: {seed}")

env.close()
app.close()
