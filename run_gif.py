import argparse
from srb.core.app import AppLauncher

launcher = AppLauncher(enable_cameras=True)
app = launcher.app

import gymnasium as gym
import torch
import imageio
import numpy as np

# Import tasks to register them
import srb.tasks

env_name = "srb/marl_waypoint_navigation"
from srb.tasks.mobile.marl_waypoint_navigation.task import MarlWaypointTaskCfg
cfg = MarlWaypointTaskCfg()
env = gym.make(env_name, cfg=cfg, render_mode="rgb_array")
env.reset()

frames = []
# Run for 100 steps
for i in range(100):
    # Sample random actions for each agent (convert to torch tensor with batch dim)
    actions = {
        agent: torch.tensor(
            env.action_space(agent).sample(), 
            dtype=torch.float32, 
            device=env.unwrapped.device
        ).unsqueeze(0)  # (action_dim,) -> (1, action_dim)
        for agent in env.possible_agents
    }
    
    # In SRB/IsaacLab MARL, actions are passed as a dict
    obs, rewards, dones, truncs, infos = env.step(actions)
    
    # Render the current frame
    img = env.render()
    if img is not None:
        frames.append(img)
    
    if i % 10 == 0:
        print(f"Step {i}/100")

# Save as GIF
output_path = "/home/niket/.gemini/antigravity-cli/brain/920fbfba-dbe6-4f10-8774-60fc4b385b6f/marl_navigation.gif"
print(f"Saving GIF to {output_path}...")
imageio.mimsave(output_path, frames, fps=30)
print("Done!")

env.close()
app.close()
