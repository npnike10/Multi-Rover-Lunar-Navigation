from pathlib import Path
from typing import TYPE_CHECKING, Literal

import gymnasium
from isaacsim.simulation_app import SimulationApp

from srb.integrations.skrl.wrapper import (
    SkrlEnvWrapper,
    SkrlMultiAgentEnvWrapper,
)
from srb.utils import logging
from srb.utils.cfg import last_file, stamp_dir
from srb.wrappers import maybe_wrap_action_smoothing

if TYPE_CHECKING:
    from srb._typing import AnyEnv, AnyEnvCfg

FRAMEWORK_NAME = "skrl"


def _unwrap_torch_optimizer_dynamo_wrappers() -> None:
    """Avoid Isaac Sim's Torch Dynamo import path when skrl builds optimizers."""

    def _unwrap_method(obj, name: str) -> None:
        method = getattr(obj, name, None)
        wrapped = getattr(method, "__wrapped__", None)
        if wrapped is not None:
            setattr(obj, name, wrapped)

    try:
        import torch.optim.optimizer as optimizer_module

        for name in ("state_dict", "load_state_dict", "zero_grad", "add_param_group"):
            _unwrap_method(optimizer_module.Optimizer, name)
    except Exception:
        pass

    try:
        import torch.optim.adam as adam_module

        _unwrap_method(adam_module.Adam, "step")
        if (wrapped := getattr(adam_module.adam, "__wrapped__", None)) is not None:
            adam_module.adam = wrapped
    except Exception:
        pass


def run(
    workflow: Literal["train", "eval"],
    env: "AnyEnv | gymnasium.Env",
    sim_app: SimulationApp,
    env_id: str,
    env_cfg: "AnyEnvCfg | None",
    agent_cfg: dict,
    logdir: Path,
    model: Path,
    continue_training: bool | None = None,
    **kwargs,
):
    # Pop the entire smoothing config dictionary to be handled separately.
    smoothing_cfg = agent_cfg.pop("smoothing", {})

    # Determine checkpoint path
    if model:
        from_checkpoint = model
    elif workflow == "eval" or continue_training:
        from_checkpoint = last_file(
            logdir.joinpath("checkpoints"), modification_time=True
        )
    else:
        from_checkpoint = ""
    if from_checkpoint:
        logging.info(f"Loading model from {from_checkpoint}")

    # Special handling for eval workflow
    if workflow == "eval":
        logdir = stamp_dir(logdir.joinpath("eval"))

    # Update agent config
    agent_cfg["seed"] = env_cfg.seed if env_cfg else 0
    agent_cfg["agent"]["experiment"]["directory"] = logdir.parent
    agent_cfg["agent"]["experiment"]["experiment_name"] = logdir

    unwrapped_env = getattr(env, "unwrapped", env)
    is_multi_agent = hasattr(unwrapped_env, "possible_agents")

    # Enable action smoothing if enabled
    if is_multi_agent and smoothing_cfg.get("enabled", False):
        logging.warning("Action smoothing is not supported for skrl multi-agent envs.")
    else:
        env = maybe_wrap_action_smoothing(
            env,  # type: ignore
            smoothing_cfg,
        )

    # Wrap the environment
    env = (
        SkrlMultiAgentEnvWrapper(env)  # type: ignore
        if is_multi_agent
        else SkrlEnvWrapper(env)  # type: ignore
    )

    # Create the runner
    _unwrap_torch_optimizer_dynamo_wrappers()
    from skrl.utils.runner.torch import Runner

    runner = Runner(
        env,  # type: ignore
        agent_cfg,
    )

    # Load checkpoint if needed
    if from_checkpoint:
        runner.agent.load(
            from_checkpoint,  # type: ignore
        )

    # Run the workflow
    runner.run(mode=workflow)
