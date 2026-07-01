import sys
import types
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

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


def _install_torch_dynamo_graph_break_stub() -> None:
    """Provide the optimizer graph-break hook without importing Torch Dynamo."""
    import torch

    dynamo_module = sys.modules.get("torch._dynamo")
    if dynamo_module is None:
        dynamo_module = types.ModuleType("torch._dynamo")
        sys.modules["torch._dynamo"] = dynamo_module

    if not hasattr(dynamo_module, "graph_break"):
        dynamo_module.graph_break = lambda *args, **kwargs: None  # type: ignore[attr-defined]

    torch._dynamo = dynamo_module  # type: ignore[attr-defined]


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


def _install_native_wandb_scalar_logging() -> None:
    """Mirror skrl's TensorBoard scalar writes to native WandB history."""

    def _tracking_payload(agent: Any) -> dict[str, float]:
        import numpy as np

        payload: dict[str, float] = {}
        for key, values in agent.tracking_data.items():
            if not values:
                continue
            if key.endswith("(min)"):
                value = np.min(values)
            elif key.endswith("(max)"):
                value = np.max(values)
            else:
                value = np.mean(values)
            payload[key] = float(value)
        return payload

    def _wandb_enabled(agent: Any) -> bool:
        return bool(agent.cfg.get("experiment", {}).get("wandb", False))

    def _ensure_global_step_axis(wandb: Any) -> None:
        run = getattr(wandb, "run", None)
        if run is None or getattr(run, "_srb_global_step_axis_defined", False):
            return
        wandb.define_metric("global_step")
        wandb.define_metric("*", step_metric="global_step")
        setattr(run, "_srb_global_step_axis_defined", True)

    def _flush_writer(agent: Any) -> None:
        writer = getattr(agent, "writer", None)
        flush = getattr(writer, "flush", None)
        if callable(flush):
            flush()

    def _wrap_write_tracking_data(cls: type) -> None:
        original = getattr(cls, "write_tracking_data")
        if getattr(original, "_srb_wandb_wrapped", False):
            return

        def write_tracking_data(self: Any, timestep: int, timesteps: int) -> None:
            payload = _tracking_payload(self) if _wandb_enabled(self) else {}
            original(self, timestep, timesteps)
            _flush_writer(self)
            if not payload:
                return

            try:
                import wandb

                if wandb.run is not None:
                    _ensure_global_step_axis(wandb)
                    payload["global_step"] = int(timestep)
                    wandb.log(payload, step=timestep, commit=True)
            except Exception as exc:
                logging.warning(f"Failed to log skrl metrics to WandB: {exc}")

        write_tracking_data.__wrapped__ = original  # type: ignore[attr-defined]
        write_tracking_data._srb_wandb_wrapped = True  # type: ignore[attr-defined]
        setattr(cls, "write_tracking_data", write_tracking_data)

    try:
        from skrl.agents.torch.base import Agent
        from skrl.multi_agents.torch.base import MultiAgent

        _wrap_write_tracking_data(Agent)
        _wrap_write_tracking_data(MultiAgent)
    except Exception as exc:
        logging.warning(f"Failed to install native WandB scalar logging: {exc}")


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
    if agent_cfg["agent"]["experiment"].get("wandb", False):
        wandb_kwargs = agent_cfg["agent"]["experiment"].setdefault("wandb_kwargs", {})
        # Use native W&B logging so the run step is the skrl environment
        # timestep, not W&B's TensorBoard-sync row counter.
        wandb_kwargs["sync_tensorboard"] = False

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
    _install_torch_dynamo_graph_break_stub()
    _unwrap_torch_optimizer_dynamo_wrappers()
    from skrl import config as skrl_config
    from skrl.utils.runner.torch import Runner

    skrl_config.torch.device = env.device
    _install_native_wandb_scalar_logging()

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
