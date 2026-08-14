import copy
import sys
import types
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import gymnasium
from isaacsim.simulation_app import SimulationApp

from srb.integrations.skrl.wrapper import (
    SingleAgentMarlAdapter,
    SkrlEnvWrapper,
    SkrlMultiAgentEnvWrapper,
)
from srb.utils import logging
from srb.utils.cfg import last_file, stamp_dir
from srb.wrappers import maybe_wrap_action_smoothing

if TYPE_CHECKING:
    from srb._typing import AnyEnv, AnyEnvCfg

FRAMEWORK_NAME = "skrl"


def _install_ppo_rnn_runner_support() -> None:
    """Add PPO_RNN support missing from SKRL's generic Torch runner.

    The bundled runner supports PPO but omits its recurrent counterpart even
    though SKRL ships ``PPO_RNN``.  Keep the patch local to SRB and only change
    the PPO-RNN component and single-agent construction path.
    """
    from skrl.utils.runner.torch import Runner

    if getattr(Runner, "_srb_ppo_rnn_support", False):
        return

    original_component = Runner._component
    original_generate_models = Runner._generate_models
    original_generate_agent = Runner._generate_agent

    def component(self, name: str):
        if name.lower() in {"ppo_rnn", "ppo_rnn_default_config"}:
            from skrl.agents.torch.ppo import PPO_DEFAULT_CONFIG, PPO_RNN

            return (
                PPO_DEFAULT_CONFIG
                if name.lower().endswith("default_config")
                else PPO_RNN
            )
        return original_component(self, name)

    def generate_models(self, env, cfg):
        """Build the recurrent models that SKRL's generic runner cannot express."""
        agent_class = cfg.get("agent", {}).get("class", "").lower()
        if agent_class != "ppo_rnn":
            return original_generate_models(self, env, cfg)

        import torch
        import torch.nn as nn
        from skrl.models.torch import DeterministicMixin, GaussianMixin, Model

        models_cfg = cfg.get("models", {})
        policy_cfg = models_cfg.get("policy", {})
        value_cfg = models_cfg.get("value", {})
        hidden_size = policy_cfg.get("lstm_hidden_size", 384)
        mlp_units = policy_cfg.get("mlp_units", [384, 384])
        sequence_length = cfg["agent"]["rollouts"]

        if (
            hidden_size != value_cfg.get("lstm_hidden_size", hidden_size)
            or mlp_units != value_cfg.get("mlp_units", mlp_units)
        ):
            raise ValueError(
                "PPO_RNN requires matching policy and value LSTM/MLP dimensions."
            )

        class _RecurrentBase(Model):
            """Shared LSTM trunk with a two-layer ELU head for PPO_RNN."""

            def __init__(self, observation_space, action_space, device):
                Model.__init__(self, observation_space, action_space, device)
                self.lstm = nn.LSTM(
                    input_size=self.num_observations,
                    hidden_size=hidden_size,
                    num_layers=1,
                    batch_first=True,
                )
                layers: list[nn.Module] = []
                in_features = hidden_size
                for units in mlp_units:
                    layers.extend((nn.Linear(in_features, units), nn.ELU()))
                    in_features = units
                self.head = nn.Sequential(*layers)

            def get_specification(self):
                # PPO_RNN stores these two LSTM states in rollout memory. A full
                # rollout is one training sequence, preserving temporal credit.
                return {
                    "rnn": {
                        "sequence_length": sequence_length,
                        "sizes": [
                            (1, env.num_envs, hidden_size),
                            (1, env.num_envs, hidden_size),
                        ],
                    }
                }

            def _features(self, inputs):
                states = inputs["states"]
                rnn_states = inputs.get("rnn")
                if rnn_states:
                    hidden = (rnn_states[0], rnn_states[1])
                else:
                    hidden = None

                # During collection each row is one environment step. During
                # learning, SKRL orders each full rollout sequence contiguously.
                # ``init_state_dict`` probes a model with a one-row sample and
                # no recurrent state. Treat that, and normal rollout collection,
                # as one-step batches.
                if rnn_states is None or states.shape[0] == env.num_envs:
                    outputs, (hidden, cell) = self.lstm(states.unsqueeze(1), hidden)
                    return self.head(outputs[:, 0]), {"rnn": [hidden, cell]}

                if states.shape[0] % sequence_length:
                    raise ValueError(
                        "PPO_RNN received a batch that is not an integer number "
                        f"of {sequence_length}-step rollout sequences."
                    )
                batch_size = states.shape[0] // sequence_length
                # SKRL stores an RNN state beside every transition, then
                # returns all of them for a sampled sequence. The LSTM needs
                # only the state before each sequence's first transition.
                if hidden is not None and hidden[0].shape[1] == states.shape[0]:
                    hidden = (
                        hidden[0][:, ::sequence_length].contiguous(),
                        hidden[1][:, ::sequence_length].contiguous(),
                    )
                sequences = states.reshape(batch_size, sequence_length, -1)
                if hidden is not None and hidden[0].shape[1] != batch_size:
                    raise ValueError(
                        "PPO_RNN LSTM state batch size does not match sampled "
                        "rollout sequences."
                    )
                outputs, (hidden, cell) = self.lstm(sequences, hidden)
                return self.head(outputs.reshape(-1, hidden_size)), {"rnn": [hidden, cell]}

        class _Policy(GaussianMixin, _RecurrentBase):
            def __init__(self, observation_space, action_space, device):
                _RecurrentBase.__init__(self, observation_space, action_space, device)
                GaussianMixin.__init__(
                    self,
                    clip_actions=False,
                    clip_log_std=True,
                    min_log_std=-20.0,
                    max_log_std=2.0,
                )
                self.mean = nn.Linear(mlp_units[-1], self.num_actions)
                self.log_std_parameter = nn.Parameter(
                    torch.full((self.num_actions,), -2.0)
                )

            def compute(self, inputs, role):
                features, outputs = self._features(inputs)
                return self.mean(features), self.log_std_parameter, outputs

        class _Value(DeterministicMixin, _RecurrentBase):
            def __init__(self, observation_space, action_space, device):
                _RecurrentBase.__init__(self, observation_space, action_space, device)
                DeterministicMixin.__init__(self, clip_actions=False)
                self.value = nn.Linear(mlp_units[-1], 1)

            def compute(self, inputs, role):
                features, outputs = self._features(inputs)
                return self.value(features), outputs

        policy = _Policy(env.observation_space, env.action_space, env.device)
        value = _Value(env.observation_space, env.action_space, env.device)
        policy.init_state_dict("policy")
        value.init_state_dict("value")
        return {"agent": {"policy": policy, "value": value}}

    def generate_agent(self, env, cfg, models):
        agent_class = cfg.get("agent", {}).get("class", "").lower()
        if agent_class != "ppo_rnn":
            return original_generate_agent(self, env, cfg, models)

        from skrl.agents.torch.ppo import PPO_DEFAULT_CONFIG, PPO_RNN

        if len(models) != 1 or "agent" not in models:
            raise ValueError(
                "PPO_RNN requires a single-agent environment. Set "
                "env.num_rovers=1 for marl_waypoint_navigation."
            )

        memory_cfg = copy.deepcopy(cfg.get("memory", {}))
        memory_class = self._component(memory_cfg.pop("class", "RandomMemory"))
        if memory_cfg.get("memory_size", -1) < 0:
            memory_cfg["memory_size"] = cfg["agent"]["rollouts"]
        memory = memory_class(
            num_envs=env.num_envs,
            device=env.device,
            **self._process_cfg(memory_cfg),
        )

        # SKRL's recurrent PPO implementation deliberately shares the ordinary
        # PPO default configuration; it does not export a separate RNN dict.
        agent_cfg = PPO_DEFAULT_CONFIG.copy()
        agent_cfg.update(self._process_cfg(copy.deepcopy(cfg["agent"])))
        agent_cfg.get("state_preprocessor_kwargs", {}).update(
            {"size": env.observation_space, "device": env.device}
        )
        agent_cfg.get("value_preprocessor_kwargs", {}).update(
            {"size": 1, "device": env.device}
        )

        return PPO_RNN(
            models=models["agent"],
            memory=memory,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device,
            cfg=agent_cfg,
        )

    Runner._component = component
    Runner._generate_models = generate_models
    Runner._generate_agent = generate_agent
    Runner._srb_ppo_rnn_support = True


def _configure_ppo_rnn_hyperparameters(
    agent_cfg: dict, env: Any, workflow: Literal["train", "eval"]
) -> None:
    """Translate PPO-RNN training settings and discard them for evaluation.

    PPO minibatches and its learning-rate schedule are meaningful only while
    updating a policy. Evaluation still constructs a PPO_RNN instance, but it
    never calls ``_update`` and must support any number of parallel worlds.
    """
    config = agent_cfg.get("agent", {})
    if config.get("class", "").lower() != "ppo_rnn":
        return

    minibatch_size = config.pop("minibatch_size", None)
    if workflow == "eval":
        # Avoid evaluating the training-only scheduler sentinel in SKRL's
        # generic config processor. ``mini_batches`` is harmless because no
        # evaluation path performs a PPO update.
        config["learning_rate_scheduler"] = None
        config["learning_rate_scheduler_kwargs"] = {}
        return

    if minibatch_size is not None:
        rollout_samples = config["rollouts"] * env.num_envs
        if rollout_samples < minibatch_size or rollout_samples % minibatch_size:
            raise ValueError(
                "PPO_RNN requires rollouts * env.num_envs to be an exact multiple "
                f"of minibatch_size. Received {config['rollouts']} * {env.num_envs} "
                f"= {rollout_samples} samples and minibatch_size={minibatch_size}."
            )
        config["mini_batches"] = rollout_samples // minibatch_size

    if config.get("learning_rate_scheduler") == "linear_to_zero":
        import torch

        rollouts = config["rollouts"]
        epochs = config["learning_epochs"]
        timesteps = agent_cfg["trainer"]["timesteps"]
        scheduler_steps = ((timesteps + rollouts - 1) // rollouts) * epochs
        config["learning_rate_scheduler"] = torch.optim.lr_scheduler.LambdaLR
        config["learning_rate_scheduler_kwargs"] = {
            "lr_lambda": lambda step: max(0.0, 1.0 - step / scheduler_steps)
        }


def _configure_mappo_hyperparameters(
    agent_cfg: dict, env: Any, workflow: Literal["train", "eval"]
) -> None:
    """Translate paper-facing MAPPO settings to SKRL's per-agent fields.

    SKRL MAPPO owns an independent rollout memory for every rover. Therefore
    its minibatch size is defined per rover as ``rollouts * env.num_envs``, not
    across the combined set of rover buffers. The centralized critic state is
    supplied separately by the MAPPO runner path.
    """
    config = agent_cfg.get("agent", {})
    if config.get("class", "").lower() not in {"mappo", "mappo_rnn"}:
        return

    minibatch_size = config.pop("minibatch_size", None)
    if workflow == "eval":
        # Evaluation never updates a MAPPO policy, so batch partitioning and
        # the training-only scheduler must not constrain its environment count.
        config["learning_rate_scheduler"] = None
        config["learning_rate_scheduler_kwargs"] = {}
        return

    if minibatch_size is not None:
        rollout_samples = config["rollouts"] * env.num_envs
        if rollout_samples < minibatch_size or rollout_samples % minibatch_size:
            raise ValueError(
                "MAPPO requires rollouts * env.num_envs to be an exact multiple "
                "of minibatch_size for each rover buffer. Received "
                f"{config['rollouts']} * {env.num_envs} = {rollout_samples} "
                f"samples and minibatch_size={minibatch_size}."
            )
        config["mini_batches"] = rollout_samples // minibatch_size

    if config.get("learning_rate_scheduler") == "linear_to_zero":
        import torch

        rollouts = config["rollouts"]
        epochs = config["learning_epochs"]
        timesteps = agent_cfg["trainer"]["timesteps"]
        scheduler_steps = ((timesteps + rollouts - 1) // rollouts) * epochs
        # MAPPO's configuration values are expanded per agent by SKRL. Its
        # kwargs must therefore be keyed by rover ID as well (a bare
        # ``{"lr_lambda": ...}`` mapping is interpreted as an invalid agent map).
        possible_agents = list(env.possible_agents)
        config["learning_rate_scheduler"] = {
            uid: torch.optim.lr_scheduler.LambdaLR for uid in possible_agents
        }
        config["learning_rate_scheduler_kwargs"] = {
            uid: {"lr_lambda": lambda step: max(0.0, 1.0 - step / scheduler_steps)}
            for uid in possible_agents
        }


def _install_mappo_rnn_runner_support() -> None:
    """Add recurrent, parameter-shared MAPPO support to SKRL's Torch runner.

    The installed SKRL package has only feed-forward MAPPO and creates one
    optimizer per agent. ``MAPPO_RNN`` is an SRB extension: it shares one LSTM
    policy and one LSTM centralized critic across the homogeneous rover team,
    while providing an internal one-hot rover identity to both networks.
    """
    from skrl.utils.runner.torch import Runner

    if getattr(Runner, "_srb_mappo_rnn_support", False):
        return

    original_generate_models = Runner._generate_models
    original_generate_agent = Runner._generate_agent

    def generate_models(self, env, cfg):
        if cfg.get("agent", {}).get("class", "").lower() != "mappo_rnn":
            return original_generate_models(self, env, cfg)

        from srb.integrations.skrl.mappo_rnn import (
            RecurrentSharedPolicy,
            RecurrentSharedValue,
        )

        possible_agents = list(env.possible_agents)
        if len(possible_agents) < 2:
            raise ValueError("MAPPO_RNN requires at least two rovers.")
        observation_space = env.observation_spaces[possible_agents[0]]
        action_space = env.action_spaces[possible_agents[0]]
        state_space = env.state_spaces[possible_agents[0]]
        models_cfg = cfg.get("models", {})
        policy_cfg = models_cfg.get("policy", {})
        value_cfg = models_cfg.get("value", {})
        hidden_size = policy_cfg.get("lstm_hidden_size", 384)
        mlp_units = policy_cfg.get("mlp_units", [384, 384])
        if (
            hidden_size != value_cfg.get("lstm_hidden_size", hidden_size)
            or mlp_units != value_cfg.get("mlp_units", mlp_units)
        ):
            raise ValueError(
                "MAPPO_RNN requires matching policy and value LSTM/MLP dimensions."
            )

        policy = RecurrentSharedPolicy(
            observation_space,
            action_space,
            env.device,
            num_agents=len(possible_agents),
            num_envs=env.num_envs,
            hidden_size=hidden_size,
            mlp_units=mlp_units,
            sequence_length=cfg["agent"]["rollouts"],
        )
        value = RecurrentSharedValue(
            state_space,
            action_space,
            env.device,
            num_agents=len(possible_agents),
            num_envs=env.num_envs,
            hidden_size=hidden_size,
            mlp_units=mlp_units,
            sequence_length=cfg["agent"]["rollouts"],
        )
        policy.init_state_dict("policy")
        value.init_state_dict("value")
        return {
            uid: {"policy": policy, "value": value}
            for uid in possible_agents
        }

    def generate_agent(self, env, cfg, models):
        if cfg.get("agent", {}).get("class", "").lower() != "mappo_rnn":
            return original_generate_agent(self, env, cfg, models)

        from skrl.memories.torch import RandomMemory
        from skrl.multi_agents.torch.mappo import MAPPO_DEFAULT_CONFIG

        from srb.integrations.skrl.mappo_rnn import SharedMAPPORNN

        possible_agents = list(env.possible_agents)
        memory_cfg = copy.deepcopy(cfg.get("memory", {}))
        memory_class = self._component(memory_cfg.pop("class", "RandomMemory"))
        if memory_class is not RandomMemory:
            raise ValueError("MAPPO_RNN currently requires RandomMemory.")
        if memory_cfg.get("memory_size", -1) < 0:
            memory_cfg["memory_size"] = cfg["agent"]["rollouts"]
        memories = {
            uid: memory_class(
                num_envs=env.num_envs,
                device=env.device,
                **self._process_cfg(memory_cfg),
            )
            for uid in possible_agents
        }

        agent_cfg = MAPPO_DEFAULT_CONFIG.copy()
        agent_cfg.update(self._process_cfg(copy.deepcopy(cfg["agent"])))
        observation_spaces = env.observation_spaces
        state_spaces = env.state_spaces
        agent_cfg["state_preprocessor_kwargs"].update(
            {
                uid: {"size": observation_spaces[uid], "device": env.device}
                for uid in possible_agents
            }
        )
        agent_cfg["shared_state_preprocessor_kwargs"].update(
            {
                uid: {"size": state_spaces[uid], "device": env.device}
                for uid in possible_agents
            }
        )
        agent_cfg["value_preprocessor_kwargs"].update(
            {uid: {"size": 1, "device": env.device} for uid in possible_agents}
        )
        return SharedMAPPORNN(
            models=models,
            memories=memories,
            observation_spaces=observation_spaces,
            action_spaces=env.action_spaces,
            shared_observation_spaces=state_spaces,
            possible_agents=possible_agents,
            device=env.device,
            cfg=agent_cfg,
        )

    Runner._generate_models = generate_models
    Runner._generate_agent = generate_agent
    Runner._srb_mappo_rnn_support = True


def _configure_wandb_global_step_axis() -> None:
    """Use environment timesteps as the default W&B x-axis."""
    try:
        import wandb

        run = getattr(wandb, "run", None)
        if run is None or getattr(run, "_srb_global_step_axis_defined", False):
            return
        wandb.define_metric("global_step")
        wandb.define_metric("*", step_metric="global_step", step_sync=True)
        setattr(run, "_srb_global_step_axis_defined", True)
    except Exception as exc:
        logging.warning(f"Failed to configure W&B global_step axis: {exc}")


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

    def _wandb_tensorboard_sync_enabled(agent: Any) -> bool:
        wandb_kwargs = agent.cfg.get("experiment", {}).get("wandb_kwargs", {})
        return bool(wandb_kwargs.get("sync_tensorboard", True))

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
            use_native_wandb = _wandb_enabled(
                self
            ) and not _wandb_tensorboard_sync_enabled(self)
            payload = _tracking_payload(self) if use_native_wandb else {}
            original(self, timestep, timesteps)
            _flush_writer(self)
            if not payload:
                return

            try:
                import wandb

                if wandb.run is not None:
                    _configure_wandb_global_step_axis()
                    payload["global_step"] = int(timestep)
                    wandb.log(payload, commit=True)
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


def _install_mappo_policy_action_logging() -> None:
    """Track MAPPO policy mean and sampled actions without patching site-packages."""

    def _track_action_stats(agent: Any, uid: str, prefix: str, actions: Any) -> None:
        if (
            actions is None
            or getattr(actions, "ndim", 0) != 2
            or actions.shape[-1] < 2
        ):
            return
        agent.track_data(
            f"Policy / {prefix} linear ({uid})", actions[:, 0].mean().item()
        )
        agent.track_data(
            f"Policy / {prefix} angular ({uid})", actions[:, 1].mean().item()
        )
        agent.track_data(f"Policy / {prefix} abs ({uid})", actions.abs().mean().item())
        agent.track_data(
            f"Policy / {prefix} max abs ({uid})",
            actions.abs().max(dim=-1).values.mean().item(),
        )

    try:
        from skrl.multi_agents.torch.mappo import MAPPO

        original = getattr(MAPPO, "act")
        if getattr(original, "_srb_policy_action_logging_wrapped", False):
            return

        def act(self: Any, states: Any, timestep: int, timesteps: int):
            actions, log_prob, outputs = original(self, states, timestep, timesteps)

            try:
                for uid, action in actions.items():
                    mean_actions = outputs.get(uid, {}).get("mean_actions")
                    _track_action_stats(self, uid, "Mean action", mean_actions)
                    _track_action_stats(self, uid, "Sampled action", action)

                    policy = self.policies.get(uid)
                    distribution = (
                        policy.distribution(role="policy") if policy is not None else None
                    )
                    stddev = getattr(distribution, "stddev", None)
                    if (
                        stddev is not None
                        and getattr(stddev, "ndim", 0) == 2
                        and stddev.shape[-1] >= 2
                    ):
                        self.track_data(
                            f"Policy / Std action linear ({uid})",
                            stddev[:, 0].mean().item(),
                        )
                        self.track_data(
                            f"Policy / Std action angular ({uid})",
                            stddev[:, 1].mean().item(),
                        )
            except Exception as exc:
                if not getattr(self, "_srb_policy_action_logging_warned", False):
                    logging.warning(f"Failed to log MAPPO policy action stats: {exc}")
                    self._srb_policy_action_logging_warned = True

            return actions, log_prob, outputs

        act.__wrapped__ = original  # type: ignore[attr-defined]
        act._srb_policy_action_logging_wrapped = True  # type: ignore[attr-defined]
        setattr(MAPPO, "act", act)
    except Exception as exc:
        logging.warning(f"Failed to install MAPPO policy action logging: {exc}")


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
        # Use native W&B logging for charts. TensorBoard event files are still
        # written locally by skrl, but W&B does not depend on TensorBoard sync.
        wandb_kwargs["sync_tensorboard"] = False

    unwrapped_env = getattr(env, "unwrapped", env)
    is_multi_agent = hasattr(unwrapped_env, "possible_agents")
    single_marl_agent = is_multi_agent and len(unwrapped_env.possible_agents) == 1

    # Enable action smoothing if enabled
    if is_multi_agent and smoothing_cfg.get("enabled", False):
        logging.warning("Action smoothing is not supported for skrl multi-agent envs.")
    else:
        env = maybe_wrap_action_smoothing(
            env,  # type: ignore
            smoothing_cfg,
        )

    # Wrap the environment
    if single_marl_agent:
        env = SkrlEnvWrapper(SingleAgentMarlAdapter(env))  # type: ignore
    elif is_multi_agent:
        env = SkrlMultiAgentEnvWrapper(env)  # type: ignore
    else:
        env = SkrlEnvWrapper(env)  # type: ignore

    # Create the runner
    _install_torch_dynamo_graph_break_stub()
    _unwrap_torch_optimizer_dynamo_wrappers()
    from skrl import config as skrl_config
    from skrl.utils.runner.torch import Runner

    _install_ppo_rnn_runner_support()
    _install_mappo_rnn_runner_support()
    _configure_ppo_rnn_hyperparameters(agent_cfg, env, workflow)
    _configure_mappo_hyperparameters(agent_cfg, env, workflow)
    skrl_config.torch.device = env.device
    _install_native_wandb_scalar_logging()
    _install_mappo_policy_action_logging()

    runner = Runner(
        env,  # type: ignore
        agent_cfg,
    )
    _configure_wandb_global_step_axis()

    # Load checkpoint if needed
    if from_checkpoint:
        runner.agent.load(
            from_checkpoint,  # type: ignore
        )

    # Run the workflow
    runner.run(mode=workflow)
