"""Recurrent, parameter-shared MAPPO components for homogeneous SRB teams."""

import itertools
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from skrl import config
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.multi_agents.torch import MultiAgent
from skrl.multi_agents.torch.mappo import MAPPO


class RecurrentSharedPolicy(GaussianMixin, Model):
    """One LSTM Gaussian policy shared by all homogeneous rovers."""

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        *,
        num_agents: int,
        num_envs: int,
        hidden_size: int,
        mlp_units: Sequence[int],
        sequence_length: int,
    ):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(
            self,
            clip_actions=False,
            clip_log_std=True,
            min_log_std=-20.0,
            max_log_std=2.0,
        )
        self.num_agents = num_agents
        self.num_envs = num_envs
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.lstm = nn.LSTM(
            self.num_observations + num_agents,
            hidden_size,
            num_layers=1,
            batch_first=True,
        )
        layers: list[nn.Module] = []
        features = hidden_size
        for units in mlp_units:
            layers.extend((nn.Linear(features, units), nn.ELU()))
            features = units
        self.head = nn.Sequential(*layers)
        self.mean = nn.Linear(features, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.full((self.num_actions,), -2.0))

    def get_specification(self):
        return {
            "rnn": {
                "sequence_length": self.sequence_length,
                "sizes": [
                    (1, self.num_envs, self.hidden_size),
                    (1, self.num_envs, self.hidden_size),
                ],
            }
        }

    def _features(self, inputs: Mapping[str, Any]):
        states = inputs["states"]
        identities = inputs.get("agent_ids")
        if identities is None:
            identities = torch.zeros(
                states.shape[0], self.num_agents, dtype=states.dtype, device=states.device
            )
        states = torch.cat((states, identities), dim=-1)
        rnn_states = inputs.get("rnn")
        hidden = (rnn_states[0], rnn_states[1]) if rnn_states else None

        if rnn_states is None or states.shape[0] == self.num_envs:
            outputs, (hidden, cell) = self.lstm(states.unsqueeze(1), hidden)
            return self.head(outputs[:, 0]), {"rnn": [hidden, cell]}

        if states.shape[0] % self.sequence_length:
            raise ValueError("MAPPO_RNN received an incomplete rollout sequence.")
        batch_size = states.shape[0] // self.sequence_length
        if hidden is not None and hidden[0].shape[1] == states.shape[0]:
            hidden = (
                hidden[0][:, :: self.sequence_length].contiguous(),
                hidden[1][:, :: self.sequence_length].contiguous(),
            )
        if hidden is not None and hidden[0].shape[1] != batch_size:
            raise ValueError("MAPPO_RNN LSTM state batch does not match sampled sequences.")
        outputs, (hidden, cell) = self.lstm(
            states.reshape(batch_size, self.sequence_length, -1), hidden
        )
        return self.head(outputs.reshape(-1, self.hidden_size)), {"rnn": [hidden, cell]}

    def compute(self, inputs, role):
        features, outputs = self._features(inputs)
        return self.mean(features), self.log_std_parameter, outputs


class RecurrentSharedValue(DeterministicMixin, Model):
    """One LSTM centralized critic shared by all homogeneous rovers."""

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        *,
        num_agents: int,
        num_envs: int,
        hidden_size: int,
        mlp_units: Sequence[int],
        sequence_length: int,
    ):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)
        self.num_agents = num_agents
        self.num_envs = num_envs
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.lstm = nn.LSTM(
            self.num_observations + num_agents,
            hidden_size,
            num_layers=1,
            batch_first=True,
        )
        layers: list[nn.Module] = []
        features = hidden_size
        for units in mlp_units:
            layers.extend((nn.Linear(features, units), nn.ELU()))
            features = units
        self.head = nn.Sequential(*layers)
        self.value = nn.Linear(features, 1)

    def get_specification(self):
        return {
            "rnn": {
                "sequence_length": self.sequence_length,
                "sizes": [
                    (1, self.num_envs, self.hidden_size),
                    (1, self.num_envs, self.hidden_size),
                ],
            }
        }

    def _features(self, inputs: Mapping[str, Any]):
        states = inputs["states"]
        identities = inputs.get("agent_ids")
        if identities is None:
            identities = torch.zeros(
                states.shape[0], self.num_agents, dtype=states.dtype, device=states.device
            )
        states = torch.cat((states, identities), dim=-1)
        rnn_states = inputs.get("rnn")
        hidden = (rnn_states[0], rnn_states[1]) if rnn_states else None

        if rnn_states is None or states.shape[0] == self.num_envs:
            outputs, (hidden, cell) = self.lstm(states.unsqueeze(1), hidden)
            return self.head(outputs[:, 0]), {"rnn": [hidden, cell]}

        if states.shape[0] % self.sequence_length:
            raise ValueError("MAPPO_RNN received an incomplete rollout sequence.")
        batch_size = states.shape[0] // self.sequence_length
        if hidden is not None and hidden[0].shape[1] == states.shape[0]:
            hidden = (
                hidden[0][:, :: self.sequence_length].contiguous(),
                hidden[1][:, :: self.sequence_length].contiguous(),
            )
        if hidden is not None and hidden[0].shape[1] != batch_size:
            raise ValueError("MAPPO_RNN LSTM state batch does not match sampled sequences.")
        outputs, (hidden, cell) = self.lstm(
            states.reshape(batch_size, self.sequence_length, -1), hidden
        )
        return self.head(outputs.reshape(-1, self.hidden_size)), {"rnn": [hidden, cell]}

    def compute(self, inputs, role):
        features, outputs = self._features(inputs)
        return self.value(features), outputs


class SharedMAPPORNN(MAPPO):
    """MAPPO with recurrent CTDE models and one optimizer shared by the team."""

    def __init__(self, *args, shared_observation_spaces=None, **kwargs):
        super().__init__(
            *args, shared_observation_spaces=shared_observation_spaces, **kwargs
        )
        first = self.possible_agents[0]
        self.policy = self.policies[first]
        self.value = self.values[first]

        # Parent MAPPO creates one optimizer per UID. Replace that construction
        # with the sole optimizer required for actual cross-rover sharing.
        self.optimizer = torch.optim.Adam(
            itertools.chain(self.policy.parameters(), self.value.parameters()),
            lr=self._learning_rate[first],
        )
        self.scheduler = None
        if self._learning_rate_scheduler[first] is not None:
            self.scheduler = self._learning_rate_scheduler[first](
                self.optimizer, **self._learning_rate_scheduler_kwargs[first]
            )
        self.optimizers = {uid: self.optimizer for uid in self.possible_agents}
        self.schedulers = (
            {uid: self.scheduler for uid in self.possible_agents}
            if self.scheduler is not None
            else {}
        )

        # The scalers represent common homogeneous spaces and are shared too.
        self.state_preprocessor = self._state_preprocessor[first]
        self.shared_state_preprocessor = self._shared_state_preprocessor[first]
        self.value_preprocessor = self._value_preprocessor[first]
        self._state_preprocessor = {
            uid: self.state_preprocessor for uid in self.possible_agents
        }
        self._shared_state_preprocessor = {
            uid: self.shared_state_preprocessor for uid in self.possible_agents
        }
        self._value_preprocessor = {
            uid: self.value_preprocessor for uid in self.possible_agents
        }
        for uid in self.possible_agents:
            self.checkpoint_modules[uid]["policy"] = self.policy
            self.checkpoint_modules[uid]["value"] = self.value
            self.checkpoint_modules[uid]["optimizer"] = self.optimizer
            self.checkpoint_modules[uid]["state_preprocessor"] = self.state_preprocessor
            self.checkpoint_modules[uid]["shared_state_preprocessor"] = self.shared_state_preprocessor
            self.checkpoint_modules[uid]["value_preprocessor"] = self.value_preprocessor

    def _identity(self, uid: str, rows: int) -> torch.Tensor:
        index = self.possible_agents.index(uid)
        return F.one_hot(
            torch.full((rows,), index, device=self.device, dtype=torch.long),
            num_classes=self.num_agents,
        ).float()

    def init(self, trainer_cfg=None):
        super().init(trainer_cfg=trainer_cfg)
        self._rnn_initial_states = {}
        self._rnn_final_states = {}
        self._rnn_tensors_names = [
            "rnn_policy_0",
            "rnn_policy_1",
            "rnn_value_0",
            "rnn_value_1",
        ]
        policy_sizes = self.policy.get_specification()["rnn"]["sizes"]
        value_sizes = self.value.get_specification()["rnn"]["sizes"]
        self._rnn_sequence_length = self.policy.get_specification()["rnn"]["sequence_length"]
        for uid in self.possible_agents:
            memory = self.memories[uid]
            for index, size in enumerate(policy_sizes):
                memory.create_tensor(
                    name=f"rnn_policy_{index}",
                    size=(size[0], size[2]),
                    dtype=torch.float32,
                    keep_dimensions=True,
                )
            for index, size in enumerate(value_sizes):
                memory.create_tensor(
                    name=f"rnn_value_{index}",
                    size=(size[0], size[2]),
                    dtype=torch.float32,
                    keep_dimensions=True,
                )
            self._rnn_initial_states[uid] = {
                "policy": [torch.zeros(size, device=self.device) for size in policy_sizes],
                "value": [torch.zeros(size, device=self.device) for size in value_sizes],
            }
            self._rnn_final_states[uid] = {"policy": [], "value": []}

    def act(self, states, timestep, timesteps):
        actions, log_prob, outputs = {}, {}, {}
        with torch.autocast(device_type=torch.device(self.device).type, enabled=self._mixed_precision):
            for uid in self.possible_agents:
                action, probability, output = self.policy.act(
                    {
                        "states": self._state_preprocessor[uid](states[uid]),
                        "agent_ids": self._identity(uid, states[uid].shape[0]),
                        "rnn": self._rnn_initial_states[uid]["policy"],
                    },
                    role="policy",
                )
                actions[uid], log_prob[uid], outputs[uid] = action, probability, output
                self._rnn_final_states[uid]["policy"] = output["rnn"]
        self._current_log_prob = log_prob
        return actions, log_prob, outputs

    def record_transition(
        self, states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps
    ):
        MultiAgent.record_transition(
            self, states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps
        )
        shared_states = infos["shared_states"]
        self._current_shared_next_states = infos["shared_next_states"]
        with torch.autocast(device_type=torch.device(self.device).type, enabled=self._mixed_precision):
            for uid in self.possible_agents:
                values, _, output = self.value.act(
                    {
                        "states": self._shared_state_preprocessor[uid](shared_states),
                        "agent_ids": self._identity(uid, shared_states.shape[0]),
                        "rnn": self._rnn_initial_states[uid]["value"],
                    },
                    role="value",
                )
                values = self._value_preprocessor[uid](values, inverse=True)
                self.memories[uid].add_samples(
                    states=states[uid],
                    actions=actions[uid],
                    rewards=rewards[uid],
                    next_states=next_states[uid],
                    terminated=terminated[uid],
                    truncated=truncated[uid],
                    log_prob=self._current_log_prob[uid],
                    values=values,
                    shared_states=shared_states,
                    rnn_policy_0=self._rnn_initial_states[uid]["policy"][0].transpose(0, 1),
                    rnn_policy_1=self._rnn_initial_states[uid]["policy"][1].transpose(0, 1),
                    rnn_value_0=self._rnn_initial_states[uid]["value"][0].transpose(0, 1),
                    rnn_value_1=self._rnn_initial_states[uid]["value"][1].transpose(0, 1),
                )
                self._rnn_final_states[uid]["value"] = output["rnn"]

        for uid in self.possible_agents:
            done = (terminated[uid] | truncated[uid]).nonzero(as_tuple=False)
            if done.numel():
                for state in self._rnn_final_states[uid]["policy"] + self._rnn_final_states[uid]["value"]:
                    state[:, done[:, 0]] = 0
            self._rnn_initial_states[uid] = self._rnn_final_states[uid]

    def post_interaction(self, timestep, timesteps):
        self._rollout += 1
        if not self._rollout % self._rollouts and timestep >= self._learning_starts:
            self.set_mode("train")
            self._update(timestep, timesteps)
            self.set_mode("eval")
        MultiAgent.post_interaction(self, timestep, timesteps)

    @staticmethod
    def _gae(rewards, dones, values, last_values, gamma, lam):
        advantage = 0
        advantages = torch.zeros_like(rewards)
        for index in reversed(range(rewards.shape[0])):
            next_values = values[index + 1] if index < rewards.shape[0] - 1 else last_values
            advantage = rewards[index] - values[index] + gamma * (~dones[index]) * (next_values + lam * advantage)
            advantages[index] = advantage
        returns = advantages + values
        return returns, (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    def _update(self, timestep, timesteps):
        batches, rnn_batches = {}, {}
        with torch.no_grad():
            for uid in self.possible_agents:
                last_values, _, _ = self.value.act(
                    {
                        "states": self._shared_state_preprocessor[uid](self._current_shared_next_states.float()),
                        "agent_ids": self._identity(uid, self._current_shared_next_states.shape[0]),
                        "rnn": self._rnn_initial_states[uid]["value"],
                    },
                    role="value",
                )
                memory = self.memories[uid]
                values = memory.get_tensor_by_name("values")
                returns, advantages = self._gae(
                    memory.get_tensor_by_name("rewards"),
                    memory.get_tensor_by_name("terminated") | memory.get_tensor_by_name("truncated"),
                    values,
                    self._value_preprocessor[uid](last_values, inverse=True),
                    self._discount_factor[uid],
                    self._lambda[uid],
                )
                memory.set_tensor_by_name("values", self._value_preprocessor[uid](values, train=True))
                memory.set_tensor_by_name("returns", self._value_preprocessor[uid](returns, train=True))
                memory.set_tensor_by_name("advantages", advantages)
                batches[uid] = memory.sample_all(self._tensors_names, self._mini_batches[uid], self._rnn_sequence_length)
                rnn_batches[uid] = memory.sample_all(self._rnn_tensors_names, self._mini_batches[uid], self._rnn_sequence_length)

        policy_loss_total = value_loss_total = entropy_loss_total = 0.0
        first = self.possible_agents[0]
        for epoch in range(self._learning_epochs[first]):
            for batch_index in range(self._mini_batches[first]):
                self.optimizer.zero_grad()
                loss = 0.0
                for uid in self.possible_agents:
                    states, shared_states, actions, old_log_prob, old_values, returns, advantages = batches[uid][batch_index]
                    policy_rnn, value_rnn = rnn_batches[uid][batch_index][:2], rnn_batches[uid][batch_index][2:]
                    policy_inputs = {
                        "states": self._state_preprocessor[uid](states, train=not epoch),
                        "agent_ids": self._identity(uid, states.shape[0]),
                        "taken_actions": actions,
                        "rnn": [state.transpose(0, 1) for state in policy_rnn],
                    }
                    value_inputs = {
                        "states": self._shared_state_preprocessor[uid](shared_states, train=not epoch),
                        "agent_ids": self._identity(uid, shared_states.shape[0]),
                        "rnn": [state.transpose(0, 1) for state in value_rnn],
                    }
                    _, log_prob, _ = self.policy.act(policy_inputs, role="policy")
                    ratio = torch.exp(log_prob - old_log_prob)
                    policy_loss = -torch.min(
                        advantages * ratio,
                        advantages * torch.clip(ratio, 1 - self._ratio_clip[uid], 1 + self._ratio_clip[uid]),
                    ).mean()
                    entropy_loss = -self._entropy_loss_scale[uid] * self.policy.get_entropy(role="policy").mean()
                    predicted_values, _, _ = self.value.act(value_inputs, role="value")
                    if self._clip_predicted_values[uid]:
                        predicted_values = old_values + torch.clip(
                            predicted_values - old_values, -self._value_clip[uid], self._value_clip[uid]
                        )
                    value_loss = self._value_loss_scale[uid] * F.mse_loss(returns, predicted_values)
                    loss = loss + (policy_loss + entropy_loss + value_loss) / self.num_agents
                    policy_loss_total += policy_loss.item()
                    value_loss_total += value_loss.item()
                    entropy_loss_total += entropy_loss.item()
                self.scaler.scale(loss).backward()
                if self._grad_norm_clip[first] > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        itertools.chain(self.policy.parameters(), self.value.parameters()),
                        self._grad_norm_clip[first],
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            if self.scheduler is not None:
                self.scheduler.step()

        updates = self._learning_epochs[first] * self._mini_batches[first] * self.num_agents
        self.track_data("Loss / Policy loss", policy_loss_total / updates)
        self.track_data("Loss / Value loss", value_loss_total / updates)
        self.track_data("Loss / Entropy loss", entropy_loss_total / updates)
        self.track_data("Policy / Standard deviation", self.policy.distribution(role="policy").stddev.mean().item())
        if self.scheduler is not None:
            self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])
