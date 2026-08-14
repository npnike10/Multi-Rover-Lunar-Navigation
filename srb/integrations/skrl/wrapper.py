from functools import cached_property
from typing import Any, Mapping, Sequence, Tuple

import gymnasium
import torch
from skrl.envs.wrappers.torch import IsaacLabMultiAgentWrapper, IsaacLabWrapper
from skrl.utils.spaces.torch import (
    flatten_tensorized_space,
    tensorize_space,
    unflatten_tensorized_space,
)


class SingleAgentMarlAdapter(gymnasium.Wrapper):
    """Expose a one-agent DirectMARLEnv through Gymnasium's single-agent API.

    A MARL task remains dictionary-based even when it has exactly one possible
    agent.  SKRL's PPO_RNN is a single-agent algorithm, so this adapter unwraps
    that one dictionary entry while preserving vectorized environments.
    """

    def __init__(self, env: Any) -> None:
        possible_agents = list(env.unwrapped.possible_agents)
        if len(possible_agents) != 1:
            raise ValueError(
                "SingleAgentMarlAdapter requires exactly one possible agent. "
                f"Received: {possible_agents}."
            )

        super().__init__(env)
        self.agent_id = possible_agents[0]
        self.action_space = env.unwrapped.action_spaces[self.agent_id]
        self.observation_space = env.unwrapped.observation_spaces[self.agent_id]
        self.state_space = env.unwrapped.state_space

    def reset(self, **kwargs) -> tuple[torch.Tensor, Any]:
        observations, info = self.env.reset(**kwargs)
        return observations[self.agent_id], info

    def step(
        self, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        observations, rewards, terminated, truncated, info = self.env.step(
            {self.agent_id: action}
        )
        return (
            observations[self.agent_id],
            rewards[self.agent_id],
            terminated[self.agent_id],
            truncated[self.agent_id],
            info,
        )


class SkrlEnvWrapper(IsaacLabWrapper):
    def __init__(
        self,
        env: Any,
        obs_keys: Sequence[str] = [],
        state_keys: Sequence[str] | None = None,
    ) -> None:
        super().__init__(env)
        self._obs_keys = obs_keys
        self._state_keys = state_keys

        self._clip_actions_min = torch.tensor(
            self.action_space.low,  # type: ignore
            device=self.device,
            dtype=torch.float32,
        )
        self._clip_actions_max = torch.tensor(
            self.action_space.high,  # type: ignore
            device=self.device,
            dtype=torch.float32,
        )
        self._warned_nonfinite_actions = False

    @cached_property
    def action_space(self) -> gymnasium.Space:
        if isinstance(self._env, SingleAgentMarlAdapter):
            action_space = self._env.action_space
            return gymnasium.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=action_space.shape,
                dtype=action_space.dtype,
            )
        return gymnasium.spaces.Box(
            low=-1.0, high=1.0, shape=super().action_space.shape
        )

    @cached_property
    def observation_space(self) -> gymnasium.Space:
        # A DirectMARLEnv exposes ``observation_space(agent)`` as a method.
        # For the one-rover PPO path, retain the Box declared by the adapter
        # instead of following ``.unwrapped`` back to that method.
        if isinstance(self._env, SingleAgentMarlAdapter):
            obs_space = self._env.observation_space
        elif hasattr(self._unwrapped, "single_observation_space"):
            obs_space = self._unwrapped.single_observation_space
        else:
            obs_space = self._unwrapped.observation_space

        if self._obs_keys:
            return gymnasium.spaces.Dict(
                {key: obs_space[key] for key in self._obs_keys}
            )
        else:
            return obs_space

    @cached_property
    def state_space(self) -> gymnasium.Space | None:
        """State space"""
        if isinstance(self._env, SingleAgentMarlAdapter):
            return self._env.state_space
        if hasattr(self._unwrapped, "state_space"):
            return self._unwrapped.state_space

        if hasattr(self._unwrapped, "single_observation_space"):
            obs_space = self._unwrapped.single_observation_space
        else:
            obs_space = self._unwrapped.observation_space

        if self._state_keys is None:
            return None
        elif self._state_keys:
            return gymnasium.spaces.Dict(
                {key: obs_space[key] for key in self._state_keys}
            )
        else:
            return obs_space

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        if not torch.isfinite(actions).all():
            if not self._warned_nonfinite_actions:
                from srb.utils import logging

                logging.warning("Non-finite skrl actions detected; replacing with bounded values.")
                self._warned_nonfinite_actions = True
            actions = torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0)
        actions = torch.clamp(
            actions, min=self._clip_actions_min, max=self._clip_actions_max
        )
        actions = unflatten_tensorized_space(self.action_space, actions)
        observations, reward, terminated, truncated, self._info = self._env.step(
            actions
        )
        self._observations = flatten_tensorized_space(
            tensorize_space(
                self.observation_space, self.__extract_observations(observations)
            )
        )
        return (
            self._observations,
            reward.view(-1, 1),
            terminated.view(-1, 1),
            truncated.view(-1, 1),
            self._info,
        )

    def reset(self) -> Tuple[torch.Tensor, Any]:
        if self._reset_once:
            observations, self._info = self._env.reset()
            self._observations = flatten_tensorized_space(
                tensorize_space(
                    self.observation_space, self.__extract_observations(observations)
                )
            )
            self._reset_once = False
        return self._observations, self._info

    def __extract_observations(
        self, observations: Mapping[str, torch.Tensor]
    ) -> Mapping[str, torch.Tensor] | torch.Tensor:
        if not self._obs_keys:
            return observations
        return {key: observations[key] for key in self._obs_keys}


class SkrlMultiAgentEnvWrapper(IsaacLabMultiAgentWrapper):
    def __init__(self, env: Any) -> None:
        super().__init__(env)
        self._warned_nonfinite_actions: set[str] = set()

        self._clip_actions_min = {
            agent: torch.tensor(
                space.low,
                device=self.device,
                dtype=torch.float32,
            )
            for agent, space in self.action_spaces.items()
            if isinstance(space, gymnasium.spaces.Box)
        }
        self._clip_actions_max = {
            agent: torch.tensor(
                space.high,
                device=self.device,
                dtype=torch.float32,
            )
            for agent, space in self.action_spaces.items()
            if isinstance(space, gymnasium.spaces.Box)
        }

    @cached_property
    def action_spaces(self) -> Mapping[str, gymnasium.Space]:
        return {
            agent: (
                gymnasium.spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=space.shape,
                    dtype=space.dtype,
                )
                if isinstance(space, gymnasium.spaces.Box)
                else space
            )
            for agent, space in super().action_spaces.items()
        }

    def step(
        self, actions: Mapping[str, torch.Tensor]
    ) -> Tuple[
        Mapping[str, torch.Tensor],
        Mapping[str, torch.Tensor],
        Mapping[str, torch.Tensor],
        Mapping[str, torch.Tensor],
        Any,
    ]:
        bounded_actions = {}
        for agent, action in actions.items():
            if not torch.isfinite(action).all():
                if agent not in self._warned_nonfinite_actions:
                    from srb.utils import logging

                    logging.warning(
                        f"Non-finite skrl actions detected for agent '{agent}'; replacing with bounded values."
                    )
                    self._warned_nonfinite_actions.add(agent)
                action = torch.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
            if agent in self._clip_actions_min:
                action = torch.clamp(
                    action,
                    min=self._clip_actions_min[agent],
                    max=self._clip_actions_max[agent],
                )
            bounded_actions[agent] = action
        actions = bounded_actions
        return super().step(actions)
