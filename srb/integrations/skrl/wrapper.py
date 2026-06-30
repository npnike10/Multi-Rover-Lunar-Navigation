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

    @cached_property
    def action_space(self) -> gymnasium.Space:
        return gymnasium.spaces.Box(
            low=-1.0, high=1.0, shape=super().action_space.shape
        )

    @cached_property
    def observation_space(self) -> gymnasium.Space:
        if hasattr(self._unwrapped, "single_observation_space"):
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
        actions = {
            agent: torch.clamp(
                action,
                min=self._clip_actions_min[agent],
                max=self._clip_actions_max[agent],
            )
            if agent in self._clip_actions_min
            else action
            for agent, action in actions.items()
        }
        return super().step(actions)
