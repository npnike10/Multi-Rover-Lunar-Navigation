import json
from collections.abc import Mapping
from dataclasses import MISSING
from typing import Any

from isaaclab.envs.utils.spaces import *  # noqa: F403
from isaaclab.envs.utils.spaces import deserialize_space as _deserialize_space


def _maybe_deserialize_space(space: Any) -> Any:
    """Deserialize Hydra-serialized spaces while preserving native specs."""
    if space is None or space is MISSING:
        return space

    if isinstance(space, str):
        try:
            decoded = json.loads(space)
        except json.JSONDecodeError:
            return space
        if isinstance(decoded, Mapping) and {"type", "space"}.issubset(decoded):
            return _deserialize_space(space)
        return space

    if isinstance(space, Mapping) and {"type", "space"}.issubset(space):
        return _deserialize_space(json.dumps(space))

    return space


def replace_strings_with_env_cfg_spaces(env_cfg: object) -> object:
    """Replace serialized env config spaces with native space specs.

    Isaac Lab's implementation assumes every field is a serialized JSON string.
    SRB's Hydra reconstruction can leave basic Python space specs, such as
    integers, already restored. Keep those values as-is so
    ``DirectRLEnv``/``DirectMARLEnv`` can build the Gymnasium spaces normally.
    """
    for attr in ["observation_space", "action_space", "state_space"]:
        if hasattr(env_cfg, attr):
            setattr(env_cfg, attr, _maybe_deserialize_space(getattr(env_cfg, attr)))

    for attr in ["observation_spaces", "action_spaces"]:
        if hasattr(env_cfg, attr):
            spaces = getattr(env_cfg, attr)
            if isinstance(spaces, Mapping):
                setattr(
                    env_cfg,
                    attr,
                    {
                        key: _maybe_deserialize_space(value)
                        for key, value in spaces.items()
                    },
                )
            else:
                setattr(env_cfg, attr, _maybe_deserialize_space(spaces))

    return env_cfg
