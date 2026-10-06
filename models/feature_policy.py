"""π0.5 policy wrapper that exposes the same action plus frozen features."""

from __future__ import annotations

import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
from openpi.policies import policy_config as _policy_config
from openpi.shared import nnx_utils
from openpi.training import config as _config

from .checkpoint_provenance import validate_official_pi05_libero_checkpoint
from .feature_hook import install_feature_hook


class Pi05FeaturePolicy:
    """Load an existing JAX π0.5 checkpoint and add a read-only feature path."""

    def __init__(
        self,
        checkpoint: str,
        *,
        seed: int = 0,
        num_steps: int = 10,
        default_prompt: str | None = None,
    ) -> None:
        checkpoint = validate_official_pi05_libero_checkpoint(checkpoint)
        install_feature_hook()
        train_config = _config.get_config("pi05_libero")
        self.policy = _policy_config.create_trained_policy(
            train_config,
            checkpoint,
            sample_kwargs={"num_steps": num_steps},
            default_prompt=default_prompt,
        )
        if getattr(self.policy, "_is_pytorch_model", False):
            raise RuntimeError("The configured π0.5 checkpoint is not the JAX backend.")
        self.model = self.policy._model
        self._sample_actions_with_features = nnx_utils.module_jit(
            self.model.sample_actions_with_features
        )
        self._rng = jax.random.key(seed)
        self.num_steps = num_steps

    def infer_with_features(
        self, obs: dict[str, Any], *, noise: np.ndarray | None = None
    ) -> dict[str, Any]:
        raw_state = np.asarray(obs["observation/state"], dtype=np.float32)
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self.policy._input_transform(inputs)
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        observation = _model.Observation.from_dict(inputs)

        self._rng, sample_rng = jax.random.split(self._rng)
        sample_kwargs = dict(self.policy._sample_kwargs)
        if noise is not None:
            noise_jax = jnp.asarray(noise)
            if noise_jax.ndim == 2:
                noise_jax = noise_jax[None, ...]
        else:
            # Match OpenPI's ordinary Policy.infer sampling exactly. Generating
            # the same keyed noise outside the extended feature JIT avoids
            # compilation-context differences in JAX's random-normal lowering.
            noise_jax = jax.random.normal(
                sample_rng,
                (
                    observation.state.shape[0],
                    int(self.model.action_horizon),
                    int(self.model.action_dim),
                ),
            )
        sample_kwargs["noise"] = noise_jax

        started = time.monotonic()
        actions, features = self._sample_actions_with_features(
            sample_rng, observation, **sample_kwargs
        )
        infer_ms = (time.monotonic() - started) * 1000.0

        raw_outputs = {
            "state": np.asarray(inputs["state"][0, ...]),
            "actions": np.asarray(actions[0, ...]),
        }
        result = self.policy._output_transform(raw_outputs)
        # Keep the raw LIBERO state consistent with normal-rollout feature
        # files, and expose the padded/normalized model input separately.
        result["state"] = raw_state
        result["policy_state"] = np.asarray(raw_outputs["state"], dtype=np.float32)

        def _remove_batch(x):
            value = np.asarray(x)
            return value[0, ...] if value.ndim > 0 and value.shape[0] == 1 else value

        result["features"] = jax.tree.map(
            _remove_batch, features
        )
        result["policy_timing"] = {"infer_ms": infer_ms}
        return result

    def infer(self, obs: dict[str, Any], *, noise: np.ndarray | None = None) -> dict[str, Any]:
        result = self.infer_with_features(obs, noise=noise)
        result.pop("features", None)
        return result

    @property
    def metadata(self) -> dict[str, Any]:
        return self.policy.metadata


def compare_action_paths(
    base_policy,
    feature_policy: Pi05FeaturePolicy,
    observation: dict[str, Any],
    noise: np.ndarray,
) -> dict[str, float]:
    """Compare ordinary OpenPI inference with the feature-hook path."""

    base = base_policy.infer(observation, noise=noise)["actions"]
    hooked = feature_policy.infer_with_features(observation, noise=noise)["actions"]
    diff = np.asarray(hooked, dtype=np.float64) - np.asarray(base, dtype=np.float64)
    return {
        "max_abs_diff": float(np.max(np.abs(diff))),
        "mean_abs_diff": float(np.mean(np.abs(diff))),
        "all_equal": bool(np.array_equal(base, hooked)),
    }
