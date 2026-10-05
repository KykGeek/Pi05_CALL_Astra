"""A read-only π0.5 feature hook implemented without editing OpenPI.

The function below is attached to the existing JAX Pi0 class at runtime. It is
the same prefix/suffix flow-matching forward used by OpenPI's
``Pi0.sample_actions``. The only additions are pooled last-layer representations
and the final action-token representation carried out of the same JAX loop.
"""

from __future__ import annotations

import einops
import jax
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.models import pi0 as _pi0


def _masked_mean(x: jax.Array, mask: jax.Array) -> jax.Array:
    mask_f = mask[..., None].astype(jnp.float32)
    x_f = x.astype(jnp.float32)
    denom = jnp.maximum(mask_f.sum(axis=1), 1.0)
    return (x_f * mask_f).sum(axis=1) / denom


def sample_actions_with_features(
    self: _pi0.Pi0,
    rng,
    observation: _model.Observation,
    *,
    num_steps: int = 10,
    noise=None,
):
    """Sample actions and return hidden features from the same forward pass.

    ``h_sem`` is a masked mean of the final PaliGemma prefix output. ``h_act``
    is a mean over the final action-token hidden states, while
    ``h_act_tokens`` retains the token-level representation. The feature
    computation is downstream-only and does not feed back into the action path.
    """

    observation = _model.preprocess_observation(None, observation, train=False)
    dt = -1.0 / num_steps
    batch_size = observation.state.shape[0]
    if noise is None:
        noise = jax.random.normal(
            rng, (batch_size, self.action_horizon, self.action_dim)
        )

    prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
    prefix_attn_mask = _pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    (prefix_out, _), kv_cache = self.PaliGemma.llm(
        [prefix_tokens, None],
        mask=prefix_attn_mask,
        positions=positions,
    )
    h_sem = _masked_mean(prefix_out, prefix_mask)

    action_width = self.action_in_proj.out_features
    h_act_init = jnp.zeros(
        (batch_size, self.action_horizon, action_width), dtype=prefix_out.dtype
    )
    flow_t_init = jnp.asarray(1.0, dtype=noise.dtype)

    def step(carry):
        x_t, time, h_act, flow_t = carry
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation, x_t, jnp.broadcast_to(time, batch_size)
        )
        suffix_attn_mask = _pi0.make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_mask = einops.repeat(
            prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1]
        )
        full_attn_mask = jnp.concatenate(
            [prefix_attn_mask, suffix_attn_mask], axis=-1
        )
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(
            suffix_mask, axis=-1
        ) - 1
        (prefix_step_out, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        del prefix_step_out
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        return (
            x_t + dt * v_t,
            time + dt,
            suffix_out[:, -self.action_horizon :],
            time,
        )

    def cond(carry):
        _, time, _, _ = carry
        return time >= -dt / 2

    x_0, _, h_act_tokens, flow_t = jax.lax.while_loop(
        cond,
        step,
        (noise, jnp.asarray(1.0, dtype=noise.dtype), h_act_init, flow_t_init),
    )
    return x_0, {
        "h_sem": h_sem,
        "h_act_tokens": h_act_tokens,
        "h_act": jnp.mean(h_act_tokens.astype(jnp.float32), axis=1),
        "flow_timestep": flow_t,
    }


def install_feature_hook() -> None:
    """Attach the hook to the already-installed OpenPI Pi0 class."""

    if not hasattr(_pi0.Pi0, "sample_actions_with_features"):
        setattr(_pi0.Pi0, "sample_actions_with_features", sample_actions_with_features)
