from common import (
    MLP,
    Params,
    PRNGKey,
    default_init,
    Dtype,
    MultiModel,
    LoRAMultiDense,
    LoRAMulti_MLP,
)
import functools
from typing import Optional, Sequence, Tuple, Callable, Dict

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors


LOG_STD_MIN = -10.0
LOG_STD_MAX = 2.0


class NormalTanhPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    state_dependent_std: bool = True
    dropout_rate: Optional[float] = -1
    log_std_scale: float = 1.0
    log_std_min: Optional[float] = None
    log_std_max: Optional[float] = None
    tanh_squash_distribution: bool = True
    softplus: bool = True
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    dtype: Optional[Dtype] = jnp.float32
    name: str = "actor"

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        temperature: float = 1.0,
        training: bool = False,
        with_logstd: bool = False,
        deterministic: bool = False,
    ) -> tfd.Distribution:
        outputs = MLP(
            self.hidden_dims,
            activations=self.activations,
            activate_final=True,
            dropout_rate=self.dropout_rate,
            dtype=self.dtype,
        )(observations, training=training)

        means = nn.Dense(
            self.action_dim, dtype=self.dtype, kernel_init=default_init(), name="means"
        )(outputs)

        if self.state_dependent_std:
            log_stds = nn.Dense(
                self.action_dim,
                dtype=self.dtype,
                kernel_init=default_init(self.log_std_scale),
                name="log_stds",
            )(outputs)
        else:
            log_stds = self.param("log_stds", nn.initializers.zeros, (self.action_dim,))
            log_stds = jnp.asarray(log_stds, dtype=self.dtype)
        if self.softplus:
            scale = nn.softplus(log_stds) + 1e-5
            log_stds = jnp.log(scale)
        else:
            log_std_min = self.log_std_min or LOG_STD_MIN
            log_std_max = self.log_std_max or LOG_STD_MAX
            log_stds = jnp.clip(log_stds, log_std_min, log_std_max)
            scale = jnp.exp(log_stds)

        if not self.tanh_squash_distribution:
            means = nn.tanh(means)

        base_dist = tfd.MultivariateNormalDiag(
            loc=means, scale_diag=scale * temperature
        )

        if self.tanh_squash_distribution:
            if deterministic:
                if with_logstd:
                    return means, log_stds
                else:
                    return nn.tanh(means)
            else:
                dist = tfd.TransformedDistribution(
                    distribution=base_dist, bijector=tfb.Tanh()
                )
                if with_logstd:
                    return dist, log_stds
                else:
                    return dist
        else:
            if deterministic:
                if with_logstd:
                    return means, log_stds
                else:
                    return means
            else:
                if with_logstd:
                    return base_dist, log_stds
                else:
                    return base_dist


class LoRANormalTanhPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    rank: int
    state_dependent_std: bool = True
    dropout_rate: Optional[float] = -1
    log_std_scale: float = 1.0
    log_std_min: Optional[float] = None
    log_std_max: Optional[float] = None
    tanh_squash_distribution: bool = True
    softplus: bool = True
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    alpha: float = 1.0
    n_task: int = 1
    base_params: Optional[Dict] = None
    dtype: Optional[Dtype] = jnp.float32
    name: str = "actor"

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        temperature: float = 1.0,
        training: bool = False,
        with_logstd: bool = False,
        deterministic: bool = False,
    ) -> tfd.Distribution:
        outputs = LoRAMulti_MLP(
            self.hidden_dims,
            rank=self.rank,
            alpha=self.alpha,
            n_task=self.n_task,
            activations=self.activations,
            activate_final=True,
            base_params=self.base_params["MLP"],
            dropout_rate=self.dropout_rate,
        )(observations, training=training)

        means = LoRAMultiDense(
            self.action_dim,
            self.rank,
            self.alpha,
            self.n_task,
            self.base_params["means"]["kernel"],
            self.base_params["means"]["bias"],
            name=f"lora_means",
        )(outputs)

        if self.state_dependent_std:
            log_stds = LoRAMultiDense(
                self.action_dim,
                self.rank,
                self.alpha,
                self.n_task,
                self.base_params["log_stds"]["kernel"],
                self.base_params["log_stds"]["bias"],
                name=f"lora_log_stds",
            )(outputs)
        else:
            log_stds = self.param("log_stds", nn.initializers.zeros, (self.action_dim,))
            log_stds = jnp.asarray(log_stds, dtype=self.dtype)

        if self.softplus:
            scale = nn.softplus(log_stds) + 1e-5
            log_stds = jnp.log(scale)
        else:
            log_std_min = self.log_std_min or LOG_STD_MIN
            log_std_max = self.log_std_max or LOG_STD_MAX
            log_stds = jnp.clip(log_stds, log_std_min, log_std_max)
            scale = jnp.exp(log_stds)

        if not self.tanh_squash_distribution:
            means = nn.tanh(means)

        base_dist = tfd.MultivariateNormalDiag(
            loc=means, scale_diag=scale * temperature
        )

        if self.tanh_squash_distribution:
            if deterministic:
                if with_logstd:
                    return means, log_stds
                else:
                    return nn.tanh(means)
            else:
                dist = tfd.TransformedDistribution(
                    distribution=base_dist, bijector=tfb.Tanh()
                )
                if with_logstd:
                    return dist, log_stds
                else:
                    return dist
        else:
            if deterministic:
                if with_logstd:
                    return means, log_stds
                else:
                    return means
            else:
                if with_logstd:
                    return base_dist, log_stds
                else:
                    return base_dist


@functools.partial(jax.jit, static_argnames=("actor_def", "distribution"))
def _sample_actions(
    rng: PRNGKey,
    actor_def: nn.Module,
    actor_params: Params,
    observations: np.ndarray,
    temperature: float = 1.0,
    distribution: str = "log_prob",
) -> Tuple[PRNGKey, jnp.ndarray]:
    if distribution == "det":
        return rng, actor_def.apply(
            {"params": actor_params},
            observations,
            temperature,
            deterministic=(distribution == "det"),
        )
    else:
        dist = actor_def.apply({"params": actor_params}, observations, temperature)
        rng, key = jax.random.split(rng)
        return rng, dist.sample(seed=key)


def sample_actions(
    rng: PRNGKey,
    actor_def: nn.Module,
    actor_params: Params,
    observations: np.ndarray,
    temperature: float = 1.0,
    distribution: str = "log_prob",
) -> Tuple[PRNGKey, jnp.ndarray]:
    return _sample_actions(
        rng, actor_def, actor_params, observations, temperature, distribution
    )


@functools.partial(jax.jit, static_argnames=("distribution"))
def _sample_actions2(
    rng: PRNGKey,
    actor: MultiModel,
    actor_params: Params,
    observations: np.ndarray,
    temperature: float = 1.0,
    distribution: str = "log_prob",
) -> Tuple[PRNGKey, jnp.ndarray]:
    if distribution == "det":
        return rng, actor.apply(
            {"params": actor_params},
            observations,
            temperature,
            deterministic=(distribution == "det"),
        )
    else:
        dist = actor.apply({"params": actor_params}, observations, temperature)
        rng, key = jax.random.split(rng)
        return rng, dist.sample(seed=key)


def sample_actions2(
    rng: PRNGKey,
    actor_def: nn.Module,
    actor_params: Params,
    observations: np.ndarray,
    temperature: float = 1.0,
    distribution: str = "log_prob",
) -> Tuple[PRNGKey, jnp.ndarray]:
    return _sample_actions2(
        rng, actor_def, actor_params, observations, temperature, distribution
    )
