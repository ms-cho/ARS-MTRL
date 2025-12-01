from common import (
    MLP,
    Params,
    PRNGKey,
    default_init,
    Dtype,
)
import functools
from typing import Optional, Sequence, Tuple, Callable

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


@functools.partial(
    jax.jit,
    static_argnames=(
        "actor_def",
        "critic_def",
        "distribution",
        "normalize_action_gradient",
    ),
)
def _sample_actions_with_ga(
    rng: PRNGKey,
    actor_def: nn.Module,
    critic_def: nn.Module,
    actor_params: Params,
    critic_params: Params,
    observations: np.ndarray,
    temperature: float = 1.0,
    distribution: str = "log_prob",
    action_lr: float = 0.1,
    normalize_action_gradient: bool = False,
) -> Tuple[PRNGKey, jnp.ndarray]:
    if distribution == "det":
        actions = actor_def.apply(
            {"params": actor_params},
            observations,
            temperature,
            deterministic=(distribution == "det"),
        )
    else:
        dist = actor_def.apply({"params": actor_params}, observations, temperature)
        rng, key = jax.random.split(rng)
        actions = dist.sample(seed=key)
    action_grads = jax.grad(
        lambda a: critic_def.apply({"params": critic_params}, observations, a)[0].sum()
    )(actions)
    if normalize_action_gradient:
        action_grads = action_grads / jnp.linalg.norm(
            action_grads, axis=-1, keepdims=True
        )
    actions = actions + action_lr * action_grads
    return rng, actions


def sample_actions_with_ga(
    rng: PRNGKey,
    actor_def: nn.Module,
    critic_def: nn.Module,
    actor_params: Params,
    critic_params: Params,
    observations: np.ndarray,
    temperature: float = 1.0,
    distribution: str = "log_prob",
    action_lr: float = 0.1,
    normalize_action_gradient: bool = False,
) -> Tuple[PRNGKey, jnp.ndarray]:
    return _sample_actions_with_ga(
        rng,
        actor_def,
        critic_def,
        actor_params,
        critic_params,
        observations,
        temperature,
        distribution,
        action_lr,
        normalize_action_gradient,
    )


@functools.partial(
    jax.jit, static_argnames=("actor_def", "critic_def", "distribution", "n_sample")
)
def _sample_actions_with_maxQ(
    rng: PRNGKey,
    actor_def: nn.Module,
    critic_def: nn.Module,
    actor_params: Params,
    critic_params: Params,
    observations: np.ndarray,
    temperature: float = 1.0,
    distribution: str = "log_prob",
    n_sample: int = 20,
    epsilon: float = 0.1,
) -> Tuple[PRNGKey, jnp.ndarray]:
    rng, key = jax.random.split(rng)
    if distribution == "det":
        actions = actor_def.apply(
            {"params": actor_params},
            observations,
            temperature,
            deterministic=(distribution == "det"),
        )
        actions = jnp.expand_dims(actions, axis=0) + epsilon * jax.random.normal(
            key, shape=(n_sample, *(actions.shape))
        ).clip(-1.0, 1.0)
    else:
        dist = actor_def.apply({"params": actor_params}, observations, temperature)
        actions = dist.sample(seed=key, sample_shape=(n_sample,))

    Q_values = critic_def.apply(
        {"params": critic_params},
        jnp.expand_dims(observations, axis=0).repeat(n_sample, axis=0),
        actions,
    )[0]
    action_indices = jnp.argmax(Q_values, axis=0)

    if actions.ndim == 2:
        actions = actions[action_indices]
    else:
        actions = actions[action_indices, jnp.arange(actions.shape[1])]
    return rng, actions


def sample_actions_with_maxQ(
    rng: PRNGKey,
    actor_def: nn.Module,
    critic_def: nn.Module,
    actor_params: Params,
    critic_params: Params,
    observations: np.ndarray,
    temperature: float = 1.0,
    distribution: str = "log_prob",
    n_sample: int = 20,
    epsilon: float = 0.1,
) -> Tuple[PRNGKey, jnp.ndarray]:
    return _sample_actions_with_maxQ(
        rng,
        actor_def,
        critic_def,
        actor_params,
        critic_params,
        observations,
        temperature,
        distribution,
        n_sample,
        epsilon,
    )
