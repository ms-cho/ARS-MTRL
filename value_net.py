from typing import Callable, Sequence, Optional

import jax.numpy as jnp
from flax import linen as nn

from common import MLP, Dtype


class Critic(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    dropout_rate: Optional[float] = -1
    layer_norm: bool = False
    init_layer_norm: bool = False
    dtype: Optional[Dtype] = jnp.float32

    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], -1)
        critic = MLP(
            (*self.hidden_dims, 1),
            activations=self.activations,
            dropout_rate=self.dropout_rate,
            layer_norm=self.layer_norm,
            init_layer_norm=self.init_layer_norm,
            dtype=self.dtype,
        )(inputs)

        return jnp.squeeze(critic, -1)


class CriticB(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    dropout_rate: Optional[float] = -1
    layer_norm: bool = False
    dtype: Optional[Dtype] = jnp.float32
    r_max: float = 1.0
    r_min: float = 0.0
    gamma: float = 0.99

    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], -1)
        critic = MLP(
            (*self.hidden_dims, 1),
            activations=self.activations,
            dropout_rate=self.dropout_rate,
            layer_norm=self.layer_norm,
            dtype=self.dtype,
        )(inputs)
        critic = (nn.sigmoid(critic) * (self.r_max - self.r_min) + self.r_min) / (
            1 - self.gamma
        )
        return jnp.squeeze(critic, -1)


class EnsembleCritic(nn.Module):
    hidden_dim: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    n_ensemble: int = 10
    dropout_rate: Optional[float] = -1
    layer_norm: bool = False
    init_layer_norm: bool = False
    dtype: Optional[Dtype] = jnp.float32
    name: str = "critic"

    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        ensemble = nn.vmap(
            target=Critic,
            in_axes=None,
            out_axes=0,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            axis_size=self.n_ensemble,
        )

        q_values = ensemble(
            self.hidden_dim,
            self.activations,
            self.dropout_rate,
            self.layer_norm,
            self.init_layer_norm,
            self.dtype,
        )(observations, actions)
        return q_values


class EnsembleCriticB(nn.Module):
    hidden_dim: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    n_ensemble: int = 10
    dropout_rate: Optional[float] = -1
    layer_norm: bool = False
    dtype: Optional[Dtype] = jnp.float32
    name: str = "critic"
    r_max: float = 1.0
    r_min: float = 0.0
    gamma: float = 0.99

    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        ensemble = nn.vmap(
            target=CriticB,
            in_axes=None,
            out_axes=0,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            axis_size=self.n_ensemble,
        )

        q_values = ensemble(
            self.hidden_dim,
            self.activations,
            self.dropout_rate,
            self.layer_norm,
            self.dtype,
            self.r_max,
            self.r_min,
            self.gamma,
        )(observations, actions)
        return q_values


class Alpha(nn.Module):
    init_value: jnp.ndarray = jnp.ones((1, 1))
    dtype: Optional[Dtype] = jnp.float32
    name: str = "alpha"

    @nn.compact
    def __call__(self):
        log_alpha = self.param("log_alpha", lambda key: jnp.log(self.init_value))
        return jnp.asarray(jnp.exp(log_alpha), dtype=self.dtype)
