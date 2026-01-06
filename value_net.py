from typing import Callable, Sequence, Optional, Dict

import jax.numpy as jnp
from flax import linen as nn

from common import MLP, Dtype, LoRAMulti_MLP, ModularGatedNet


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


class LoRACritic(nn.Module):
    hidden_dims: Sequence[int]
    rank: int
    alpha: float = 1.0
    n_task: int = 1
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    init_layer_norm: bool = False
    layer_norm: bool = False
    dropout_rate: Optional[float] = -1

    @nn.compact
    def __call__(
        self,
        base_params: Optional[Dict],
        observations: jnp.ndarray,
        actions: jnp.ndarray,
    ) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], -1)
        critic = LoRAMulti_MLP(
            (*self.hidden_dims, 1),
            rank=self.rank,
            alpha=self.alpha,
            n_task=self.n_task,
            activations=self.activations,
            base_params=base_params,
            dropout_rate=self.dropout_rate,
            layer_norm=self.layer_norm,
            init_layer_norm=self.init_layer_norm,
        )(inputs)

        return jnp.squeeze(critic, -1)


class EnsembleLoRACritic(nn.Module):
    hidden_dims: Sequence[int]
    rank: int
    scale: jnp.array
    alpha: float = 1.0
    n_task: int = 1
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    n_ensemble: int = 10
    base_params: Optional[Dict] = None
    init_layer_norm: bool = False
    layer_norm: bool = False
    dropout_rate: Optional[float] = -1
    name: str = "critic"

    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        ensemble = nn.vmap(
            target=LoRACritic,
            in_axes=(0, None, None),
            out_axes=0,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            axis_size=self.n_ensemble,
        )

        q_values = ensemble(
            hidden_dims=self.hidden_dims,
            rank=self.rank,
            alpha=self.alpha,
            n_task=self.n_task,
            activations=self.activations,
            init_layer_norm=self.init_layer_norm,
            layer_norm=self.layer_norm,
            dropout_rate=self.dropout_rate,
        )(self.base_params, observations, actions)
        q_values = q_values / jnp.reshape(self.scale, (1, self.n_task, 1))
        return q_values


class ModularGatedCritic(nn.Module):
    base_hidden_dims: Sequence[int]
    em_hidden_dims: Sequence[int]
    num_layers: int
    num_modules: int
    module_hidden: int
    gating_hidden: int
    num_gating_layers: int
    n_task: int
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        inputs = jnp.concatenate([observations[..., : -self.n_task], actions], -1)
        critic = ModularGatedNet(
            1,
            self.base_hidden_dims,
            self.em_hidden_dims,
            self.num_layers,
            self.num_modules,
            self.module_hidden,
            self.gating_hidden,
            self.num_gating_layers,
            activations=self.activations,
        )(inputs, observations[..., -self.n_task :])

        return jnp.squeeze(critic, -1)


class EnsembleModularGatedCritic(nn.Module):
    base_hidden_dims: Sequence[int]
    em_hidden_dims: Sequence[int]
    num_layers: int
    num_modules: int
    module_hidden: int
    gating_hidden: int
    num_gating_layers: int
    n_task: int
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    n_ensemble: int = 2
    name: str = "critic"

    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        ensemble = nn.vmap(
            target=ModularGatedCritic,
            in_axes=None,
            out_axes=0,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            axis_size=self.n_ensemble,
        )

        q_values = ensemble(
            self.base_hidden_dims,
            self.em_hidden_dims,
            self.num_layers,
            self.num_modules,
            self.module_hidden,
            self.gating_hidden,
            self.num_gating_layers,
            self.n_task,
            self.activations,
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
