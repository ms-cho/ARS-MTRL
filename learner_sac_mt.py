"""Unified multi-task (multi-head) SAC learner with optional PCGrad support."""

from typing import Optional, Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

import policy
import value_net
from actor import sac_update_actor, sac_update_actor_pcgrad
from common import (
    Batch,
    InfoDict,
    Model,
    MultiModel,
    MultiModelPCGrad,
    PRNGKey,
    activation_func,
)
from critic import sac_update_alpha, sac_update_q, sac_update_q_pcgrad


def target_update(critic: Model, target_critic: Model, tau: float) -> Model:
    new_target_params = jax.tree_map(
        lambda p, tp: p * tau + tp * (1 - tau), critic.params, target_critic.params
    )
    return target_critic.replace(params=new_target_params)


@jax.jit
def _update_jit(
    rng: PRNGKey,
    actor: Model,
    critic: Model,
    target_critic: Model,
    alpha: Model,
    batch: Batch,
    discount: float,
    tau: float,
    target_entropy: float,
) -> Tuple[PRNGKey, Model, Model, Model, Model, Model, InfoDict]:
    key, rng = jax.random.split(rng)
    new_critic, critic_info = sac_update_q(
        key, actor, critic, target_critic, alpha, batch, discount
    )

    key, rng = jax.random.split(rng)
    new_actor, actor_info = sac_update_actor(key, actor, new_critic, alpha, batch)
    new_alpha, alpha_info = sac_update_alpha(
        alpha, actor_info["log_pi"], target_entropy
    )

    new_target_critic = target_update(new_critic, target_critic, tau)

    return (
        rng,
        new_actor,
        new_critic,
        new_target_critic,
        new_alpha,
        {**critic_info, **actor_info, **alpha_info},
    )


@jax.jit
def _update_jit_pcgrad(
    rng: PRNGKey,
    actor: Model,
    critic: Model,
    target_critic: Model,
    alpha: Model,
    batch: Batch,
    discount: float,
    tau: float,
    target_entropy: float,
) -> Tuple[PRNGKey, Model, Model, Model, Model, Model, InfoDict]:
    key, rng = jax.random.split(rng)
    new_critic, critic_info = sac_update_q_pcgrad(
        key, actor, critic, target_critic, alpha, batch, discount
    )

    key, rng = jax.random.split(rng)
    new_actor, actor_info = sac_update_actor_pcgrad(
        key, actor, new_critic, alpha, batch
    )
    new_alpha, alpha_info = sac_update_alpha(
        alpha, actor_info["log_pi"], target_entropy
    )

    new_target_critic = target_update(new_critic, target_critic, tau)

    return (
        rng,
        new_actor,
        new_critic,
        new_target_critic,
        new_alpha,
        {**critic_info, **actor_info, **alpha_info},
    )


class Learner(object):
    def __init__(
        self,
        seed: int,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        hidden_dims: Sequence[int] = (400, 400, 400, 400),
        discount: float = 0.99,
        tau: float = 0.005,
        dropout_rate: Optional[float] = None,
        n_task: int = 1,
        n_critic: int = 2,
        softplus: bool = True,
        critic_layernorm: bool = False,
        critic_init_layernorm: bool = False,
        activation: str = "tanh",
        multi_head: bool = False,
        use_pcgrad: bool = False,
    ):
        """
        Multi-task SAC learner. Toggle `use_pcgrad` to enable PCGrad projection of
        per-task gradients; toggle `multi_head` to build separate policy heads.
        """

        self.tau = tau
        self.discount = discount
        self.n_task = n_task
        self.use_pcgrad = use_pcgrad

        critic_base_opt = optax.adam
        actor_base_opt = optax.adam
        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, alpha_key = jax.random.split(rng, 4)
        actor_p_key = critic_p_key = None
        if use_pcgrad:
            rng, actor_p_key, critic_p_key = jax.random.split(rng, 3)

        action_dim = actions.shape[-1]
        self.target_entropy = -float(action_dim)

        actor_cls = (
            policy.NormalTanhMulitHeadPolicy if multi_head else policy.NormalTanhPolicy
        )
        actor_kwargs = {
            "hidden_dims": hidden_dims,
            "action_dim": action_dim,
            "log_std_min": -5.0,
            "dropout_rate": dropout_rate,
            "state_dependent_std": True,
            "tanh_squash_distribution": True,
            "activations": nn.relu,
            "name": "actor",
            "softplus": softplus,
        }
        init_observations = observations
        init_actions = actions
        if multi_head:
            actor_kwargs["n_task"] = n_task
            init_observations = observations[np.newaxis]
            init_actions = actions[np.newaxis]
        actor_def = actor_cls(**actor_kwargs)

        def _create_model(model_def, inputs, tx=None, rng_key=None):
            if use_pcgrad:
                return MultiModelPCGrad.create(
                    model_def,
                    inputs=inputs,
                    tx=tx,
                    n_task=n_task,
                    rng=rng_key,
                )
            return MultiModel.create(
                model_def,
                inputs=inputs,
                tx=tx,
                n_task=n_task,
            )

        actor_optimiser = actor_base_opt(learning_rate=actor_lr)
        actor = _create_model(
            actor_def,
            inputs=[actor_key, init_observations],
            tx=actor_optimiser,
            rng_key=actor_p_key,
        )

        alpha_def = value_net.Alpha(name="alpha", init_value=jnp.ones((n_task, 1)))
        alpha = MultiModel.create(
            alpha_def,
            inputs=[alpha_key],
            tx=optax.adam(learning_rate=alpha_lr),
            n_task=n_task,
        )

        critic_def = value_net.EnsembleCritic(
            hidden_dims,
            activations=activation_func(activation),
            n_ensemble=n_critic,
            name="critic",
            layer_norm=critic_layernorm,
            init_layer_norm=critic_init_layernorm,
        )
        critic = _create_model(
            critic_def,
            inputs=[critic_key, init_observations, init_actions],
            tx=critic_base_opt(learning_rate=critic_lr),
            rng_key=critic_p_key,
        )
        target_critic = _create_model(
            critic_def,
            inputs=[critic_key, init_observations, init_actions],
            rng_key=critic_p_key,
        )
        self.actor = actor
        self.alpha = alpha
        self.critic = critic
        self.models = {"alpha": self.alpha, "actor": self.actor, "critic": self.critic}
        self.target_critic = target_critic

        self._update_fn = _update_jit_pcgrad if use_pcgrad else _update_jit
        self.rng = rng

    def sample_actions(
        self,
        observations: np.ndarray,
        temperature: float = 1.0,
        distribution: str = "log_prob",
        params=None,
    ) -> jnp.ndarray:
        actor_params = params or self.actor.params

        rng, actions = policy.sample_actions(
            self.rng,
            self.actor.apply_fn,
            actor_params,
            observations,
            temperature,
            distribution,
        )

        actions = np.squeeze(actions)
        actions = np.asarray(actions)

        self.rng = rng
        return np.clip(actions, -1, 1)

    def update(self, batch: Batch) -> InfoDict:
        new_rng, new_actor, new_critic, new_target_critic, new_alpha, info = (
            self._update_fn(
                self.rng,
                self.actor,
                self.critic,
                self.target_critic,
                self.alpha,
                batch,
                self.discount,
                self.tau,
                self.target_entropy,
            )
        )

        self.rng = new_rng
        self.actor = new_actor
        self.critic = new_critic
        self.target_critic = new_target_critic
        self.alpha = new_alpha
        self.models = {"alpha": self.alpha, "actor": self.actor, "critic": self.critic}

        return info
