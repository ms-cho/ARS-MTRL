from typing import Tuple
import jax.numpy as jnp
from common import Batch, InfoDict, Model, Params, PRNGKey


def sac_update_q(
    key: PRNGKey,
    actor: Model,
    critic: Model,
    target_critic: Model,
    alpha: Model,
    batch: Batch,
    discount: float,
) -> Tuple[Model, InfoDict]:
    alpha_value = alpha()
    dist = actor(batch.next_observations)
    next_actions = dist.sample(seed=key)
    next_log_probs = dist.log_prob(next_actions)

    next_qs = target_critic(batch.next_observations, next_actions)
    next_q = jnp.min(next_qs, axis=0)

    target_q = batch.rewards + discount * batch.masks * (
        next_q - alpha_value * next_log_probs
    )

    def critic_loss_fn(
        critic_params: Params, task_id=None
    ) -> Tuple[jnp.ndarray, InfoDict]:
        qs = critic.apply({"params": critic_params}, batch.observations, batch.actions)
        critic_loss = jnp.sum((qs - target_q) ** 2, axis=0)
        return critic_loss[task_id].mean(), {
            "critic_loss": critic_loss.mean(axis=-1),
            "q": qs[0].mean(axis=-1),
        }

    new_critic, info = critic.apply_gradient(critic_loss_fn)

    return new_critic, info


def sac_update_alpha(
    alpha: Model, log_pi: jnp.array, target_entropy: float
) -> Tuple[Model, InfoDict]:

    def actor_loss_fn(
        alpha_params: Params, task_id=None
    ) -> Tuple[jnp.ndarray, InfoDict]:
        alpha_value = alpha.apply({"params": alpha_params})
        alpha_loss = alpha_value[:, 0] * (-log_pi - target_entropy)
        return alpha_loss[task_id].mean(), {
            "alpha_loss": alpha_loss.mean(axis=-1),
            "alpha_value": alpha_value.mean(axis=-1),
        }

    new_alpha, info = alpha.apply_gradient(actor_loss_fn)
    return new_alpha, info
