import collections
import os
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import functools
import flax
import flax.linen as nn
import flax.training
import jax
import jax.numpy as jnp
import optax
from flax.training import checkpoints
from flax.training import dynamic_scale as dynamic_scale_lib

Batch = collections.namedtuple(
    "Batch", ["observations", "actions", "rewards", "masks", "next_observations"]
)

default_kernel_init = nn.initializers.lecun_normal()
default_bias_init = nn.initializers.zeros


def activation_func(activation_name: str = "relu"):
    if activation_name == "swiglu":
        activation_func = SwiGLU
    else:
        try:
            activation_func = getattr(nn, activation_name)
        except AttributeError:
            raise ValueError(
                f"Activation function '{activation_name}' is not found in flax.linen."
            )
    return activation_func


def save_model(model: "Model", checkpoint_dir: str, prefix: str = "checkpoint"):
    """
    Save the model state to the specified directory.

    Args:
        model (Model): The model instance to save.
            checkpoint_dir (str): Directory where checkpoints will be saved.
            prefix (str): Prefix for checkpoint filenames.

    Example:
        save_model(
            actor,
            checkpoint_dir="./models",
        )
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoints.save_checkpoint(
        ckpt_dir=checkpoint_dir,
        target=model,
        step=model.step,
        prefix=prefix,
        overwrite=True,  # Set to False if you want to keep multiple checkpoints
    )


def load_model(
    checkpoint_dir: str,
    apply_fn: nn.Module,
    tx: Optional[optax.GradientTransformation] = None,
) -> "Model":
    """
    Load the model state from the specified directory.

    Args:
        checkpoint_dir (str): Directory from where to load checkpoints.
        apply_fn (nn.Module): The model architecture.
        tx (Optional[optax.GradientTransformation]): The optimizer used during training.

    Returns:
        Model: The loaded model instance.

    Example:
        pretrain_actor = load_model(
            apply_fn=actor_def,
            checkpoint_dir="./model",
            tx=optimiser,
        )
    """
    restored = checkpoints.restore_checkpoint(
        ckpt_dir=checkpoint_dir,
        target=None,  # We'll handle reconstruction manually
        step=None,  # Load the latest checkpoint
        prefix="checkpoint",
    )

    if restored is None:
        raise ValueError(f"No checkpoint found in directory {checkpoint_dir}")

    # Extract the saved state
    step = restored["step"]
    params = restored["params"]
    opt_state = restored["opt_state"]

    return Model(
        step=step, apply_fn=apply_fn, params=params, tx=tx, opt_state=opt_state
    )


def flat_vals(vals, path=None, n_task=None):
    if path is None:
        path = []

    values = []
    keys = []
    for key, value in vals.items():
        new_path = path + [key]
        if isinstance(value, dict):
            # If the value is a dictionary, make a recursive call
            keys_, values_ = flat_vals(value, new_path, n_task)
            values.extend(values_)
            keys.extend(keys_)
        else:
            # Otherwise, add the value to the list
            if n_task is None:
                values.append(value.reshape((-1,)))
            else:
                values.append(value.reshape((n_task, -1)))
            keys.append("/".join(new_path))
    return keys, values


def default_init(scale: Optional[float] = jnp.sqrt(2)):
    return nn.initializers.xavier_normal()


PRNGKey = Any
Params = flax.core.FrozenDict[str, Any]
Shape = Sequence[int]
Dtype = Any
InfoDict = Dict[str, float]


def SwiGLU(x):
    x, gate = jnp.split(x, 2, axis=-1)
    return nn.silu(gate) * x


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: int = False
    dropout_rate: Optional[float] = -1
    layer_norm: bool = False
    init_layer_norm: bool = False
    dtype: Optional[Dtype] = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        if self.init_layer_norm:
            x = nn.LayerNorm()(x)
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, dtype=self.dtype, kernel_init=default_init())(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
                x = self.activations(x)
                if self.dropout_rate >= 0:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training
                    )
        return x


@flax.struct.dataclass
class Model:
    step: int
    apply_fn: nn.Module = flax.struct.field(pytree_node=False)
    params: Params
    tx: Optional[optax.GradientTransformation] = flax.struct.field(pytree_node=False)
    opt_state: Optional[optax.OptState] = None
    dynamic_scale: Optional[dynamic_scale_lib.DynamicScale] = None

    @classmethod
    def create(
        cls,
        model_def: nn.Module,
        inputs: Sequence[jnp.ndarray],
        tx: Optional[optax.GradientTransformation] = None,
        dynamic_scale: Optional[dynamic_scale_lib.DynamicScale] = None,
    ) -> "Model":
        variables = model_def.init(*inputs)
        params = variables.pop("params")
        opt_state = tx.init(params) if tx else None
        return cls(
            step=1,
            apply_fn=model_def,
            params=params,
            tx=tx,
            opt_state=opt_state,
            dynamic_scale=dynamic_scale,
        )

    def __call__(self, *args, **kwargs):
        return self.apply_fn.apply({"params": self.params}, *args, **kwargs)

    def apply(self, *args, **kwargs):
        return self.apply_fn.apply(*args, **kwargs)

    def apply_gradient(self, loss_fn) -> Tuple[Any, "Model"]:
        if self.dynamic_scale:
            grad_fn = self.dynamic_scale.value_and_grad(loss_fn, has_aux=True)
            dynamic_scale, is_fin, aux, grads = grad_fn(self.params)
            info = aux[1]

            updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
            info.update({"grad/{}".format(self.apply_fn.name): grads})

            new_params = optax.apply_updates(self.params, updates)
            return (
                self.replace(
                    step=self.step + 1,
                    params=jax.tree_util.tree_map(
                        functools.partial(jnp.where, is_fin), new_params, self.params
                    ),
                    opt_state=jax.tree_util.tree_map(
                        functools.partial(jnp.where, is_fin),
                        new_opt_state,
                        self.opt_state,
                    ),
                    dynamic_scale=dynamic_scale,
                ),
                info,
            )
        else:
            grad_fn = jax.grad(loss_fn, has_aux=True)
            grads, info = grad_fn(self.params)

            updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
            info.update({"grad/{}".format(self.apply_fn.name): grads})
            new_params = optax.apply_updates(self.params, updates)

            return (
                self.replace(
                    step=self.step + 1, params=new_params, opt_state=new_opt_state
                ),
                info,
            )

    def save(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(flax.serialization.to_bytes(self.params))

    def load(self, load_path: str) -> "Model":
        with open(load_path, "rb") as f:
            params = flax.serialization.from_bytes(self.params, f.read())
        return self.replace(params=params)

    def compute_gradient(self, loss_fn) -> Any:
        return jax.grad(loss_fn, has_aux=True)(self.params)


@flax.struct.dataclass
class MultiModel:
    step: int
    n_task: int
    apply_fn: nn.Module = flax.struct.field(pytree_node=False)
    params: Params
    tx: Optional[optax.GradientTransformation] = flax.struct.field(pytree_node=False)
    opt_state: Optional[optax.OptState] = None

    @classmethod
    def create(
        cls,
        model_def: nn.Module,
        inputs: Sequence[jnp.ndarray],
        tx: Optional[optax.GradientTransformation] = None,
        n_task: int = 1,
    ) -> "Model":
        variables = model_def.init(*inputs)
        params = variables.pop("params")
        opt_state = tx.init(params) if tx else None
        return cls(
            step=1,
            apply_fn=model_def,
            params=params,
            tx=tx,
            opt_state=opt_state,
            n_task=n_task,
        )

    def __call__(self, *args, **kwargs):
        return self.apply_fn.apply({"params": self.params}, *args, **kwargs)

    def apply(self, *args, **kwargs):
        return self.apply_fn.apply(*args, **kwargs)

    def apply_gradient(self, loss_fn) -> Tuple[Any, "Model"]:
        grad_fn = jax.grad(loss_fn, has_aux=True)
        grads, info = grad_fn(self.params)

        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return (
            self.replace(
                step=self.step + 1, params=new_params, opt_state=new_opt_state
            ),
            info,
        )

    def save(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(flax.serialization.to_bytes(self.params))

    def load(self, load_path: str) -> "Model":
        with open(load_path, "rb") as f:
            params = flax.serialization.from_bytes(self.params, f.read())
        return self.replace(params=params)
