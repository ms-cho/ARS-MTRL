import os
import inspect

import uuid
import gym
import numpy as np
import tqdm
import math
import pyrallis
import wandb
from dataclasses import dataclass, asdict

import wrappers
from learner_sac_mt import Learner
import metaworld
from dataset_utils import MultiTaskReplayBuffer
import utils

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


@dataclass
class Config:
    # wandb params
    project: str = "MTRL"
    group: str = ""
    name: str = ""
    # model params
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    value_lr: float = 3e-4
    hidden_dim: int = 400
    n_layer: int = 4
    discount: float = 0.99
    tau: float = 5e-3
    n_critic: int = 2
    critic_layernorm: bool = False
    critic_init_layernorm: bool = False
    dropout_rate: float = -1.0  # -1.0 means no dropout
    activation: str = "relu"
    # training params
    domain: str = "metaworld"
    env_name: str = "MT10"
    batch_size: int = 100
    max_steps: int = int(2e6)
    log_interval: int = 1000
    n_reset: int = 4
    schedule_interval: int = int(1e3)
    opt_decay_schedule: str = ""
    replay_buffer_size: int = int(1e6)
    initial_step: int = int(25e3)
    n_train_per_step: int = 1
    # env params
    max_path_length: int = 500
    # evaluation params
    eval_episodes: int = 1
    eval_interval: int = int(1e4)
    # general params
    seed: int = 10
    eval_seed: int = 42

    def __post_init__(self):
        self.group = f"{self.domain}-{self.env_name}(H={self.max_path_length})"
        reset_interval = math.ceil(self.max_steps / (1000 * (self.n_reset + 1))) * 1000
        if reset_interval % 1e5 == 0:
            r_interval = int(reset_interval / 1e5)
        else:
            r_interval = reset_interval / 1e5
        alg_name = "ARS"
        if self.critic_layernorm and self.critic_init_layernorm:
            alg_name += "-LN-ILN"
        elif self.critic_layernorm:
            alg_name += "-LN"

        self.name = f"{alg_name}-NReset{self.n_reset}-Interval{r_interval}e5"


def make_benchmark(
    domain: str, benchmark_name: str, seed: int, max_path_length: int = 500
) -> gym.Env:
    try:
        benchmark_class = getattr(metaworld, benchmark_name)
        benchmark = benchmark_class(seed=seed)
    except TypeError:
        benchmark = benchmark_class()
    except AttributeError:
        raise NotImplementedError(f"{benchmark_name} is not implemented in {domain}.")

    envs = []
    tasks_per_env = {}
    tasks = benchmark.train_tasks
    env_names = list(benchmark.train_classes.keys())

    for env_id, env_cls in enumerate(benchmark.train_classes.values()):
        env = env_cls()
        env.set_task(tasks[env_id * 50])
        env._freeze_rand_vec = False

        env = wrappers.EpisodeMonitor(env)

        env.seed(seed + env_id + 1)
        env.action_space.seed(seed + env_id + 1)
        env.observation_space.seed(seed + env_id + 1)

        env.max_path_length = max_path_length

        envs.append(env)

        tasks_per_env[env_names[env_id]] = tasks[env_id * 50 : (env_id + 1) * 50]

    return envs, env_names, tasks_per_env


def make_eval_benchmark(
    domain: str, benchmark_name: str, seed: int, tasks, max_path_length: int = 500
) -> gym.Env:
    try:
        benchmark_class = getattr(metaworld, benchmark_name)
        benchmark = benchmark_class(seed=seed)
    except TypeError:
        benchmark = benchmark_class()
    except AttributeError:
        raise NotImplementedError(f"{benchmark_name} is not implemented in {domain}.")

    envs = []
    env_names = list(benchmark.train_classes.keys())

    for env_id, env_cls in enumerate(benchmark.train_classes.values()):
        env = env_cls()
        env.set_task(tasks[env_names[env_id]][0])

        env = wrappers.EpisodeMonitor(env)

        env.seed(seed + env_id + 1)
        env.action_space.seed(seed + env_id + 1)
        env.observation_space.seed(seed + env_id + 1)

        env.max_path_length = max_path_length

        envs.append(env)

    return envs


def main(config: Config):
    wandb.init(
        config=asdict(config),
        project=config.project,
        group=config.group,
        name=config.name,
        id=str(uuid.uuid4()),
    )

    envs, env_names, tasks = make_benchmark(
        config.domain, config.env_name, config.seed, config.max_path_length
    )
    env_names = list(env_names)
    env = wrappers.MultiParallelEnvExecutor(envs)
    action_dim = env.action_space.shape[0]
    n_envs = env.n_envs

    eval_tasks = {
        env_name: tasks_per_env[:10] for env_name, tasks_per_env in tasks.items()
    }
    eval_envs = make_eval_benchmark(
        config.domain,
        config.env_name,
        config.eval_seed,
        eval_tasks,
        config.max_path_length,
    )
    eval_env = wrappers.MultiParallelEnvExecutor(eval_envs, 10, eval_tasks)

    kwargs_ = asdict(config)
    kwargs_["hidden_dims"] = tuple([config.hidden_dim for _ in range(config.n_layer)])
    learner_args = inspect.signature(Learner).parameters.keys()

    # Filter config to only include arguments that Learner accepts
    kwargs = {k: v for k, v in kwargs_.items() if k in learner_args}
    agent = None

    replay_buffer = MultiTaskReplayBuffer(
        env.observation_space,
        action_dim,
        config.replay_buffer_size or config.max_steps,
        n_envs,
    )

    observations, dones = env.reset(), False
    rew_scale = np.ones((n_envs, 1))
    success = np.zeros(n_envs)

    reset_interval = math.ceil(config.max_steps / (1000 * (config.n_reset + 1))) * 1000
    for i in tqdm.tqdm(range(1, config.max_steps + 1), smoothing=0.1, disable=False):
        if i > config.initial_step:
            if agent is None:
                mean_rews = np.mean(
                    replay_buffer.rewards[:, : replay_buffer.size], axis=-1
                )
                rew_scale = np.reshape(
                    np.max(mean_rews) / np.maximum(mean_rews, 0.01), (n_envs, 1)
                )

                utils.log_rew_scale(i, rew_scale, env_names)
                agent = Learner(
                    observations=env.observation_space.sample()[np.newaxis],
                    actions=env.action_space.sample()[np.newaxis],
                    n_task=n_envs,
                    **kwargs,
                )
            actions = agent.sample_actions(np.array(observations), distribution="stc")
        else:
            actions = [env.action_space.sample() for _ in range(n_envs)]

        next_observations, rewards, dones, infos = env.step(actions)
        masks = []
        dones_float = []
        env_id = 0

        for done, info in zip(dones, infos):
            masks.append(int(not done or "TimeLimit.truncated" in info))
            dones_float.append(float(done))

            if "success" in info:
                success[env_id] += info["success"]
                env_id += 1

        replay_buffer.insert(
            observations, actions, rewards, masks, dones_float, next_observations
        )

        observations = next_observations

        if all(dones):
            observations, dones = env.reset(), False
            if i % config.log_interval:
                utils.log_episode_info(i, infos, env_names, config.env_name, success)
            success = np.zeros(n_envs)

        if i > config.batch_size and i > config.initial_step:
            for _ in range(config.n_train_per_step):
                batch = replay_buffer.sample(config.batch_size, rew_scale=rew_scale)
                update_info = agent.update(batch)

            if i % config.log_interval == 0:
                utils.log_update_metrics(i, update_info, env_names)

        if i % config.eval_interval == 0 and i > config.initial_step:
            utils.run_eval(
                i,
                agent,
                eval_env,
                config.eval_episodes,
                n_envs,
                env_names,
                config.env_name,
            )

        if i % reset_interval == 0 and i > config.batch_size:
            mean_rews = np.mean(replay_buffer.rewards[:, : replay_buffer.size], axis=-1)
            rew_scale = np.reshape(
                np.max(mean_rews) / np.maximum(mean_rews, 0.01), (n_envs, 1)
            )
            utils.log_rew_scale(i, rew_scale, env_names)
            agent = Learner(
                observations=env.observation_space.sample()[np.newaxis],
                actions=env.action_space.sample()[np.newaxis],
                n_task=n_envs,
                **kwargs,
            )
            for _ in range(int(1e3)):
                batch = replay_buffer.sample(config.batch_size, rew_scale=rew_scale)
                update_info = agent.update(batch)
    env.close()
    eval_env.close()
    wandb.finish()


if __name__ == "__main__":
    config = pyrallis.parse(Config)
    main()
