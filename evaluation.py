from typing import Dict

import flax.linen as nn
import gym
import numpy as np


def evaluates(
    eval_f,
    **kwargs,
) -> Dict[str, Dict[str, float]]:
    results = eval_f(**kwargs)
    return results


def evaluate(
    agent: nn.Module,
    env: gym.Env,
    num_episodes: int,
    distribution="det",
    tasks=None,
    params=None,
    random=False,
    f_ac=lambda x: x,
) -> Dict[str, float]:
    stats = {"return": [], "length": []}
    if tasks is not None:
        num_episodes = len(tasks)

    for task_id in range(num_episodes):
        if tasks is not None:
            env.set_task(tasks[task_id])
        observation, done = env.reset(), False
        success = 0
        while not done:
            if random:
                action = env.action_space.sample()
            else:
                action = agent.sample_actions(
                    np.array(observation),
                    temperature=float(distribution != "det"),
                    distribution=distribution,
                    params=params,
                )
            observation, _, done, info = env.step(f_ac(action))
            if "success" in info:
                if "success" not in stats:
                    stats.update({"success": []})
                success += info["success"]

        for k in stats.keys():
            if k == "success":
                stats[k].append(int(success > 0))
            else:
                stats[k].append(info["episode"][k])

    _stats = {}
    for k, v in stats.items():
        stats[k] = np.mean(v)
        _stats[f"{k}-max"] = np.max(v)
        _stats[f"{k}-min"] = np.min(v)
    stats.update(_stats)

    return stats


def evaluate_n_envs(
    agent: nn.Module,
    env: gym.Env,
    num_episodes: int,
    n_envs: int = 1,
    distribution="det",
    tasks=None,
    params=None,
    random=False,
    f_ac=lambda x: x,
) -> Dict[str, float]:
    stats = [{"return": [], "length": []} for _ in range(n_envs)]

    if tasks is not None:
        num_episodes = len(tasks)

    for task_id in range(num_episodes):
        if tasks is not None:
            env.set_tasks(tasks[task_id])
        observations = env.reset()
        n_tasks = len(observations) // n_envs

        dones = [False for _ in range(len(observations))]
        success = [0.0 for _ in range(len(observations))]

        while not all(dones):
            if random:
                actions = [env.action_space.sample() for _ in range(len(observations))]
            else:
                actions = agent.sample_actions(
                    np.array(observations).reshape(n_envs, n_tasks, -1),
                    temperature=float(distribution != "det"),
                    distribution=distribution,
                    params=params,
                )
            actions = np.reshape(actions, (n_tasks * n_envs, -1))
            observations, _, dones, infos = env.step(f_ac(actions))
            if "success" in infos[0]:
                if "success" not in stats[0]:
                    for stat in stats:
                        stat.update({"success": []})
                for env_id, info in enumerate(infos):
                    success[env_id] += info["success"]

        for k in stats[0].keys():
            if k == "success":
                for i, stat in enumerate(stats):
                    for j in range(n_tasks):
                        stat[k].append(int(success[n_tasks * i + j] > 0))
            else:
                for i, stat in enumerate(stats):
                    for j in range(n_tasks):
                        stat[k].append(infos[n_tasks * i + j]["episode"][k])

    for stat in stats:
        _stat = {}
        for k, v in stat.items():
            stat[k] = np.mean(v)
            _stat[f"{k}-max"] = np.max(v)
            _stat[f"{k}-min"] = np.min(v)
        stat.update(_stat)

    return stats


def evaluate_parallel_envs(
    agent: nn.Module,
    env: gym.Env,
    num_episodes: int,
    distribution="det",
    tasks=None,
    params=None,
    random=False,
    f_ac=lambda x: x,
) -> Dict[str, float]:
    stats = {"return": [], "length": []}

    if tasks is not None:
        num_episodes = len(tasks)

    for episode in range(num_episodes):
        if tasks is not None:
            env.set_tasks(tasks[episode])
        observations = env.reset()

        dones = [False for _ in range(len(observations))]
        success = [0.0 for _ in range(len(observations))]
        active_tasks = list(np.arange(len(observations)))
        while not all(dones):
            if random:
                actions = np.array(
                    [env.action_space.sample() for _ in range(len(observations))]
                )
            else:
                actions = agent.sample_actions(
                    np.array(observations),
                    temperature=float(distribution != "det"),
                    distribution=distribution,
                    params=params,
                )
            actions = np.reshape(actions, (-1, env.action_space.shape[0]))
            observations_, _, dones_, infos, active_tasks_ = env.step(f_ac(actions))
            observations = []
            for k, obs, info, done in zip(active_tasks_, observations_, infos, dones_):
                if "success" in info:
                    if "success" not in stats:
                        stats.update({"success": []})
                    success[k] += info["success"]
                dones[k] = done
                if done:
                    active_tasks.remove(k)
                    for key in stats.keys():
                        if key == "success":
                            stats[key].append(int(success[k] > 0))
                        else:
                            stats[key].append(info["episode"][key])
                else:
                    observations.append(obs)
    _stats = {}
    for k, v in stats.items():
        stats[k] = np.mean(v)
        _stats[f"{k}-max"] = np.max(v)
        _stats[f"{k}-min"] = np.min(v)
    stats.update(_stats)
    return stats
