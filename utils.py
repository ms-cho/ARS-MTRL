import numpy as np
import wandb

from evaluation import evaluate_n_envs, evaluates


def log_rew_scale(step, rew_scale, env_names):
    wandb.log(
        {
            "time step": step,
            **{
                f"{env_name}/training/rew_scale": np.mean(rew_scale[env_id])
                for env_id, env_name in enumerate(env_names)
            },
        }
    )


def log_episode_info(step, infos, env_names, env_name, success):
    wandb.log(
        {
            "time step": step,
            f"{env_name}/training/average_successs": np.mean(success > 0),
        }
    )
    for env_id, info in enumerate(infos):
        for key, val in info.items():
            if key == "episode":
                val.pop("length")
                val.pop("duration")
                wandb.log(
                    {
                        "time step": step,
                        **{
                            f"{env_names[env_id]}/training/average_{k}s": v
                            for k, v in val.items()
                        },
                    }
                )
            elif key == "success":
                wandb.log(
                    {
                        "time step": step,
                        f"{env_names[env_id]}/training/average_success": int(
                            success[env_id] > 0
                        ),
                    }
                )


def log_update_metrics(step, update_info, env_names):
    for k, v in update_info.items():
        if v.ndim == 0:
            wandb.log({"time step": step, f"training/{k}": v})
        else:
            wandb.log(
                {
                    f"time step": step,
                    **{
                        f"{env_name}/training/{k}": vv
                        for env_name, vv in zip(env_names, v)
                    },
                }
            )


def run_eval(step, agent, eval_env, eval_episodes, n_envs, env_names, env_name):
    eval_stats = evaluates(
        eval_f=evaluate_n_envs,
        agent=agent,
        env=eval_env,
        num_episodes=eval_episodes,
        n_envs=n_envs,
        distribution="stc",
    )
    eval_returns = [eval_stat["return"] for eval_stat in eval_stats]
    if "success" in eval_stats[0].keys():
        eval_success = [eval_stat["success"] for eval_stat in eval_stats]
    else:
        eval_success = [0 for _ in range(n_envs)]
    wandb.log(
        {
            "time step": step,
            f"{env_name}/evaluation/average_successs": np.mean(eval_success),
        }
    )
    for env_name_item, eval_stat in zip(env_names, eval_stats):
        wandb.log(
            {
                "time step": step,
                **{
                    f"{env_name_item}/evaluation/average_{k}s": v
                    for k, v in eval_stat.items()
                    if "length" not in k
                },
            }
        )
    return eval_returns, eval_success
