#%% Set up env
import sys
from MT_env import create_env
from ray.tune.registry import register_env
register_env("TradingEnv", create_env)

# %% resume
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
import json

ray.shutdown()
#ray.services.get_node_ip_address = lambda: '127.0.0.1'
ray._private.services.address_to_ip = lambda _node_ip_address: _node_ip_address

ray.init(
    _system_config={
        "automatic_object_spilling_enabled": True,
        #"object_store_memory": 1024*1024*1500,
        "min_spilling_size": 100 * 1024 * 1024,
        "object_spilling_config": json.dumps(
            {"type": "filesystem", "params": {"directory_path": "~/downloads/tmp"}},
        )
    },
)
analysis = tune.run(
    "A2C",
    name="A2C_exp",
    checkpoint_freq=100,
    stop={
      "episode_reward_mean": 50000
    },
    mode="max",
    #restore=r"C:\Users\xianli\ray_results\A2C_exp\A2C_TradingEnv_67d34_00000_0_2021-03-10_15-15-34\checkpoint_5021\checkpoint-5021",
    #resume=True,
    config={
        "env": "TradingEnv",
        "env_config": {
            "window_size": 180,
        },        
        "model": {
            "fcnet_hiddens": [512, 512, 512],
            "fcnet_activation": "relu",            
        },
        "log_level": "DEBUG",
        "framework": "tfe",
        "ignore_worker_failures": False,
        "num_workers": 1,
        "num_gpus": 0,
        "clip_rewards": True,
    },
    checkpoint_at_end=True
)

checkpoints = analysis.get_trial_checkpoints_paths(
    trial=analysis.get_best_trial("episode_reward_mean", mode="max"),
    metric="episode_reward_mean"
)
checkpoint_path = checkpoints[0][0]