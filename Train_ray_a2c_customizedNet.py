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
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.tf.attention_net import GTrXLNet

ray.shutdown()
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

ModelCatalog.register_custom_model("attention_net", GTrXLNet)
config = dict(
            {
                "env": "TradingEnv",
                "env_config": {
                    "window_size": 180,
                    #"min_periods":180,#warmup 1 hour
                },
                "num_workers": 1,
                "num_gpus": 0,
                "entropy_coeff": 0.01,
                "vf_loss_coeff": 0.001,
                "lr": 0.0008,
                "model": {
                    "custom_model": "attention_net",
                    "max_seq_len": 60,
                    "custom_model_config": {
                        "num_transformer_units": 1,
                        "attention_dim": 180,
                        "num_heads": 1,
                        "memory_inference": 10,
                        "memory_training": 10,
                        "head_dim": 32,
                        "position_wise_mlp_dim": 32,
                    },
                },
            },
        )

stop = {
    "episode_reward_mean": 5000,
    "timesteps_total": 5000000,
}


analysis = tune.run(
    "A2C",
    name="A2C_exp_Customized_Attention",
    checkpoint_freq=50,
    stop=stop,
    mode="max",
    #restore=r"C:\Users\xianli\ray_results\A2C_exp\A2C_TradingEnv_67d34_00000_0_2021-03-10_15-15-34\checkpoint_5021\checkpoint-5021",
    #resume=True,
    config=config,
    checkpoint_at_end=True
)

checkpoints = analysis.get_trial_checkpoints_paths(
    trial=analysis.get_best_trial("episode_reward_mean", mode="max"),
    metric="episode_reward_mean"
)
checkpoint_path = checkpoints[0][0]