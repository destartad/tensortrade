import ray
import numpy as np
import json
from ray import tune
from ray.tune.registry import register_env

import pandas as pd
import ta

from tensortrade.feed.core import Stream, DataFeed, NameSpace
from tensortrade.oms.instruments import *
from tensortrade.oms.wallets import Wallet, Portfolio, Position
from tensortrade.oms.exchanges import Exchange,ExchangeOptions
from tensortrade.oms.instruments.exchange_pair import ExchangePair

from tensortrade.oms.services.execution.simulated_MT4 import execute_order

##############
#Manipulate data: 
#1. Convert index into datetime
#2. TO-DO: normalization?
#################

def load_csv(filename):
    df = pd.read_csv(filename)
    df['open_time']=pd.to_datetime(df['open_time'])
    df.sort_values(by='open_time', ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['open_time'] = df['open_time'].dt.strftime('%Y-%m-%d %H:%M:%S %p')
    return df

def create_env(config):
    minute_EURUSD = load_csv('y.csv')

    #minute_EURUSD_ta = ta.add_all_ta_features(minute_EURUSD, 'open', 'high', 'low', 'close', 'volume', fillna=True)

    price_history = minute_EURUSD[['open_time','open','high','low','close','volume']]
    #display(price_history.head(3))

    minute_EURUSD.drop(columns=['open_time'], inplace=True)


    #################
    #Create Env
    #################


    simYunHe_options = ExchangeOptions(trading_instruments=[EURUSD])

    simYunHe = Exchange("simYunhe", service=execute_order, options=simYunHe_options)(
        Stream.source(price_history['close'].tolist(), dtype="float").rename("USD-EURUSD"),
        #Stream.source(price_history['close'].tolist(), dtype="float").rename("USD-USDJPY")
    )
    ################
    #ta.add_all_ta_features(
    #    minute_EURUSD,
    #    **{k: k for k in ['open','high','low','close','volume']}
    #)
    ##################

    minute_EURUSD_streams = [
        Stream.source(list(minute_EURUSD[c]), dtype="float").rename(c) for c in minute_EURUSD.columns
    ]

    feed = DataFeed(minute_EURUSD_streams)
    #feed.next()

    #############
    #Portfolio
    ###############

    portfolio = Portfolio(USD, [
        Wallet(simYunHe, 10000 * USD),
        ])

    #available_exchange_pairs= [ExchangePair(simYunHe, EURUSD)]

    renderer_feed = DataFeed([
        Stream.source(list(price_history["open_time"])).rename("date"),
        Stream.source(list(price_history["open"]), dtype="float").rename("open"),
        Stream.source(list(price_history["high"]), dtype="float").rename("high"),
        Stream.source(list(price_history["low"]), dtype="float").rename("low"),
        Stream.source(list(price_history["close"]), dtype="float").rename("close"), 
        Stream.source(list(price_history["volume"]), dtype="float").rename("volume") 
    ])

    ####################
    #Env
    ###################

    import tensortrade.env.MT4 as mt4

    env = mt4.create(
        #exchange=simYunHe,
        portfolio=portfolio,
        action_scheme="mt4", #TODO: override with own action;DONE
        reward_scheme="risk-adjusted", #TODO: override with own reward
        feed=feed,
        min_periods=60,#warmup obs
        window_size=60,
        renderer_feed=renderer_feed,
        renderer="screen-log"
        )
    return env

#myEnv = create_env()
register_env("TradingEnv", create_env)

"""To enable object spilling to remote storage (any URI supported by smart_open):

ray.init(
    _system_config={
        "automatic_object_spilling_enabled": True,
        "max_io_workers": 4,  # More IO workers for remote storage.
        "min_spilling_size": 100 * 1024 * 1024,  # Spill at least 100MB at a time.
        "object_spilling_config": json.dumps(
            {"type": "smart_open", "params": {"uri": "s3:///bucket/path"}},
        )
    },
)
"""

ray.init(
    _system_config={
        "automatic_object_spilling_enabled": True,
        "object_spilling_config": json.dumps(
            {"type": "filesystem", "params": {"directory_path": "/tmp/spill"}},
        )
    },
)

"""
analysis = tune.run(
    "PPO",
    stop={
      "episode_reward_mean": 500
    },
    config={
        "env": "TradingEnv",
        "env_config": {
            "window_size": 60
        },
        "log_level": "DEBUG",
        "framework": "tfe",
        "ignore_worker_failures": True,
        "num_workers": 2,
        "num_gpus": 0,
        "clip_rewards": True,
        "lr": 8e-6,
        "lr_schedule": [
            [0, 1e-1],
            [int(1e2), 1e-2],
            [int(1e3), 1e-3],
            [int(1e4), 1e-4],
            [int(1e5), 1e-5],
            [int(1e6), 1e-6],
            [int(1e7), 1e-7]
        ],
        "gamma": 0,
        "observation_filter": "MeanStdFilter",
        "lambda": 0.72,
        "vf_loss_coeff": 0.5,
        "entropy_coeff": 0.01
    },
    checkpoint_at_end=True
)
"""

import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.logger import pretty_print

config = ppo.DEFAULT_CONFIG.copy()
config["num_gpus"] = 0
config["num_workers"] = 1
config["framework"] = "tfe"
trainer = ppo.PPOTrainer(config=config, env="TradingEnv")

# Can optionally call trainer.restore(path) to load a checkpoint.

for i in range(1000):
   # Perform one iteration of training the policy with PPO
   result = trainer.train()
   print(pretty_print(result))

   if i % 100 == 0:
       checkpoint = trainer.save()
       print("checkpoint saved at", checkpoint)

# Also, in case you have trained a model outside of ray/RLlib and have created
# an h5-file with weight values in it, e.g.
# my_keras_model_trained_outside_rllib.save_weights("model.h5")
# (see: https://keras.io/models/about-keras-models/)

# ... you can load the h5-weights into your Trainer's Policy's ModelV2
# (tf or torch) by doing:
#trainer.import_model("my_weights.h5")
# NOTE: In order for this to work, your (custom) model needs to implement
# the `import_from_h5` method.
# See https://github.com/ray-project/ray/blob/master/rllib/tests/test_model_imports.py
# for detailed examples for tf- and torch trainers/models.