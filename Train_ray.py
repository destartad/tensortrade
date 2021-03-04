#%% Set up env
import numpy as np
import json
import pandas as pd
import ta
from tensortrade.feed.core import Stream, DataFeed, NameSpace
from tensortrade.oms.instruments import *
from tensortrade.oms.wallets import Wallet, Portfolio, Position
from tensortrade.oms.exchanges import Exchange,ExchangeOptions
from tensortrade.oms.instruments.exchange_pair import ExchangePair
from tensortrade.oms.services.execution.simulated_MT4 import execute_order
from decimal import Decimal
from ray.tune.registry import register_env
import sys
def create_env(config):
    def load_csv(filename):
        df = pd.read_csv(filename)
        #minute_EURUSD = minute_EURUSD.set_index('open_time')
        #minute_EURUSD.index = pd.to_datetime(minute_EURUSD.index)
        df['open_time']=pd.to_datetime(df['open_time'])
        df['weekofyear'] = df['open_time'].dt.weekofyear
        df['year'] = df['open_time'].dt.year

        df['month'] = df['open_time'].dt.month
        df['weekday'] = df['open_time'].dt.dayofweek
        df['day'] = df['open_time'].dt.day
        df['hour'] = df['open_time'].dt.hour
        df['minute'] = df['open_time'].dt.minute
        df.sort_values(by='open_time', ascending=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
        #df['open_time'] = df['open_time'].dt.strftime('%Y-%m-%d %H:%M:%S %p')
        #df=df.add_prefix("EUR:")
        return df

    minute_EURUSD = load_csv("C:\\Users\\xianli\\Desktop\\Trade\\MyProject\\tensortrade\\data\\EURUSD_1minute.csv")
    #minute_EURUSD = load_csv('y.csv')
    #minute_EURUSD_ta = ta.add_all_ta_features(minute_EURUSD, 'open', 'high', 'low', 'close', 'volume', fillna=True)

    #%%
    minute_EURUSD = minute_EURUSD.loc[minute_EURUSD['weekday']!=6]
    price_history = minute_EURUSD[['open','high','low','close','volume', 'weekofyear', 'month', 'weekday', 'day', 'hour', 'minute']]

    minute_EURUSD_streams = [
        Stream.source(list(minute_EURUSD[c]), dtype="float").rename(c) for c in minute_EURUSD.columns]

    tech_history_streams = [
        Stream.source(list(price_history[c]), dtype="float").rename(c) for c in price_history.columns]

    feed = DataFeed(tech_history_streams)

    #%%
    #################
    #Create Env
    #################


    MT4_options = ExchangeOptions(trading_instruments=[EURUSD])

    MT4 = Exchange("simYunhe", service=execute_order, options=MT4_options)(
        Stream.source(minute_EURUSD['close'].tolist(), dtype="float").rename("USD-EURUSD"),
        Stream.source(minute_EURUSD['open_time'].tolist(), dtype="TimeStamp").rename("CurrentTime"),
        #Stream.source(price_history['close'].tolist(), dtype="float").rename("USD-USDJPY")
    )

    #############
    #Portfolio
    ###############

    portfolio = Portfolio(USD, [
        Wallet(MT4, 10000 * USD)])

    renderer_feed = DataFeed([
        Stream.source(list(minute_EURUSD["open_time"])).rename("date"),
        Stream.source(list(minute_EURUSD["open"]), dtype="float").rename("open"),
        Stream.source(list(minute_EURUSD["high"]), dtype="float").rename("high"),
        Stream.source(list(minute_EURUSD["low"]), dtype="float").rename("low"),
        Stream.source(list(minute_EURUSD["close"]), dtype="float").rename("close"), 
        Stream.source(list(minute_EURUSD["volume"]), dtype="float").rename("volume") 
    ])

    ####################
    #Env
    ###################

    import tensortrade.env.MT4 as mt4

    env = mt4.create(
        #exchange=simYunHe,
        portfolio=portfolio,
        action_scheme="mt4", #TODO: override with own action;DONE
        reward_scheme="MT4", #TODO: override with own reward
        feed=feed,
        min_periods=60*3,#warmup 1 hour
        window_size=60*3, #3 hours
        renderer_feed=renderer_feed,
        renderer="matplot"
        )
    return env

register_env("TradingEnv", create_env)

#%% init ray env
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer

ray.init(
    _system_config={
        "automatic_object_spilling_enabled": True,
        "object_spilling_config": json.dumps(
            {"type": "filesystem", "params": {"directory_path": "C:\\Users\\xianli\\Desktop\\Trade\\tmp"}},
        )
    },
)

#%% fine tune hyperparameters

analysis = tune.run(
    "PPO",
    stop={
      "episode_reward_mean": 100
    },
    config={
        "env": "TradingEnv",
        "env_config": {
            "window_size": 180
        },
        "log_level": "DEBUG",
        "framework": "tfe",
        "ignore_worker_failures": False,
        "num_workers": 4,
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
