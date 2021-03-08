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
"""
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
"""
# %% resume
import ray
from ray import tune
import json
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.analysis.experiment_analysis import ExperimentAnalysis

"""
ray.init(
    _system_config={
        "automatic_object_spilling_enabled": True,
        "object_spilling_config": json.dumps(
            {"type": "filesystem", "params": {"directory_path": "C:\\Users\\xianli\\Desktop\\Trade\\tmp"}},
        )
    },
)

analysis = ExperimentAnalysis(
    experiment_checkpoint_path="~/ray_results/PPO_vannila/experiment_state-2021-03-08_11-22-27.json")

checkpoints = analysis.get_trial_checkpoints_paths(
    trial=analysis.get_best_trial("episode_reward_mean", mode="max"),
    metric="episode_reward_mean",
)
checkpoint_path = checkpoints[0][0]

"""

from ray.tune import Analysis
analysis = ExperimentAnalysis(experiment_checkpoint_path="~/ray_results/PPO_vannila/experiment_state-2021-03-08_15-10-33.json")

dfs = analysis.trial_dataframes
[d.mean_accuracy.plot() for d in dfs.values()]

# %%
