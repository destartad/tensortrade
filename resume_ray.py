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

# %% resume
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
import numpy as np
import json
import pandas as pd
import ta

ray.init(
    _system_config={
        "automatic_object_spilling_enabled": True,
        "object_spilling_config": json.dumps(
            {"type": "filesystem", "params": {"directory_path": "C:\\Users\\xianli\\Desktop\\Trade\\tmp"}},
        )
    },
)
analysis = tune.run(
    "PPO",
    name="try_2nd",
    checkpoint_freq=2,
    #resume=True,
    restore=r"C:\Users\xianli\ray_results\PPO\PPO_TradingEnv_e7208_00000_0_2021-03-04_17-13-37\checkpoint_126",
    stop={"training_iteration": 10}, # train 5 more iterations than previous
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
    },
    mode='min'
)

checkpoints = analysis.get_trial_checkpoints_paths(
    trial=analysis.get_best_trial("episode_reward_mean"),
    metric="episode_reward_mean"
)
checkpoint_path = checkpoints[0][0]

"""
best_trial = analysis.best_trial  # Get best trial
best_config = analysis.best_config  # Get best trial's hyperparameters
best_logdir = analysis.best_logdir  # Get best trial's logdir
best_checkpoint = analysis.best_checkpoint  # Get best trial's best checkpoint
best_result = analysis.best_result  # Get best trial's last results
best_result_df = analysis.best_result_df  # Get best result as pandas dataframe
"""

#%%
"""
agent = PPOTrainer(
    env="TradingEnv",
    config={
        "env_config": {
            "window_size": 180
        },
        "framework": "tfe",
        "log_level": "DEBUG",
        "ignore_worker_failures": True,
        "num_workers": 8,
        "num_gpus": 0,
        "clip_rewards": True,
    }
)
agent.restore(checkpoint_path)
"""