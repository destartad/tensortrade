import pandas as pd
import ta

from tensortrade.feed.core import Stream, DataFeed, NameSpace
from tensortrade.oms.instruments import *
from tensortrade.oms.wallets import Wallet, Portfolio, Position
from tensortrade.oms.exchanges import Exchange,ExchangeOptions
from tensortrade.oms.instruments.exchange_pair import ExchangePair

from tensortrade.oms.services.execution.simulated_MT4 import execute_order
import ray
from ray import tune
from ray.tune.registry import register_env
import tensortrade.env.MT4 as mt4

def load_csv(filename):
    df = pd.read_csv(filename)
    df['open_time']=pd.to_datetime(df['open_time'])
    df.sort_values(by='open_time', ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['open_time'] = df['open_time'].dt.strftime('%Y-%m-%d %H:%M:%S %p')
    return df

def create_env(config):
       
    minute_EURUSD = load_csv('y.csv')
    price_history = minute_EURUSD[['open_time','open','high','low','close','volume']]
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

register_env("TradingEnv", create_env)


analysis = tune.run(
    "PPO",
    stop={
      "episode_reward_mean": 500
    },
    config={
        "env": "TradingEnv",
        "env_config": {
            "window_size": 25
        },
        "log_level": "DEBUG",
        "framework": "torch",
        "ignore_worker_failures": True,
        "num_workers": 1,
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