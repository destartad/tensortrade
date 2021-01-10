#%% set up env
import pandas as pd
import ta

from tensortrade.feed.core import Stream, DataFeed, NameSpace
from tensortrade.oms.instruments import USD, EUR
from tensortrade.oms.wallets import Wallet, Portfolio
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.services.execution.simulated import execute_order

##############
#Manipulate data: 
#1. Convert index into datetime
#2. TO-DO: normalization?
#################

def load_csv(filename):
    df = pd.read_csv(filename)
#minute_EURUSD = minute_EURUSD.set_index('open_time')
#minute_EURUSD.index = pd.to_datetime(minute_EURUSD.index)
    df['open_time']=pd.to_datetime(df['open_time'])
    df.sort_values(by='open_time', ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['open_time'] = df['open_time'].dt.strftime('%Y-%m-%d %H:%M:%S %p')
    #df=df.add_prefix("EUR:")
    return df
#minute_EURUSD[['open','high','low','close','volume']]=minute_EURUSD[['open','high','low','close','volume']].apply(pd.to_numeric)


minute_EURUSD = load_csv('y.csv')

#minute_EURUSD_ta = ta.add_all_ta_features(minute_EURUSD, 'open', 'high', 'low', 'close', 'volume', fillna=True)

price_history = minute_EURUSD[['open_time','open','high','low','close','volume']]
#display(price_history.head(3))

minute_EURUSD.drop(columns=['open_time'], inplace=True)


#################
#Create Env
#################


simYunHe = Exchange("simYunhe", service=execute_order)(
    Stream.source(price_history['close'].tolist(), dtype="float").rename("USD-EUR")
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
    Wallet(simYunHe, 0 * EUR)
    ])

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

import tensortrade.env.default as default

env = default.create(
    portfolio=portfolio,
    action_scheme="simple", #TODO: override with own action
    reward_scheme="simple", #TODO: override with own reward
    feed=feed,
    window_size=15,
    renderer_feed=renderer_feed,
    renderer=default.renderers.PlotlyTradingChart(),  
    )


#%%run agent
from tensortrade.agents import DQNAgent


agent = DQNAgent(env)
agent.train(n_steps=200, n_episodes=2, render_interval=10)

'''
from tensorforce.agents import Agent

agent_spec = {
    "type": "ppo_agent",
    "step_optimizer":{
        "type":"adam",
        "learning_rate": 1e-4
    },
    "discount": 0.99,
    "likelihood_ratio_clipping": 0.2,
}

network_spec = [
    dict(type='dense', size=64, activation="tanh"),
    dict(type='dense', size=32, activation="tanh")
]

agent = Agent.create(spec=agent_spec,
                        kwargs=dict(
                            network=network_spec,
                            states=env.observer.feed,
                            actions=env.actions
                        ))
'''

# %%
