#%% set up env
import pandas as pd
import ta

from tensortrade.feed.core import Stream, DataFeed, NameSpace
from tensortrade.oms.instruments import *
from tensortrade.oms.wallets import Wallet, Portfolio, Position
from tensortrade.oms.exchanges import Exchange,ExchangeOptions
from tensortrade.oms.instruments.exchange_pair import ExchangePair

from tensortrade.oms.services.execution.simulated_MT4 import execute_order
from decimal import Decimal
import sys

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

minute_EURUSD = load_csv(sys.path[0] + "/tensortrade/data/EURUSD_1minute.csv")
#minute_EURUSD = load_csv('y.csv')
#minute_EURUSD_ta = ta.add_all_ta_features(minute_EURUSD, 'open', 'high', 'low', 'close', 'volume', fillna=True)


minute_EURUSD = minute_EURUSD.loc[minute_EURUSD['weekday']!=6]
price_history = minute_EURUSD[['open','high','low','close','volume']]

ta.add_all_ta_features(
    price_history,
    **{k: "" + k for k in ['open', 'high', 'low', 'close', 'volume']}
)

print(price_history)

#%%

import pandas as pd
import ta
price_history = pd.read_csv('momentum.csv')

import seaborn as sns
import matplotlib.pyplot as plt

momentum = price_history[[c for c in price_history.columns if c.startswith('momentum')]]
momentum=momentum.drop(columns='momentum_kama')
momentum = momentum.dropna(how='any')

volatility = price_history.loc[:, price_history.columns.str.contains('^volatility')]
trend = price_history.loc[:, price_history.columns.str.contains('^trend')]
others = price_history.loc[:, price_history.columns.str.contains('^others')]
volume = price_history.loc[:, price_history.columns.str.contains('^volume')]



# %%
#Correlation heatmap plot
Var_Corr = momentum.corr()
# plot the heatmap and annotation on it
sns.heatmap(Var_Corr, xticklabels=Var_Corr.columns, yticklabels=Var_Corr.columns, annot=True)
# %%
