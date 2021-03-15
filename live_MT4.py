#%% set up env
import pandas as pd
import ta

from tensortrade.feed.core import Stream, DataFeed, NameSpace
from tensortrade.oms.instruments import *
from tensortrade.oms.wallets import Wallet, Portfolio, Position
from tensortrade.oms.exchanges import Exchange,ExchangeOptions,Exchange_live_mt4
from tensortrade.oms.instruments.exchange_pair import ExchangePair

from tensortrade.oms.services.execution.live_MT4 import execute_order
from decimal import Decimal
import sys

from connector.DWX_ZeroMQ_Connector_v2_0_1_RC8_Lear import DWX_ZeroMQ_Connector
_exchange = DWX_ZeroMQ_Connector()

_exchange._DWX_ZMQ_SHUTDOWN_()


_exchange1 = DWX_ZeroMQ_Connector()

positions = _exchange1._DWX_MTX_GET_ALL_OPEN_TRADES_()
account_info = _exchange1._DWX_MTX_GET_ACCOUNT_INFO_()
print(positions)


print(account_info)
"""
exchange = Exchange_live_mt4("demo_MT4",service=execute_order) #("EURUSD")

#positions = exchange._exchange._DWX_MTX_GET_ALL_OPEN_TRADES_()

#print(positions

while not True:
    1

portfolio = Portfolio(USD, [
    Wallet(MT4, cash_deposit * USD)])


env = mt4.create(
    #exchange=simYunHe,
    portfolio=portfolio,
    action_scheme="mt4", 
    reward_scheme="Max_profit", 
    min_periods=180,#warmup 1 hour
    window_size=180, #3 hours
    renderer="empty",
    )

"""

'''
done=False
while not done:
    if exchange._price_streams['EURUSD']:
        print(exchange._price_streams['EURUSD'])
'''
"""
_live_mt4_exchange = DWX_ZeroMQ_Connector()

account_info = _live_mt4_exchange._DWX_MTX_GET_ACCOUNT_INFO_()
"""
{'_accountBalance': 11275.27, '_accountEquity': 17869.17, '_accountFreeMargin': 15395.09, '_accountMargin': 2474.08, '_accountLeverage': 100.0, '_accountProfit': 6593.9, '_accountMarginLevel': 722.25514131}
"""

cash_deposit = account_info("_accountBalance")

open_positions = _live_mt4_exchange._DWX_MTX_GET_ALL_OPEN_TRADES_()


portfolio = Portfolio(USD, [
    Wallet(MT4, cash_deposit * USD)])

EURUSD_stream = _live_mt4_exchange._DWX_MTX_SUBSCRIBE_MARKETDATA_(_symbol="EURUSD")

def update_minute_bar():
    pass

EURUSD_minute_bar = update_minute_bar(EURUSD_stream)

open_positions = _live_mt4_exchange._DWX_MTX_GET_ALL_OPEN_TRADES_()

account_info = _live_mt4_exchange._DWX_MTX_GET_ACCOUNT_INFO_()


exchange = Exchange("demo_MT4",service=execute_order)

env = mt4.create(
    #exchange=simYunHe,
    portfolio=portfolio,
    action_scheme="mt4", 
    reward_scheme="Max_profit", 
    min_periods=180,#warmup 1 hour
    window_size=180, #3 hours
    renderer="empty",
    )


done = False
obs = env.reset()
import random

while not done:
    action = int(input("Action:"))
    obs, reward, done, info = env.step(action)
    print(reward)
    #print(shape(obs))

"""

# %%
