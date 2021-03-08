from typing import List


import numpy as np
import pandas as pd

from gym.spaces import Box, Space


from tensortrade.feed.core import Stream, NameSpace, DataFeed
from tensortrade.oms.wallets import Wallet
from tensortrade.oms.wallets import Position
from tensortrade.env.generic import Observer
from collections import OrderedDict


def _create_wallet_source(wallet: 'Wallet', include_worth: bool = True) -> 'List[Stream[float]]':
    """Creates a list of streams to describe a `Wallet`.

    Parameters
    ----------
    wallet : `Wallet`
        The wallet to make streams for.
    include_worth : bool, default True
        Whether or

    Returns
    -------
    `List[Stream[float]]`
        A list of streams to describe the `wallet`.
    """
    exchange_name = wallet.exchange.name
    symbol = wallet.instrument.symbol

    streams = []

    with NameSpace(exchange_name + ":/" + symbol):
        balance = Stream.sensor(wallet, lambda w: w.balance.as_float(), dtype="float").rename("wallet_balance")
        equity = Stream.sensor(wallet, lambda w: float(w.equity), dtype="float").rename("wallet_equity")
        free_margin = Stream.sensor(wallet, lambda w: float(w.free_margin), dtype="float").rename("wallet_free_margin")
        
        #free_balance = Stream.sensor(wallet, lambda w: w.balance.as_float(), dtype="float").rename("wallet_free")
        #locked_balance = Stream.sensor(wallet, lambda w: w.locked_balance.as_float(), dtype="float").rename("wallet_locked")
        #total_balance = Stream.sensor(wallet, lambda w: w.total_balance.as_float(), dtype="float").rename("wallet_total")

        streams += [balance, equity, free_margin]
        """
        if include_worth:
            price = Stream.select(wallet.exchange.streams(), lambda node: node.name.endswith(symbol))
            worth = (price * total_balance).rename("worth")
            streams += [worth]
        """
    return streams

"""
def _create_position_source(position: 'Position', include_worth: bool = True) -> 'List[Stream[float]]':
    exchange_name = position.exchange.name
    symbol = position.instrument.symbol
    streams = []
    with NameSpace(exchange_name + ":/p"):
        profit = Stream.sensor(position, lambda p: p.profit, dtype='Decimal').rename("p_profit")
        
        margin = Stream.sensor(position, lambda p: p.margin, dtype="float").rename("position_margin")
        size = Stream.sensor(position, lambda p: p.size.as_float(), dtype="float").rename("position_size")
        side = Stream.sensor(position, lambda p: p.side.value(), dtype="str").rename("position_side")
        sym = Stream.sensor(position, lambda p: p.instrument.symobl, dtype="str").rename("position_symbol")
        

        streams += [profit]
        
        if include_worth:
            price = Stream.select(position.exchange.streams(), lambda node: node.name.endswith(symbol))
            worth = (price * total_balance).rename("worth")
            streams += [worth]
    return streams
"""

def _create_portfolio_source(portfolio: 'Portfolio', include_worth: bool = False) -> 'List[Stream[float]]':
    streams = []
    #free_margin = Stream.sensor(portfolio, lambda p: p.free_margin, dtype='float').rename("portfolio_free_margin")
    side = Stream.sensor(portfolio, lambda p: p.position_side_EURUSD, dtype='float').rename("portfolio_position_side_EURUSD")
    buying_power_ratio = Stream.sensor(portfolio, lambda p: p.buying_power_ratio, dtype='float').rename("portfolio_buying_power_ratio")
    profit_loss = Stream.sensor(portfolio, lambda p: p.profit_loss, dtype='float').rename("portfolio_profit_loss")
    drawdown_pct = Stream.sensor(portfolio, lambda p: p.drawdown_pct, dtype='float').rename("portfolio_drawdown_pct")

    streams += [buying_power_ratio, side, profit_loss, drawdown_pct]
    return streams

def _create_internal_streams(portfolio: 'Portfolio') -> 'List[Stream[float]]':
    """Creates a list of streams to describe a `Portfolio`.

    Parameters
    ----------
    portfolio : `Portfolio`
        The portfolio to make the streams for.

    Returns
    -------
    `List[Stream[float]]`
        A list of streams to describe the `portfolio`.
    """
    base_symbol = portfolio.base_instrument.symbol
    sources = []

    for wallet in portfolio.wallets:
        symbol = wallet.instrument.symbol
        sources += wallet.exchange.price_streams()
        sources += [wallet.exchange._time_stream]
        sources += _create_wallet_source(wallet, include_worth=(symbol != base_symbol))
    """
    if portfolio.positions:
        for p in portfolio.positions:
            sources += _create_position_source(p, include_worth=(symbol != base_symbol))
    """

    worth_streams = []
    #max_profit_streams = Stream.sensor(portfolio, lambda p: p.max_net_worth, dtype='float').rename("Portfolio_max_profit")
    max_loss_streams = []
    max_drawdown_streams = []
    
    for s in sources:
        if s.name.endswith(base_symbol + ":/wallet_equity"):
            worth_streams += [s]

    net_worth = Stream.reduce(worth_streams).sum().rename("net_worth")
    sources += [net_worth]
    #sources += [max_profit_streams]
    
    return sources

def _dict_merge(dict1, dict2): 
    res = {**dict1, **dict2}
    return res

class ObservationHistory(object):
    """Stores observations from a given episode of the environment.

    Parameters
    ----------
    window_size : int
        The amount of observations to keep stored before discarding them.

    Attributes
    ----------
    window_size : int
        The amount of observations to keep stored before discarding them.
    rows : pd.DataFrame
        The rows of observations that are used as the environment observation
        at each step of an episode.

    """

    def __init__(self, window_size: int) -> None:
        self.window_size = window_size
        self.rows = OrderedDict()
        self.index = 0

    def push(self, row: dict) -> None:
        """Stores an observation.

        Parameters
        ----------
        row : dict
            The new observation to store.
        """
        self.rows[self.index] = row
        self.index +=1
        if len(self.rows.keys()) > self.window_size:
            del self.rows[list(self.rows.keys())[0]]

    def observe(self) -> 'np.array':
        """Gets the observation at a given step in an episode

        Returns
        -------
        `np.array`
            The current observation of the environment.
        """
        rows = self.rows.copy()

        if len(rows) < self.window_size:
            size = self.window_size - len(rows)
            padding = np.zeros((size, len(rows[list(rows.keys())[0]])))
            r = np.array([list(inner_dict.values()) for inner_dict in rows.values()])
            rows = np.concatenate((padding, r))

        if isinstance(rows, OrderedDict):
            rows = np.array([list(inner_dict.values()) for inner_dict in rows.values()])

        rows = np.nan_to_num(rows)

        return rows

    def reset(self) -> None:
        """Resets the observation history"""
        self.rows = OrderedDict()
        self.index = 0


class TensorTradeObserver(Observer):
    """The TensorTrade observer that is compatible with the other `default`
    components.

    Parameters
    ----------
    portfolio : `Portfolio`
        The portfolio to be used to create the internal data feed mechanism.
    feed : `DataFeed`
        The feed to be used to collect observations to the observation window.
    renderer_feed : `DataFeed`
        The feed to be used for giving information to the renderer.
    window_size : int
        The size of the observation window.
    min_periods : int
        The amount of steps needed to warmup the `feed`.
    **kwargs : keyword arguments
        Additional keyword arguments for observer creation.

    Attributes
    ----------
    feed : `DataFeed`
        The master feed in charge of streaming the internal, external, and
        renderer data feeds.
    window_size : int
        The size of the observation window.
    min_periods : int
        The amount of steps needed to warmup the `feed`.
    history : `ObservationHistory`
        The observation history.
    renderer_history : `List[dict]`
        The history of the renderer data feed.
    """

    def __init__(self,
                 portfolio: 'Portfolio',
                 feed: 'DataFeed' = None,
                 renderer_feed: 'DataFeed' = None,
                 window_size: int = 1,
                 min_periods: int = None,
                 #seed_start: int = None,
                 **kwargs) -> None:
        internal_group = Stream.group(_create_internal_streams(portfolio)).rename("internal")
        external_group = Stream.group(feed.inputs).rename("external")
        portfolio_group = Stream.group(_create_portfolio_source(portfolio)).rename("portfolio")

        if renderer_feed:
            renderer_group = Stream.group(renderer_feed.inputs).rename("renderer")

            self.feed = DataFeed([
                internal_group,
                external_group,
                renderer_group,
                portfolio_group
            ])
        else:
            self.feed = DataFeed([
                internal_group,
                external_group,
                portfolio_group
            ])

        self.window_size = window_size
        self.min_periods = min_periods
        #self.seed_start = seed_start

        self._observation_dtype = kwargs.get('dtype', np.float32)
        self._observation_lows = kwargs.get('observation_lows', -np.inf)
        self._observation_highs = kwargs.get('observation_highs', np.inf)

        self.history = ObservationHistory(window_size=window_size)

        initial_obs = _dict_merge(self.feed.next()["external"], self.feed.next()["portfolio"])
        n_features = len(initial_obs.keys())

        self._observation_space = Box(
            low=self._observation_lows,
            high=self._observation_highs,
            shape=(self.window_size, n_features),
            dtype=self._observation_dtype
        )

        self.feed = self.feed.attach(portfolio)

        self.renderer_history = []

        self.feed.reset()
        self.warmup()

    @property
    def observation_space(self) -> Space:
        return self._observation_space

    def warmup(self) -> None:
        """Warms up the data feed.
        """
        if self.min_periods is not None:
            """
            if self.seed_start is not None:
                seedX = np.random.randint(1,24*100) * self.seed_start
                obs_row = _dict_merge(self.feed.next()["external"],self.feed.next()["portfolio"])
                for _ in range(seedX):
                    if self.has_next():
                        self.history.push(obs_row)
                for _ in range(seedX, self.min_periods+seedX):
                    if self.has_next():
                        obs_row = _dict_merge(self.feed.next()["external"],self.feed.next()["portfolio"])
                        self.history.push(obs_row)
            """
            #else:
            for _ in range(self.min_periods):
                if self.has_next():
                    obs_row = _dict_merge(self.feed.next()["external"],self.feed.next()["portfolio"])
                    self.history.push(obs_row)

    def observe(self, env: 'TradingEnv') -> np.array:
        """Observes the environment.

        As a consequence of observing the `env`, a new observation is generated
        from the `feed` and stored in the observation history.

        Returns
        -------
        `np.array`
            The current observation of the environment.
        """
        data = self.feed.next()

        # Save renderer information to history
        if "renderer" in data.keys():
            self.renderer_history += [data["renderer"]]

        # Push new observation to observation history
        # Not successfull: Try only observe external data, portfolio status only being observed by current
        #obs_row = data["external"]
        obs_row = _dict_merge(data["external"], data["portfolio"])
        self.history.push(obs_row)

        obs = self.history.observe()
        obs = obs.astype(self._observation_dtype) #force converting all obs to
        return obs

    def has_next(self) -> bool:
        """Checks if there is another observation to be generated.

        Returns
        -------
        bool
            Whether there is another observation to be generated.
        """
        return self.feed.has_next()

    def reset(self) -> None:
        """Resets the observer"""
        self.renderer_history = []
        self.history.reset()
        self.feed.reset()
        self.warmup()
