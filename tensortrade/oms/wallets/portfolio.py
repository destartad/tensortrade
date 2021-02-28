# Copyright 2019 The TensorTrade Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import re
from decimal import Decimal

from typing import Callable, Tuple, List, TypeVar


from tensortrade.core import Component, TimedIdentifiable
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.orders import OrderListener
from tensortrade.oms.instruments import *
from tensortrade.oms.wallets.wallet import Wallet
from tensortrade.oms.wallets.position import Position
from tensortrade.oms.wallets.ledger import Ledger
from collections import OrderedDict
from tensortrade.oms.orders.trade import TradeSide

WalletType = TypeVar("WalletType", Wallet, Tuple[Exchange, Instrument, float])
PositionType = TypeVar("PositionType", Position, Tuple[Exchange, Instrument, float, TradeSide, Decimal, Decimal, str])


class Portfolio(Component, TimedIdentifiable):
    """A portfolio of wallets on exchanges.

    Parameters
    ----------
    base_instrument : `Instrument`
        The exchange instrument used to measure value and performance statistics.
    wallets : `List[WalletType]`
        The wallets to be used in the portfolio.
    order_listener : `OrderListener`
        The order listener to set for all orders executed by this portfolio.
    performance_listener : `Callable[[OrderedDict], None]`
        The performance listener to send all portfolio updates to.
    """

    registered_name = "portfolio"

    def __init__(self,
                 base_instrument: Instrument,
                 wallets: List[WalletType] = None,
                 positions: List[PositionType] = None,
                 order_listener: 'OrderListener' = None,
                 performance_listener: Callable[[OrderedDict], None] = None):
        super().__init__()

        #Order to change positions
        wallets = wallets or []
        positions = positions or []

        self.base_instrument = self.default('base_instrument', base_instrument)
        self.order_listener = self.default('order_listener', order_listener)
        self.performance_listener = self.default('performance_listener', performance_listener)
        self._wallets = {}
        self._positions = {}

        for wallet in wallets:
            self.add_wallet(wallet)

        for position in positions:
            self.add_position(position)

        self._initial_balance = self.base_balance
        self._initial_net_worth = None
        self._net_worth = None
        self._performance = None
        self._keys = None
        self._max_net_worth: float = float(self._initial_balance.size)
        self._max_profit: float = 0.00
        self._max_loss: float = 0.00
        self._max_drawdown: float = 0.00

        #TODO: calculation sum of open postion - margin/profit/swap/commission
        self._current_balanace = self._initial_balance
        self._equity = self._initial_balance.size
        self._margin: Decimal = 0.00
        self._free_margin = self._initial_balance
        self._total_open_position_size = None
        self._total_open_buy_position_size = None
        self._total_open_sell_position_size = None
        self._total_position_profit = None
        self._total_open_buy_position_profit = None
        self._total_open_sell_position_profit = None

        """if self._margin != None
            self._margin_level = self._equity/self._margin
        """
    @property
    def max_net_worth(self):
        return self._max_net_worth

    @property
    def wallets(self) -> 'List[Wallet]':
        """All the wallets in the portfolio. (`List[Wallet]`, read-only)"""
        return list(self._wallets.values())

    @property
    def positions(self) -> 'List[Position]':
        """All the positions in the portfolio. (`List[Position]`, read-only)"""
        return list(self._positions.values())

    @property
    def exchanges(self) -> 'List[Exchange]':
        """All the exchanges in the portfolio. (`List[Exchange]`, read-only)"""
        exchanges = []
        for w in self.wallets:
            if w.exchange not in exchanges:
                exchanges += [w.exchange]
        for p in self.positions:
            if p.exchange not in exchanges:
                exchanges += [p.exchange]
        return exchanges

    @property
    def ledger(self) -> 'Ledger':
        """The ledger that keeps track of transactions. (`Ledger`, read-only)"""
        return Wallet.ledger

    @property
    def exchange_pairs(self) -> 'List[ExchangePair]':
        """All the exchange pairs in the portfolio. (`List[ExchangePair]`, read-only)"""
        exchange_pairs = []
        for exchange in self.exchanges:
            for ti in exchange.options.trading_instruments:
                exchange_pairs += [ExchangePair(exchange, self.base_instrument/ti)]
        return exchange_pairs

    @property
    def initial_balance(self) -> 'Quantity':
        """The initial balance of the base instrument over all wallets. (`Quantity`, read-only)"""
        return self._initial_balance

    @property
    def base_balance(self) -> 'Quantity':
        """The current balance of the base instrument over all wallets. (`Quantity`, read-only)"""
        return self.balance(self.base_instrument) 

    @property
    def initial_net_worth(self) -> float:
        """The initial net worth of the portfolio. (float, read-only)"""
        return self._initial_net_worth

    @property
    def net_worth(self) -> float:
        """The current net worth of the portfolio. (float, read-only)"""
        return self._net_worth

    @property
    def profit_loss(self) -> float:
        """The percent loss in net worth since the last reset. (float, read-only)"""
        _profit_loss = 1.0 - self.net_worth / self.initial_net_worth
        return float(_profit_loss)

    @property
    def performance(self) -> 'OrderedDict':
        """The performance of the portfolio since the last reset. (`OrderedDict`, read-only)"""
        return self._performance

    @property
    def balances(self) -> 'List[Quantity]':
        """The current unlocked balance of each instrument over all wallets. (`List[Quantity]`, read-only)"""
        return [wallet.balance for wallet in self._wallets.values()]

    @property
    def locked_balances(self) -> 'List[Quantity]':
        """The current locked balance of each instrument over all wallets. (`List[Quantity]`, read-only)"""
        return [wallet.locked_balance for wallet in self._wallets.values()]

    @property
    def total_balances(self) -> 'List[Quantity]':
        """The current total balance of each instrument over all wallets. (`List[Quantity]`, read-only)"""
        return [wallet.total_balance for wallet in self._wallets.values()]
    @property
    def margin(self) -> float:
        return get_all_margin()
        
    def get_all_margin(self):
        """calculate margin for all open positions"""
        for p in self._positions.items():
            margin += p.margin
        return margin

    def balance(self, instrument: Instrument) -> 'Quantity':
        """Gets the total balance of the portfolio in a specific instrument
        available for use.

        Parameters
        ----------
        instrument : `Instrument`
            The instrument to compute the balance for.

        Returns
        -------
        `Quantity`
            The balance of the instrument over all wallets.
        """
        balance = Quantity(instrument, 0)

        for (_, symbol), wallet in self._wallets.items():
            if symbol == instrument.symbol:
                balance += wallet.balance

        return balance

    def locked_balance(self, instrument: Instrument) -> 'Quantity':
        """Gets the total balance a specific instrument locked in orders over
        the entire portfolio.

        Parameters
        ----------
        instrument : `Instrument`
            The instrument to find locked balances for.

        Returns
        -------
        `Quantity`
            The total locked balance of the instrument.
        """
        balance = Quantity(instrument, 0)

        for (_, symbol), wallet in self._wallets.items():
            if symbol == instrument.symbol:
                balance += wallet.locked_balance

        return balance

    def total_balance(self, instrument: Instrument) -> 'Quantity':
        """Gets the total balance of a specific instrument over the portfolio,
        both available for use and locked in orders.

        Parameters
        ----------
        instrument : `Instrument`
            The instrument to get total balance of.

        Returns
        -------
        `Quantity`
            The total balance of `instrument` over the portfolio.
        """
        return self.balance(instrument) + self.locked_balance(instrument)

    def get_wallet(self, exchange_id: str, instrument: 'Instrument') -> 'Wallet':
        """Gets wallet by the `exchange_id` and `instrument`.

        Parameters
        ----------
        exchange_id : str
            The exchange id used to identify the wallet.
        instrument : `Instrument`
            The instrument used to identify the wallet.

        Returns
        -------
        `Wallet`
            The wallet associated with `exchange_id` and `instrument`.
        """
        return self._wallets[(exchange_id, instrument.symbol)]

    def get_position(self, exchange_id: str, instrument: 'Instrument') -> 'Position':
        return self._positions[(exchange_id, instrument.symbol)]

    def add_wallet(self, wallet: WalletType) -> None:
        """Adds a wallet to the portfolio.

        Parameters
        ----------
        wallet : `WalletType`
            The wallet to add to the portfolio.
        """
        if isinstance(wallet, tuple):
            wallet = Wallet.from_tuple(wallet)
        self._wallets[(wallet.exchange.id, wallet.instrument.symbol)] = wallet

    def add_position(self, position: PositionType) -> None:
        """Adds a position to the portfolio.

        Parameters
        ----------
        Position : `PositionType`
            The position to add to the portfolio.
        """
        if isinstance(position, tuple):
            position = position.from_tuple(position)
        self._positions[(position.id, position.exchange.id, position.instrument.symbol)] = position


    def remove_wallet(self, wallet: 'Wallet') -> None:
        """Removes a wallet from the portfolio.

        Parameters
        ----------
        wallet : `Wallet`
            The wallet to be removed.
        """
        self._wallets.pop((wallet.exchange.id, wallet.instrument.symbol), None)
    
    def remove_position(self, position: 'Position') -> None:
        """Removes a wallet from the portfolio.

        Parameters
        ----------
        wallet : `Wallet`
            The wallet to be removed.
        """
        self._positions.pop((position.id, position.exchange.id, position.instrument.symbol), None)

    def remove_pair(self, exchange: 'Exchange', instrument: 'Instrument') -> None:
        """Removes a wallet from the portfolio by `exchange` and `instrument`.

        Parameters
        ----------
        exchange : `Exchange`
            The exchange of the wallet to be removed.
        instrument : `Instrument`
            The instrument of the wallet to be removed.
        """
        self._wallets.pop((exchange.id, instrument.symbol), None)
        if self._positions:
            self._positions.pop((exchange.id, instrument.symbol), None)

    @staticmethod
    def _find_keys(data: dict) -> 'List[str]':
        """Finds the keys that can attributed to the net worth of the portfolio.

        Parameters
        ----------
        data : dict
            The observer feed data point to search for keys attributed to net
            worth.

        Returns
        -------
        `List[str]`
            The list of strings attributed to net worth.
        """
        price_pattern = re.compile("\\w+:/([A-Z]{3,4}).([A-Z]{3,4})")
        endings = [
            ":/free",
            ":/locked",
            ":/total",
            "worth"
        ]

        keys = []
        for k in data.keys():
            if any(k.endswith(end) for end in endings):
                keys += [k]
            elif price_pattern.match(k):
                keys += [k]

        return keys
    
    def update(self):
        _margins = Decimal('0.00').quantize(Decimal('0.00'))
        _profits = Decimal('0.00').quantize(Decimal('0.00'))
        for p in self.positions:
            _margins += p.margin
            _profits += p.profit
            #p.last_update_time = 
            p.exchange_current_time = p.exchange._time_stream.value.to_pydatetime()
            for ep in self.exchange_pairs:
                if p.instrument == ep.pair.quote:
                    p.current_price = ep.price
        for w in self.wallets:
            w.margin = _margins
            w.profit = _profits

        self._max_net_worth = max(float(self._max_net_worth), float(self.net_worth))
        self._max_drawdown = max((float(self._max_net_worth - float(self.net_worth))), self._max_drawdown)
        
        if float(self.net_worth - self._initial_net_worth) >= 0.00:
            self._max_profit = max(self._max_profit, float(self.net_worth - self._initial_net_worth))
        else:
            self._max_loss = max(self._max_loss, float(self._initial_net_worth - self.net_worth))

    def on_next(self, data: dict) -> None:
        """Updates the performance metrics.

        Parameters
        ----------
        data : dict
            The data produced from the observer feed that is used to
            update the performance metrics.
        """
        data = data["internal"]

        if not self._keys:
            self._keys = self._find_keys(data)

        index = self.clock.step
        performance_data = {k: data[k] for k in self._keys}
        performance_data['base_symbol'] = self.base_instrument.symbol
        performance_step = OrderedDict()
        performance_step[index] = performance_data
        
        net_worth = data['net_worth']

        if self._performance is None:
            self._performance = performance_step
            self._initial_net_worth = net_worth
            self._net_worth = net_worth
        else:
            self._performance.update(performance_step)
            self._net_worth = net_worth

        if self.performance_listener:
            self.performance_listener(performance_step)

    def reset(self) -> None:
        """Resets the portfolio."""
        self._initial_balance = self.base_balance
        self._initial_net_worth = None
        self._net_worth = None
        self._performance = None
        self._max_net_worth: float = float(self._initial_balance.size)
        self._max_profit: float = 0.00
        self._max_loss: float = 0.00
        self._max_drawdown: float = 0.00

        self.ledger.reset()
        for wallet in self._wallets.values():
            wallet.reset()
        if self._positions:
            for p in self.positions:
                self.remove_position(p)