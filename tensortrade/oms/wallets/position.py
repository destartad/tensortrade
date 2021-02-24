# Copyright 2020 The TensorTrade Authors.
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

from typing import Dict, Tuple
from collections import namedtuple
from decimal import Decimal
from enum import Enum

import numpy as np

from tensortrade.core import Identifiable,TimedIdentifiable

from tensortrade.core.exceptions import (
    InsufficientFunds,
    DoubleLockedQuantity,
    DoubleUnlockedQuantity,
    QuantityNotLocked,
    InvalidTradeSide
)
from tensortrade.oms.instruments import Instrument, Quantity, ExchangePair
from tensortrade.oms.orders import Order
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.wallets.ledger import Ledger


class PositionStatus(Enum):
    """An enumeration for the status of a position."""

    OPEN = "open"
    CLOSED = "closed"

    def __str__(self):
        return self.value

class Position(TimedIdentifiable):
    """A position stores the balance of a specific instrument on a specific exchange.

    Parameters
    ----------
    exchange : `Exchange`
        The exchange associated with this position.
    balance : `Quantity`
        The initial balance quantity for the postion.
    """

    ledger = Ledger()

    def __init__(self, exchange: 'Exchange', balance: 'Quantity', side, executed_price, current_price, exchange_current_time):
        self.exchange = exchange
        self.current_price = current_price
        self.side = side
        self.instrument = balance.instrument
        self.size = balance.size
        self._executed_price = executed_price
        self._margin = self._executed_price * self.instrument.contract_size * self.size / exchange.options.leverage
        self.status = PositionStatus.OPEN
        self._profit: Decimal = 0.00
        self._filled_time = exchange_current_time
        self._swap = Decimal = 0.00

    @property
    def swap(self) -> "Decimal":
        return self._swap

    @swap.setter
    def swap(self, value) -> "Decimal":
        self._swap = swap

    @property
    def filled_time(self) -> "Datetime":
        return self._filled_time

    @property
    def exchange_current_time(self) -> "Datetime":
        return exchange_current_time

    @property
    def evaluated_price(self) -> "Quantity":
        if self.side.value == "buy":
            _evaluated_price = self.current_price - Decimal(self.instrument.spread)
        elif self.side.value == "sell":
            _evaluated_price = self.current_price + Decimal(self.instrument.spread)
        else:
            raise InvalidTradeSide()
        return _evaluated_price.quantize(Decimal(10)**(-self.instrument.precision))
        
    @property
    def margin(self) -> 'Decimal':
        self._margin = self._margin.quantize(Decimal(10)**-2)
        return self._margin

    @property
    def profit(self) -> 'Decimal':
        if self.side.value == "buy":
            self._profit = (self.evaluated_price - self._executed_price) * self.size * self.instrument.contract_size
        else:
            self._profit = (-self.evaluated_price + self._executed_price) * self.size * self.instrument.contract_size

        self._profit = self._profit.quantize(Decimal(10)**-2)

        #self._profit += self.swap

        return self._profit

    @classmethod
    def from_tuple(cls, position_tuple: 'Tuple[Exchange, Instrument, float]') -> 'Position':
        """Creates a wallet from a wallet tuple.

        Parameters
        ----------
        wallet_tuple : `Tuple[Exchange, Instrument, float]`
            A tuple containing an exchange, instrument, and amount.

        Returns
        -------
        `Wallet`
            A wallet corresponding to the arguments given in the tuple.
        """
        exchange, instrument, balance = position_tuple
        return cls(exchange, Quantity(instrument, balance))

    
    def reset(self) -> None:
        """Resets the position."""
        self.balance = Quantity(self.instrument, self._initial_size).quantize()
        self._locked = {}

    def __str__(self) -> str:
        return '<Wallet: balance={}, locked={}>'.format(self.balance, self.locked_balance)

    def __repr__(self) -> str:
        return str(self)

    """
    TODO: swap
    @property
    def swap(self) : 
        return _swap

    @swap.setter(self):
    def swap(self,self.clock):
        return _swap
    """