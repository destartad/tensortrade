import logging
from decimal import Decimal

from tensortrade.core import Clock
from tensortrade.oms.wallets import Wallet
from tensortrade.oms.wallets import Position
from tensortrade.oms.exchanges import ExchangeOptions
from tensortrade.oms.orders import Order, Trade, TradeType, TradeSide


def execute_buy_order(order: 'Order',
                      cash_wallet: 'Wallet',
                      portfolio: 'Portfolio',
                      current_price: float,
                      options: 'ExchangeOptions',
                      clock: 'Clock',
                      exchange_current_time: 'Datatime') -> 'Trade':
    """
    Executes a buy order on the exchange.
    1. create position object
    2. add position object in portfolio
    3. portfolio update

    Parameters
    ----------
    order : `Order`
        The order that is being filled.
    base_wallet : `Wallet`
        The wallet of the base instrument.
    quote_wallet : `Wallet`
        The wallet of the quote instrument.
    current_price : float
        The current price of the exchange pair.
    options : `ExchangeOptions`
        The exchange options.
    clock : `Clock`
        The clock for the trading process..

    Returns
    -------
    `Trade`
        The executed trade that was made.
    """
    if order.type == TradeType.LIMIT and order.price < current_price:
        return None

    filled = order.remaining.contain(order.exchange_pair)

    if order.type == TradeType.MARKET:
        scale = order.price / max(current_price, order.price)
        filled = scale * filled

    commission = options.commission * filled
    quantity = filled - commission

    executed_price = current_price.quantize(Decimal(10) ** -quantity.instrument.precision)
    
    position = Position(
        exchange=order.exchange_pair.exchange,
        current_price=current_price,
        balance=quantity,
        side=order.side,
        executed_price=executed_price,
        exchange_current_time=exchange_current_time
    )
    
    if position.margin >= cash_wallet.free_margin:
        return None

    portfolio.add_position(position)
    cash_wallet.update_by_position(position)

    trade = Trade(
        order_id=order.id,
        step=clock.step,
        exchange_pair=order.exchange_pair,
        side=TradeSide.BUY,
        trade_type=order.type,
        quantity=quantity,
        price=current_price,
        commission=commission,
        exchange_current_time=exchange_current_time
    )

    return trade


def execute_sell_order(order: 'Order',
                      cash_wallet: 'Wallet',
                      portfolio: 'Portfolio',
                      current_price: float,
                      options: 'ExchangeOptions',
                      clock: 'Clock',
                      exchange_current_time: 'Datatime') -> 'Trade':
    """Executes a sell order on the exchange.

    Parameters
    ----------
    order : `Order`
        The order that is being filled.
    base_wallet : `Wallet`
        The wallet of the base instrument.
    quote_wallet : `Wallet`
        The wallet of the quote instrument.
    current_price : float
        The current price of the exchange pair.
    options : `ExchangeOptions`
        The exchange options.
    clock : `Clock`
        The clock for the trading process..

    Returns
    -------
    `Trade`
        The executed trade that was made.
    """
    if order.type == TradeType.LIMIT and order.price > current_price:
        return None

    filled = order.remaining.contain(order.exchange_pair)
    
    if order.type == TradeType.MARKET:
        scale = order.price / max(current_price, order.price)
        filled = scale * filled

    commission = options.commission * filled
    quantity = filled - commission

    """
    if commission.size < Decimal(10) ** -quantity.instrument.precision:
        logging.warning("Commission is less than instrument precision. Canceling order. "
                        "Consider defining a custom instrument with a higher precision.")
        order.cancel("COMMISSION IS LESS THAN PRECISION.")
        return None
    """

    executed_price = current_price.quantize(Decimal(10) ** -quantity.instrument.precision)
    
    position = Position(
        exchange=order.exchange_pair.exchange,
        current_price=current_price,
        balance=quantity,
        side=order.side,
        executed_price=executed_price,
        exchange_current_time=exchange_current_time
    )
    
    if position.margin >= cash_wallet.free_margin:
        return None
    
    portfolio.add_position(position)
    cash_wallet.update_by_position(position)

    trade = Trade(
        order_id=order.id,
        step=clock.step,
        exchange_pair=order.exchange_pair,
        side=TradeSide.SELL,
        trade_type=order.type,
        quantity=quantity,
        price=current_price,
        commission=commission,
        exchange_current_time=exchange_current_time
    )

    return trade

def execute_close_order(order: 'Order',
                       cash_wallet: 'Wallet',
                       portfolio: 'Portfolio',
                       current_price: float,
                       options: 'ExchangeOptions',
                       clock: 'Clock',
                       exchange_current_time: 'Datatime') -> 'Trade':
    """Executes a close order on the exchange."""

    if portfolio.positions == None:
        return None

    portfolio.update()
    for p in portfolio.positions:
        portfolio.remove_position(p)
    cash_wallet.update_on_close()

    trade = Trade(
        order_id=order.id,
        step=clock.step,
        exchange_pair=order.exchange_pair,
        side=TradeSide.CLOSE,
        trade_type=order.type,
        quantity=0.00,
        price=current_price,
        commission=0.00,
        exchange_current_time=exchange_current_time
    )

    return trade

def execute_order(order: 'Order',
                  cash_wallet: 'Wallet',
                  portfolio: 'Portfolio',
                  current_price: float,
                  options: 'Options',
                  clock: 'Clock',
                  exchange_current_time: 'Datatime') -> 'Trade':
    """Executes an order on the exchange.

    Parameters
    ----------
    order : `Order`
        The order that is being filled.
    base_wallet : `Wallet`
        The wallet of the base instrument.
    quote_wallet : `Wallet`
        The wallet of the quote instrument.
    current_price : float
        The current price of the exchange pair.
    options : `ExchangeOptions`
        The exchange options.
    clock : `Clock`
        The clock for the trading process..

    Returns
    -------
    `Trade`
        The executed trade that was made.
    """
    kwargs = {"order": order,
              "cash_wallet": cash_wallet,
              "portfolio": portfolio,
              "current_price": current_price,
              "options": options,
              "clock": clock,
              "exchange_current_time": exchange_current_time}

    if order.is_buy:
        trade = execute_buy_order(**kwargs)
    elif order.is_sell:
        trade = execute_sell_order(**kwargs)
    elif order.is_close:
        trade = execute_close_order(**kwargs)
    else:
        trade = None

    return trade
