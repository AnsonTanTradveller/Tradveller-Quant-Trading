from os import environ
from datetime import datetime, timedelta, timezone
from cybotrade.strategy import Strategy as BaseStrategy
from cybotrade.runtime import Runtime
from cybotrade.models import (
    Exchange,
    Interval,
    OrderSide,
    RuntimeConfig,
    RuntimeMode,
)
import math
import numpy as np
import talib
import asyncio
import logging
import colorlog
import cybotrade_indicators
from logging.handlers import TimedRotatingFileHandler
from cybotrade.permutation import Permutation

class Strategy(BaseStrategy):
    # Indicator params
    # atr
    atr_length = 108
    atr_stoploss_multiplier = 6
    # CCI
    cci_length1 = 18
    cci_length2 = 54
    upper_cci = 200
    lower_cci = -200
    # To keep track of the buy/sell flat
    can_entry_buy = True
    can_entry_sell = True
    # To keep track of the trend in CCI
    cci_up_trend = False
    cci_down_trend = False
    # risk to reward
    risk_to_reward = 1
    qty = 0.1
    btcount = 0

    def __init__(self):
        handler = colorlog.StreamHandler()
        handler.setFormatter(
            colorlog.ColoredFormatter(f"%(log_color)s{Strategy.LOG_FORMAT}")
        )
        file_handler = TimedRotatingFileHandler("y_st017-test.log", when="h")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(Strategy.LOG_FORMAT))
        super().__init__(log_level=logging.INFO, handlers=[handler, file_handler])

    async def set_param(self, identifier, value):
        logging.info(f"Setting {identifier} to {value}")
        if identifier == "atr_length":
            self.atr_length = int(value)
        elif identifier == "atr_stoploss_multiplier":
            self.atr_stoploss_multiplier = float(value)
        elif identifier == "cci_length1":
            self.cci_length1 = int(value)
        elif identifier == "cci_length2":
            self.cci_length2 = int(value)
        else:
            logging.error(f"Could not set {identifier}, not found")

    async def on_candle_closed(self, strategy, topic, symbol):
        candles = self.data_map[topic]
        start_time = np.array(list(map(lambda c: float(c["start_time"]), candles)))
        open = np.array(list(map(lambda c: float(c["open"]), candles)))
        high = np.array(list(map(lambda c: float(c["high"]), candles)))
        low = np.array(list(map(lambda c: float(c["low"]), candles)))
        close = np.array(list(map(lambda c: float(c["close"]), candles)))
        volume = np.array(list(map(lambda c: float(c["volume"]), candles)))
        logging.debug(
            f"open : {open[-1]}, close: {close[-1]}, high: {high[-1]}, low: {low[-1]}, at {start_time[-1]}"
        )
        if (
            len(close) < self.atr_length * 3
            or len(close) < self.cci_length1 * 3
            or len(close) < self.cci_length2 * 3
        ):
            logging.info("Not enough candles to calculate indicators")
            return

        # atr
        atr = cybotrade_indicators.atr(
            high=high, low=low, close=close, period=self.atr_length
        )
        # CCI
        cci_fast_line = cybotrade_indicators.cci(
            high=high, low=low, close=close, period=self.cci_length1
        )
        cci_slow_line = cybotrade_indicators.cci(
            high=high, low=low, close=close, period=self.cci_length2
        )

        position = await strategy.position(symbol=symbol, exchange=Exchange.BybitLinear)
        # wallet_balance = await strategy.get_current_available_balance(symbol=symbol)
        qty = self.qty

        if cci_fast_line[-1] < self.lower_cci:
            self.cci_up_trend = True
            self.cci_down_trend = False
        if cci_fast_line[-1] > self.upper_cci:
            self.cci_up_trend = False
            self.cci_down_trend = True

        # if cci_fast_line < 200 => up trend
        # cci_fast_line[-1] > cci_slow_line[-1] & cci_fast_line[-2] < cci_slow_line[-2] & cci up trend
        if (
            position.long.quantity <= 0.0
            and position.short.quantity <= 0.0
            and self.cci_up_trend
            and cci_fast_line[-1] > cci_slow_line[-1]
            and cci_fast_line[-2] < cci_slow_line[-2]
            and self.can_entry_buy
        ):
            stop_loss = low[-1] - atr[-1] * self.atr_stoploss_multiplier
            take_profit = close[-1] + (self.risk_to_reward * (close[-1] - stop_loss))
            await strategy.open(side=OrderSide.Buy, quantity=qty, is_hedge_mode=False, take_profit=take_profit, stop_loss=stop_loss, symbol=symbol, exchange=Exchange.BybitLinear, is_post_only=False)
            logging.info(
                f"Placed a buy order with qty {qty} when close: {close[-1]}, tp: {take_profit}, sl: {stop_loss} at time {start_time[-1]}"
            )
            self.cci_up_trend = False
            self.can_entry_buy = False
            self.can_entry_sell = True

        # if cci_fast_line > 200 => down trend
        # cci_fast_line[-1] < cci_slow_line[-1] & cci_fast_line[-2] > cci_slow_line[-2] & cci down trend
        if (
            position.short.quantity <= 0.0
            and position.long.quantity <= 0.0
            and self.cci_down_trend
            and cci_fast_line[-1] < cci_slow_line[-1]
            and cci_fast_line[-2] > cci_slow_line[-2]
            and self.can_entry_sell
        ):
            stop_loss = high[-1] + self.atr_stoploss_multiplier * atr[-1]
            take_profit = close[-1] - (self.risk_to_reward * (stop_loss - close[-1]))
            await strategy.open(side=OrderSide.Sell, quantity=qty, is_hedge_mode=False, take_profit=take_profit, stop_loss=stop_loss, symbol=symbol, exchange=Exchange.BybitLinear, is_post_only=False)
            logging.info(
                f"Placed a sell order with qty {qty} when close: {close[-1]}, tp: {take_profit}, sl: {stop_loss} at time {start_time[-1]}"
            )
            self.cci_down_trend = False
            self.can_entry_buy = True
            self.can_entry_sell = False

config = RuntimeConfig(
    mode=RuntimeMode.Backtest,
    datasource_topics=[],
    active_order_interval=1,
    initial_capital=10000.0,
    candle_topics=["candles-1h-BTC/USDT-bybit"],
    start_time=datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
    end_time=datetime(2024, 1, 30, 0, 0, 0, tzinfo=timezone.utc),
    api_key="YOUR_CYBOTRADE_API_KEY",
    api_secret="YOUR_CYBOTRADE_API_SECRET",
    data_count=500,
    # exchange_keys="./z_exchange-keys.json",
    )

permutation = Permutation(config)
hyper_parameters = {}
hyper_parameters['atr_length'] = [108]
hyper_parameters['atr_stoploss_multiplier'] = [6]
hyper_parameters['cci_length1'] = [18]
hyper_parameters['cci_length2'] = [54]
 
async def start_backtest():
  await permutation.run(hyper_parameters, Strategy)
 
asyncio.run(start_backtest())