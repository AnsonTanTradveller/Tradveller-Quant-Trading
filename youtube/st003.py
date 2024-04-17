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
    atr_length = 13
    # MACD
    short_period = 13
    long_period = 34
    signal_period = 9
    # Previous candles to get divergence
    diver_candle_range = 15
    # The gap between two pivot high/low
    diver_gap = 4
    # To keep track of the bull / bear divergence
    bull_diver = False
    bear_diver = False
    # risk to reward
    risk_to_reward = 1.5
    qty = 0.1
    btcount = 0

    def __init__(self):
        handler = colorlog.StreamHandler()
        handler.setFormatter(
            colorlog.ColoredFormatter(f"%(log_color)s{Strategy.LOG_FORMAT}")
        )
        file_handler = TimedRotatingFileHandler("y_st003_macd_hist_diver-test.log", when="h")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(Strategy.LOG_FORMAT))
        super().__init__(log_level=logging.INFO, handlers=[handler, file_handler])

    async def set_param(self, identifier, value):
        logging.info(f"Setting {identifier} to {value}")
        if identifier == "atr_length":
            self.atr_length = int(value)
        elif identifier == "short_period":
            self.short_period = int(value)
        elif identifier == "long_period":
            self.long_period = int(value)
        elif identifier == "diver_gap":
            self.diver_gap = int(value)
        elif identifier == "signal_period":
            self.signal_period = int(value)
        elif identifier == "diver_candle_range":
            self.diver_candle_range = int(value)
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
            or len(close) < self.long_period * 3
            or len(close) < self.short_period * 3
            or len(close) < self.signal_period * 3
            or self.diver_candle_range < 6
        ):
            logging.info("Not enough candles to calculate indicators")
            return

        # atr
        atr = cybotrade_indicators.atr(
            high=high, low=low, close=close, period=self.atr_length
        )
        # MACD Histogram divergence
        (macd1, macd2, macd_hist) = cybotrade_indicators.macd(
            real=close,
            short_period=self.short_period,
            long_period=self.long_period,
            signal_period=self.signal_period,
        )
        # Get the latest candles range
        sliced_macd_hist = macd_hist[len(macd_hist) - self.diver_candle_range :]
        sliced_high = high[len(high) - self.diver_candle_range :]
        sliced_low = low[len(low) - self.diver_candle_range :]

        pivot_high_hist_index = []
        pivot_low_hist_index = []
        # Bearlish Divergence
        if np.all(sliced_macd_hist > 0.0):
            for i in range(2, len(sliced_macd_hist)):
                if (
                    sliced_macd_hist[i - 1] > sliced_macd_hist[i - 2]
                    and sliced_macd_hist[i - 1] > sliced_macd_hist[i]
                ):
                    if len(pivot_high_hist_index) == 0:
                        pivot_high_hist_index.append(i - 1)
                    else:
                        if i - 1 - pivot_high_hist_index[-1] >= self.diver_gap:
                            pivot_high_hist_index.append(i - 1)

            if len(pivot_high_hist_index) >= 2:
                for i in range(1, len(pivot_high_hist_index)):
                    if (
                        sliced_macd_hist[pivot_high_hist_index[i - 1]]
                        > sliced_macd_hist[pivot_high_hist_index[i]]
                        and sliced_high[pivot_high_hist_index[i - 1]]
                        < sliced_high[pivot_high_hist_index[i]]
                        and pivot_high_hist_index[i] - pivot_high_hist_index[i - 1]
                        >= self.diver_gap
                    ):
                        self.bear_diver = True

        # Bullish Divergence
        if np.all(sliced_macd_hist < 0.0):
            for i in range(2, len(sliced_macd_hist)):
                if (
                    sliced_macd_hist[i - 1] < sliced_macd_hist[i - 2]
                    and sliced_macd_hist[i - 1] < sliced_macd_hist[i]
                ):
                    if len(pivot_low_hist_index) == 0:
                        pivot_low_hist_index.append(i - 1)
                    else:
                        if i - 1 - pivot_low_hist_index[-1] >= self.diver_gap:
                            pivot_low_hist_index.append(i - 1)

            if len(pivot_low_hist_index) >= 2:
                for i in range(1, len(pivot_low_hist_index)):
                    if (
                        sliced_macd_hist[pivot_low_hist_index[i - 1]]
                        < sliced_macd_hist[pivot_low_hist_index[i]]
                        and sliced_low[pivot_low_hist_index[i - 1]]
                        > sliced_low[pivot_low_hist_index[i]]
                        and pivot_low_hist_index[i] - pivot_low_hist_index[i - 1]
                        >= self.diver_gap
                    ):
                        self.bull_diver = True

        # Once the macd hist change direction, divergence will be invalid
        if self.bull_diver and macd_hist[-1] > 0.0:
            self.bull_diver = False
        if self.bear_diver and macd_hist[-1] < 0.0:
            self.bear_diver = False

        position = await strategy.position(symbol=symbol, exchange=Exchange.BybitLinear)
        # wallet_balance = await strategy.get_current_available_balance(symbol=symbol)
        qty = self.qty

        if (
            position.long.quantity <= 0.0
            and position.short.quantity <= 0.0
            and self.bull_diver
        ):
            stop_loss = low[-1] - atr[-1]
            take_profit = close[-1] + (self.risk_to_reward * (close[-1] - stop_loss))
            await strategy.open(side=OrderSide.Buy, quantity=qty, is_hedge_mode=False, take_profit=take_profit, stop_loss=stop_loss, symbol=symbol, exchange=Exchange.BybitLinear, is_post_only=False)
            logging.info(
                f"Placed a buy order with qty {qty} when close: {close[-1]}, tp: {take_profit}, sl: {stop_loss}, atr: {atr[-1]}, macd_hist: {macd_hist[-1]}, self.bear_diver: {self.bear_diver} at time {start_time[-1]}"
            )

        if (
            position.short.quantity <= 0.0
            and position.long.quantity <= 0.0
            and self.bear_diver
        ):
            stop_loss = high[-1] + atr[-1]
            take_profit = close[-1] - (self.risk_to_reward * (stop_loss - close[-1]))
            await strategy.open(side=OrderSide.Sell, quantity=qty, is_hedge_mode=False, take_profit=take_profit, stop_loss=stop_loss, symbol=symbol, exchange=Exchange.BybitLinear, is_post_only=False)
            logging.info(
                f"Placed a sell order with qty {qty} when close: {close[-1]}, tp: {take_profit}, sl: {stop_loss}, atr: {atr[-1]}, macd_hist: {macd_hist[-1]}, self.bear_diver: {self.bear_diver} at time {start_time[-1]}"
            )

config = RuntimeConfig(
    mode=RuntimeMode.Backtest,
    datasource_topics=[],
    active_order_interval=1,
    initial_capital=10000.0,
    candle_topics=["candles-15m-BTC/USDT-bybit"],
    start_time=datetime(2024, 1, 25, 0, 0, 0, tzinfo=timezone.utc),
    end_time=datetime(2024, 1, 30, 0, 0, 0, tzinfo=timezone.utc),
    api_key="YOUR_CYBOTRADE_API_KEY",
    api_secret="YOUR_CYBOTRADE_API_SECRET",
    data_count=200,
    # exchange_keys="./z_exchange-keys.json",
    )
 
permutation = Permutation(config)
hyper_parameters = {}
hyper_parameters['atr_length'] = [13]
hyper_parameters['short_period'] = [13]
hyper_parameters['long_period'] = [34]
hyper_parameters['signal_period'] = [9]
hyper_parameters['diver_candle_range'] = [15]
hyper_parameters['diver_gap'] = [4]
 
async def start_backtest():
  await permutation.run(hyper_parameters, Strategy)
 
asyncio.run(start_backtest())