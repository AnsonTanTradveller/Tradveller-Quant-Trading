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
    # EMA
    ema_length = 200
    # Stoch Rsi
    smooth_k = 3
    smooth_d = 3
    stoch_rsi_length = 14
    stoch_length = 14
    stochrsi_up_trend = False
    stochrsi_down_trend = False
    upper_bound_stochrsi = 80
    lower_bound_stochrsi = 20
    # Swing high/low period
    swing_period = 10
    # risk_to_reward
    risk_to_reward = 1
    qty = 0.1
    btcount = 0

    def __init__(self):
        handler = colorlog.StreamHandler()
        handler.setFormatter(
            colorlog.ColoredFormatter(f"%(log_color)s{Strategy.LOG_FORMAT}")
        )
        file_handler = TimedRotatingFileHandler("y_st004_ha_ema_stochrsi-test.log", when="h")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(Strategy.LOG_FORMAT))
        super().__init__(log_level=logging.INFO, handlers=[handler, file_handler])

    async def set_param(self, identifier, value):
        logging.info(f"Setting {identifier} to {value}")
        if identifier == "ema_length":
            self.ema_length = int(value)
        elif identifier == "smooth_k":
            self.smooth_k = int(value)
        elif identifier == "smooth_d":
            self.smooth_d = int(value)
        elif identifier == "stoch_rsi_length":
            self.stoch_rsi_length = int(value)
        elif identifier == "stoch_length":
            self.stoch_length = int(value)
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
            len(close) < self.ema_length * 3
            or len(close) < self.stoch_rsi_length * 3
            or len(close) < self.stochrsi_up_trend * 3
        ):
            logging.info("Not enough candles to calculate indicators")
            return

        # HA
        (ha_open, ha_high, ha_low, ha_close) = cybotrade_indicators.ha(
            open=open, high=high, low=low, close=close
        )
        # EMA
        ema = cybotrade_indicators.ema(ha_close, self.ema_length)

        # Stoch Rsi
        (stochrsi_fast, stochrsi_slow) = cybotrade_indicators.stoch_rsi(
            close=ha_close,
            rsi_period=self.stoch_rsi_length,
            stoch_period=self.stoch_length,
            smoothd=self.smooth_d,
            smoothk=self.smooth_k,
        )
        # stoch rsi golden cross / death cross to identify up/down trend
        # up trend => if fast line > upper bound no more up trend
        # down trend => if fast line < lower bound no more down trend
        if (
            stochrsi_fast[-1] < self.lower_bound_stochrsi
            and stochrsi_fast[-1] > stochrsi_slow[-1]
            and stochrsi_fast[-2] < stochrsi_slow[-2]
        ):
            self.stochrsi_up_trend = True
            self.stochrsi_down_trend = False
        if (
            stochrsi_fast[-1] > self.upper_bound_stochrsi
            and stochrsi_fast[-1] < stochrsi_slow[-1]
            and stochrsi_fast[-2] > stochrsi_slow[-2]
        ):
            self.stochrsi_up_trend = False
            self.stochrsi_down_trend = True

        if stochrsi_fast[-1] > self.upper_bound_stochrsi and self.stochrsi_up_trend:
            self.stochrsi_up_trend = False
        if stochrsi_fast[-1] < self.lower_bound_stochrsi and self.stochrsi_down_trend:
            self.stochrsi_down_trend = False

        position = await strategy.position(symbol=symbol, exchange=Exchange.BybitLinear)
        # wallet_balance = await strategy.get_current_available_balance(symbol=symbol)
        qty = self.qty

        # ha_close close above ema
        # fast > slow and prev_fast < prev_slow and up trend
        # ha_open = ha_low and ha_close > ha_open and ha_high > prev ha_high
        # stop loss = swing low
        if (
            self.stochrsi_up_trend
            and ha_low[-1] > ema[-1]
            and ha_close[-1] > ha_open[-1]
            and ha_open[-1] == ha_low[-1]
            and ha_high[-1] > ha_high[-2]
            and position.long.quantity <= 0.0
            and position.short.quantity <= 0.0
        ):
            stop_loss = min(low[len(low) - self.swing_period : len(low)])
            take_profit = close[-1] + (self.risk_to_reward * (close[-1] - stop_loss))
            await strategy.open(side=OrderSide.Buy, quantity=qty, is_hedge_mode=False, take_profit=take_profit, stop_loss=stop_loss, symbol=symbol, exchange=Exchange.BybitLinear, is_post_only=False)
            logging.info(
                f"Placed a buy order with qty {qty} when close: {close[-1]}, tp: {take_profit}, sl: {stop_loss}, ha_close: {ha_close[-1]}, ha_open: {ha_open[-1]}, ha_low: {ha_low[-1]}, ema: {ema[-1]}, self.stochrsi_down_trend: {self.stochrsi_down_trend} at time {start_time[-1]}"
            )

        # ha_close below above ema
        # fast < slow and prev_fast > prev_slow and down trend
        # ha_open = ha_high and ha_close < ha_open and ha_low < prev ha_low
        # stop loss = swing high
        if (
            self.stochrsi_down_trend
            and ha_high[-1] < ema[-1]
            and ha_close[-1] < ha_open[-1]
            and ha_open[-1] == ha_high[-1]
            and ha_low[-1] < ha_low[-2]
            and position.long.quantity <= 0.0
            and position.short.quantity <= 0.0
        ):
            stop_loss = max(high[len(high) - self.swing_period : len(high)])
            take_profit = close[-1] - (self.risk_to_reward * (stop_loss - close[-1]))
            await strategy.open(side=OrderSide.Sell, quantity=qty, is_hedge_mode=False, take_profit=take_profit, stop_loss=stop_loss, symbol=symbol, exchange=Exchange.BybitLinear, is_post_only=False)
            logging.info(
                f"Placed a sell order with qty {qty} when close: {close[-1]}, tp: {take_profit}, sl: {stop_loss}, ha_close: {ha_close[-1]}, ha_open: {ha_open[-1]}, ha_low: {ha_low[-1]}, ema: {ema[-1]}, self.stochrsi_down_trend: {self.stochrsi_down_trend} at time {start_time[-1]}"
            )

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
    data_count=700,
    # exchange_keys="./z_exchange-keys.json",
    )
 
permutation = Permutation(config)
hyper_parameters = {}
hyper_parameters['ema_length'] = [200]
hyper_parameters['smooth_k'] = [3]
hyper_parameters['smooth_d'] = [3]
hyper_parameters['stoch_rsi_length'] = [14]
hyper_parameters['stoch_length'] = [14]
 
async def start_backtest():
  await permutation.run(hyper_parameters, Strategy)
 
asyncio.run(start_backtest())
