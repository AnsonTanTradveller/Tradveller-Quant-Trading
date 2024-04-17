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
    # 3 EMA
    fast_ema_length = 25
    medium_ema_length = 50
    slow_ema_length = 100
    ema_up_trend = False
    ema_down_trend = False
    # MACD
    fast_length = 12
    slow_length = 26
    signal_length = 9
    hist_up_trend = False
    hist_down_trend = False
    # PSAR
    accel_max = 0.2
    accel_start = 0.02
    accel_step = 0.02
    # rsi
    rsi_length = 14
    upper_bound_rsi = 80
    lower_bound_rsi = 20
    # Swing high/low period
    swing_period = 5
    # risk_to_reward
    risk_to_reward = 2
    qty = 0.1
    btcount = 0

    def __init__(self):
        handler = colorlog.StreamHandler()
        handler.setFormatter(
            colorlog.ColoredFormatter(f"%(log_color)s{Strategy.LOG_FORMAT}")
        )
        file_handler = TimedRotatingFileHandler("y_ste013_macd_psar_ema-test.log", when="h")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(Strategy.LOG_FORMAT))
        super().__init__(log_level=logging.INFO, handlers=[handler, file_handler])

    async def set_param(self, identifier, value):
        logging.info(f"Setting {identifier} to {value}")
        if identifier == "fast_ema_length":
            self.fast_ema_length = int(value)
        elif identifier == "medium_ema_length":
            self.medium_ema_length = int(value)
        elif identifier == "slow_ema_length":
            self.slow_ema_length = int(value)
        elif identifier == "fast_length":
            self.fast_length = int(value)
        elif identifier == "slow_length":
            self.slow_length = int(value)
        elif identifier == "signal_length":
            self.signal_length = int(value)
        elif identifier == "accel_max":
            self.accel_max = float(value)
        elif identifier == "accel_start":
            self.accel_start = float(value)
        elif identifier == "accel_step":
            self.accel_step = float(value)
        elif identifier == "rsi_length":
            self.rsi_length = int(value)
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
            len(close) < self.fast_ema_length * 3
            or len(close) < self.medium_ema_length * 3
            or len(close) < self.rsi_length * 3
            or len(close) < self.slow_ema_length * 3
            or len(close) < self.fast_length * 3
            or len(close) < self.slow_length * 3
            or len(close) < self.signal_length * 3
        ):
            logging.info("Not enough candles to calculate indicators")
            return

        # 3 EMA
        fast_ema = cybotrade_indicators.ema(close, self.fast_ema_length)
        medium_ema = cybotrade_indicators.ema(close, self.medium_ema_length)
        slow_ema = cybotrade_indicators.ema(close, self.slow_ema_length)

        # MACD
        macd_data = talib.MACD(
            close, self.slow_length, self.fast_length, self.signal_length
        )
        macd_hist = macd_data[2]
        # MACD histogram bar color, deep green/red only can identify trend
        if macd_hist[-1] >= 0.0:
            if macd_hist[-2] < macd_hist[-1]:
                self.hist_up_trend = True
                self.hist_down_trend = False
        else:
            if macd_hist[-2] > macd_hist[-1]:
                self.hist_down_trend = True
                self.hist_up_trend = False
        # PSAR
        psar = cybotrade_indicators.psar(
            high=high,
            low=low,
            close=close,
            accel_max=self.accel_max,
            accel_start=self.accel_start,
            accel_step=self.accel_step,
        )
        # rsi
        rsi = talib.RSI(close, self.rsi_length)

        position = await strategy.position(symbol=symbol, exchange=Exchange.BybitLinear)
        # wallet_balance = await strategy.get_current_available_balance(symbol=symbol)
        qty = self.qty

        # 3 ema up trend => fast > medium & medium > slow & medium[-2] < slow[-2]
        if (
            fast_ema[-1] > medium_ema[-1]
            and medium_ema[-1] > slow_ema[-1]
            and (
                medium_ema[-2] < slow_ema[-2]
                or fast_ema[-2] < medium_ema[-2]
                or fast_ema[-2] < slow_ema[-2]
            )
        ):
            self.ema_up_trend = True
            self.ema_down_trend = False
        # 3 ema down trend => fast < medium & medium < slow & medium[-2] > slow[-2]
        if (
            fast_ema[-1] < medium_ema[-1]
            and medium_ema[-1] < slow_ema[-1]
            and (
                medium_ema[-2] > slow_ema[-2]
                or fast_ema[-2] > medium_ema[-2]
                or fast_ema[-2] > slow_ema[-2]
            )
        ):
            self.ema_up_trend = False
            self.ema_down_trend = True

        # Check rsi got hit overbought / oversold during ema up/down trend is true
        if self.ema_up_trend and rsi[-1] >= self.upper_bound_rsi:
            self.ema_up_trend = False
        elif self.ema_down_trend and rsi[-1] <= self.lower_bound_rsi:
            self.ema_down_trend = False

        # 3 ema up trend
        # Once 3ema turn to up trend, the rsi must cant > 80
        # psar up trend => psar < high & psar[-2] > low
        # macd hist is dark green
        # candle open and close above the fast_ema_line
        # stop loss = swing low
        if (
            self.ema_up_trend
            and psar[-1] < high[-1]
            and psar[-2] > low[-1]
            and self.hist_up_trend
            and low[-1] > fast_ema[-1]
            and position.long.quantity <= 0.0
            and position.short.quantity <= 0.0
        ):
            stop_loss = min(low[len(low) - self.swing_period : len(low)])
            take_profit = close[-1] + (self.risk_to_reward * (close[-1] - stop_loss))
            await strategy.open(side=OrderSide.Buy, quantity=qty, is_hedge_mode=False, take_profit=take_profit, stop_loss=stop_loss, symbol=symbol, exchange=Exchange.BybitLinear, is_post_only=False)
            logging.info(
                f"Placed a buy order with qty {qty} when close: {close[-1]}, tp: {take_profit}, sl: {stop_loss} at time {start_time[-1]}"
            )

        # 3 ema down trend
        # Once 3ema turn to down trend, the rsi must cant < 20
        # psar down trend => psar > low & psar[-2] < high
        # macd hist is dark red
        # candle open and close below the fast_ema_line
        # stop loss = swing high
        if (
            self.ema_down_trend
            and psar[-1] > low[-1]
            and psar[-2] < high[-1]
            and self.hist_down_trend
            and high[-1] < fast_ema[-1]
            and position.long.quantity <= 0.0
            and position.short.quantity <= 0.0
        ):
            stop_loss = max(high[len(high) - self.swing_period : len(high)])
            take_profit = close[-1] - (self.risk_to_reward * (stop_loss - close[-1]))
            await strategy.open(side=OrderSide.Sell, quantity=qty, is_hedge_mode=False, take_profit=take_profit, stop_loss=stop_loss, symbol=symbol, exchange=Exchange.BybitLinear, is_post_only=False)
            logging.info(
                f"Placed a sell order with qty {qty} when close: {close[-1]}, tp: {take_profit}, sl: {stop_loss} at time {start_time[-1]}"
            )

config = RuntimeConfig(
    mode=RuntimeMode.Backtest,
    datasource_topics=[],
    active_order_interval=1,
    initial_capital=10000.0,
    candle_topics=["candles-15m-BTC/USDT-bybit"],
    start_time=datetime(2024, 1, 20, 0, 0, 0, tzinfo=timezone.utc),
    end_time=datetime(2024, 1, 30, 0, 0, 0, tzinfo=timezone.utc),
    api_key="YOUR_CYBOTRADE_API_KEY",
    api_secret="YOUR_CYBOTRADE_API_SECRET",
    data_count=700,
    # exchange_keys="./z_exchange-keys.json",
    )

permutation = Permutation(config)
hyper_parameters = {}
hyper_parameters['fast_ema_length'] = [25]
hyper_parameters['medium_ema_length'] = [50]
hyper_parameters['slow_ema_length'] = [100]
hyper_parameters['fast_length'] = [12]
hyper_parameters['slow_length'] = [26]
hyper_parameters['signal_length'] = [9]
hyper_parameters['accel_max'] = [0.2]
hyper_parameters['accel_start'] = [0.02]
hyper_parameters['accel_step'] = [0.02]
hyper_parameters['rsi_length'] = [14]
 
async def start_backtest():
  await permutation.run(hyper_parameters, Strategy)
 
asyncio.run(start_backtest())