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
    # rsi
    rsi_length = 14
    mid_band_rsi = 50
    # Stoch
    k_period = 14
    d_period = 3
    k_slowing_period = 3
    upper_stoch = 80
    lower_stoch = 20
    # MACD
    long_length = 26
    short_length = 12
    signal_length = 9
    # Volume SMA
    sma_length = 30
    # Swing high/low period
    swing_period = 7
    # To keep track of the buy/sell flat
    can_entry_buy = True
    can_entry_sell = True
    # To keep track of the trend in Stoch
    stoch_up_trend = False
    stoch_down_trend = False
    # RR
    risk_to_reward = 2
    qty = 0.1
    btcount = 0

    def __init__(self):
        handler = colorlog.StreamHandler()
        handler.setFormatter(
            colorlog.ColoredFormatter(f"%(log_color)s{Strategy.LOG_FORMAT}")
        )
        file_handler = TimedRotatingFileHandler("y_ste005_rsi_macd_stoch-test.log", when="h")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(Strategy.LOG_FORMAT))
        super().__init__(log_level=logging.INFO, handlers=[handler, file_handler])

    async def set_param(self, identifier, value):
        logging.info(f"Setting {identifier} to {value}")
        if identifier == "rsi_length":
            self.rsi_length = int(value)
        elif identifier == "mid_band_rsi":
            self.mid_band_rsi = int(value)
        elif identifier == "k_period":
            self.k_period = int(value)
        elif identifier == "d_period":
            self.d_period = int(value)
        elif identifier == "k_slowing_period":
            self.k_slowing_period = int(value)
        elif identifier == "long_length":
            self.long_length = int(value)
        elif identifier == "short_length":
            self.short_length = int(value)
        elif identifier == "signal_length":
            self.signal_length = int(value)
        elif identifier == "sma_length":
            self.sma_length = int(value)
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
            len(close) < self.rsi_length * 3
            or len(close) < self.k_period * 3
            or len(close) < self.d_period * 3
            or len(close) < self.k_slowing_period * 3
            or len(close) < self.long_length * 3
            or len(close) < self.short_length * 3
            or len(close) < self.signal_length * 3
        ):
            logging.info("Not enough candles to calculate indicators")
            return

        # volume sma
        sma_vol = talib.SMA(volume, self.sma_length)
        # rsi
        rsi = cybotrade_indicators.rsi(real=close, period=self.rsi_length)
        # Stoch
        stoch_data = cybotrade_indicators.stoch(
            high=high,
            low=low,
            close=close,
            k_period=self.k_period,
            k_slowing_period=self.k_slowing_period,
            d_period=self.d_period,
        )
        stoch = stoch_data[0]
        stoch_ma = stoch_data[1]
        # MACD
        macd_data = talib.MACD(
            close, self.short_length, self.long_length, self.signal_length
        )
        macd_line = macd_data[0]
        signal_line = macd_data[1]

        position = await strategy.position(symbol=symbol, exchange=Exchange.BybitLinear)
        # wallet_balance = await strategy.get_current_available_balance(symbol=symbol)
        qty = self.qty

        # both stoch lines above upper band => up trend
        # both sotch liners below lower band => down trend
        if stoch[-1] < self.lower_stoch and stoch_ma[-1] < self.lower_stoch:
            self.stoch_up_trend = True
            self.stoch_down_trend = False
        if stoch[-1] > self.upper_stoch and stoch_ma[-1] > self.upper_stoch:
            self.stoch_up_trend = False
            self.stoch_down_trend = True

        # Check stoch first, oversold(20)
        # if stoch true, rsi must cross above mid band(50)
        # macd => blue > orange = bull and both lines below 0
        # macd and rsi need meet condition before stoch change to overbuy
        # check volume > ma_vol and green volmue
        if (
            position.long.quantity <= 0.0
            and position.short.quantity <= 0.0
            and self.stoch_up_trend
            and rsi[-1] > self.mid_band_rsi
            and macd_line[-1] > signal_line[-1]
            and macd_line[-2] < signal_line[-2]
            and macd_line[-1] < 0.0
            and signal_line[-1] < 0.0
            and volume[-1] > sma_vol[-1]
            and close[-1] > open[-1]
        ):
            stop_loss = min(low[len(low) - self.swing_period : len(low)])
            take_profit = close[-1] + (self.risk_to_reward * (close[-1] - stop_loss))
            await strategy.open(side=OrderSide.Buy, quantity=qty, is_hedge_mode=False, take_profit=take_profit, stop_loss=stop_loss, symbol=symbol, exchange=Exchange.BybitLinear, is_post_only=False)
            logging.info(
                f"Placed a buy order with qty {qty} when close: {close[-1]}, tp: {take_profit}, sl: {stop_loss} at time {start_time[-1]}"
            )

        # Check stoch first, overbuy(80)
        # if stoch true, rsi must cross below mid band(50)
        # macd => blue < orange = bear and both lines above 0
        # macd and rsi need meet condition before stoch change to oversold
        # check volume > ma_vol and red volmue
        if (
            position.short.quantity <= 0.0
            and position.long.quantity <= 0.0
            and self.stoch_down_trend
            and rsi[-1] < self.mid_band_rsi
            and macd_line[-1] < signal_line[-1]
            and macd_line[-2] > signal_line[-2]
            and macd_line[-1] > 0.0
            and signal_line[-1] > 0.0
            and volume[-1] > sma_vol[-1]
            and close[-1] < open[-1]
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
    candle_topics=["candles-5m-BTC/USDT-bybit"],
    start_time=datetime(2024, 1, 25, 0, 0, 0, tzinfo=timezone.utc),
    end_time=datetime(2024, 1, 30, 0, 0, 0, tzinfo=timezone.utc),
    api_key="YOUR_CYBOTRADE_API_KEY",
    api_secret="YOUR_CYBOTRADE_API_SECRET",
    data_count=500,
    # exchange_keys="./z_exchange-keys.json",
    )

permutation = Permutation(config)
hyper_parameters = {}
hyper_parameters['rsi_length'] = [14]
hyper_parameters['mid_band_rsi'] = [50]
hyper_parameters['k_period'] = [14]
hyper_parameters['d_period'] = [3]
hyper_parameters['k_slowing_period'] = [3]
hyper_parameters['long_length'] = [26]
hyper_parameters['short_length'] = [12]
hyper_parameters['signal_length'] = [9]
hyper_parameters['sma_length'] = [30]
 
async def start_backtest():
  await permutation.run(hyper_parameters, Strategy)
 
asyncio.run(start_backtest())