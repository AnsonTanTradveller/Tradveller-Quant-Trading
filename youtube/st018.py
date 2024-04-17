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
    # QQE
    rsi_length1 = 6
    rsi_smoothing1 = 5
    QQE_factor1 = 3
    threshold1 = 3
    rsi_length2 = 6
    rsi_smoothing2 = 5
    QQE_factor2 = 1.61
    threshold2 = 3
    bollinger_length = 50
    QQE_multiplier = 0.35
    # Supertrend
    atr_period = 9
    factor = 3.9
    # Trend Indicator A-V2
    ema_length = 52
    ema_length_smoothing = 10
    # risk to reward
    risk_to_reward = 1.5
    qty = 0.1
    btcount = 0

    def __init__(self):
        handler = colorlog.StreamHandler()
        handler.setFormatter(
            colorlog.ColoredFormatter(f"%(log_color)s{Strategy.LOG_FORMAT}")
        )
        file_handler = TimedRotatingFileHandler("y_st018_supertrend_qqe_A-test.log", when="h")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(Strategy.LOG_FORMAT))
        super().__init__(log_level=logging.INFO, handlers=[handler, file_handler])

    async def set_param(self, identifier, value):
        logging.info(f"Setting {identifier} to {value}")
        if identifier == "rsi_length1":
            self.rsi_length1 = int(value)
        elif identifier == "rsi_smoothing1":
            self.rsi_smoothing1 = int(value)
        elif identifier == "QQE_factor1":
            self.QQE_factor1 = float(value)
        elif identifier == "threshold1":
            self.threshold1 = float(value)
        elif identifier == "rsi_length2":
            self.rsi_length2 = int(value)
        elif identifier == "rsi_smoothing2":
            self.rsi_smoothing2 = int(value)
        elif identifier == "QQE_factor2":
            self.QQE_factor2 = float(value)
        elif identifier == "threshold2":
            self.threshold2 = float(value)
        elif identifier == "bollinger_length":
            self.bollinger_length = int(value)
        elif identifier == "QQE_multiplier":
            self.QQE_multiplier = float(value)
        elif identifier == "atr_period":
            self.atr_period = int(value)
        elif identifier == "factor":
            self.factor = float(value)
        elif identifier == "ema_length":
            self.ema_length = int(value)
        elif identifier == "ema_length_smoothing":
            self.ema_length_smoothing = int(value)
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
            len(close) < self.rsi_length1 * 3
            or len(close) < self.rsi_smoothing1 * 3
            or len(close) < self.rsi_length2 * 3
            or len(close) < self.rsi_smoothing2 * 3
            or len(close) < self.bollinger_length * 3
            or len(close) < self.atr_period * 3
            or len(close) < self.ema_length * 3
        ):
            logging.info("Not enough candles to calculate indicators")
            return

        # Supertred
        supertrend_data = cybotrade_indicators.supertrend(
            high=high,
            low=low,
            close=close,
            period=self.atr_period,
            factor=float(self.factor),
        )
        supertrend_signal = supertrend_data[3]
        # QQE mod
        qqe_data = cybotrade_indicators.qqe_mod(
            close=close,
            rsi_length_one=self.rsi_length1,
            rsi_smoothing_one=self.rsi_smoothing1,
            qqe_factor_one=self.QQE_factor1,
            threshold_one=self.threshold1,
            rsi_length_two=self.rsi_length2,
            rsi_smoothing_two=self.rsi_smoothing2,
            qqe_factor_two=self.QQE_factor2,
            threshold_two=self.threshold2,
            bollinger_length=self.bollinger_length,
            qqe_multiplier=self.QQE_multiplier,
        )
        qqe_greenbar = qqe_data[0]
        qqe_redbar = qqe_data[1]
        # Trend indicator A-V2
        ema_open = cybotrade_indicators.ema(open, self.ema_length)
        ema_high = cybotrade_indicators.ema(high, self.ema_length)
        ema_low = cybotrade_indicators.ema(low, self.ema_length)
        ema_close = cybotrade_indicators.ema(close, self.ema_length)
        ha_ohlc = cybotrade_indicators.ha(
            open=ema_open, high=ema_high, low=ema_low, close=ema_close
        )
        ha_open = ha_ohlc[0]
        ha_high = ha_ohlc[1]
        ha_low = ha_ohlc[2]
        ha_close = ha_ohlc[3]
        ema_ha_open = cybotrade_indicators.ema(ha_open, self.ema_length_smoothing)
        ema_ha_high = cybotrade_indicators.ema(ha_high, self.ema_length_smoothing)
        ema_ha_low = cybotrade_indicators.ema(ha_low, self.ema_length_smoothing)
        ema_ha_close = cybotrade_indicators.ema(ha_close, self.ema_length_smoothing)
        color_trend = np.array(list([0.0] * len(close)))
        for i in range(0, len(close)):
            color_trend[i] = 1 if ema_ha_close[i] >= ema_ha_open[i] else -1

        position = await strategy.position(symbol=symbol, exchange=Exchange.BybitLinear)
        # wallet_balance = await strategy.get_current_available_balance(symbol=symbol)
        qty = self.qty

        # When supertrend change trend then will close position if have position
        # if position.long.quantity != 0.0 and supertrend_signal[-1] == -1:
        #     await strategy.close(side=OrderSide.Buy,quantity=position.long.quantity,symbol=symbol)
        #     logging.info(
        #         f"Placed a close long position with qty {qty} at time {start_time[-1]}"
        #     )

        # if position.short.quantity != 0.0 and supertrend_signal[-1] == 1:
        #     await strategy.close(side=OrderSide.Sell,quantity=position.short.quantity,symbol=symbol)
        #     logging.info(
        #         f"Placed a close short position with qty {qty} at time {start_time[-1]}"
        #     )

        # supertrend buy signal and qqe_mod green bar and Trend A-V2 is green line
        if (
            supertrend_signal[-1] == 1
            and qqe_greenbar[-1] == 1
            and color_trend[-1] == 1
            and position.long.quantity <= 0.0
        ):
            stop_loss = ema_ha_low[-1]
            take_profit = close[-1] + (self.risk_to_reward * (close[-1] - stop_loss))
            await strategy.open(side=OrderSide.Buy, quantity=qty, is_hedge_mode=False, take_profit=take_profit, stop_loss=stop_loss, symbol=symbol, exchange=Exchange.BybitLinear, is_post_only=False)
            logging.info(
                f"Placed a buy order with qty {qty} when close: {close[-1]}, tp: {take_profit}, sl: {stop_loss} at time {start_time[-1]}"
            )

        # supertrend sell signal and qqe_mod red bar and Trend A-V2 is red line
        if (
            supertrend_signal[-1] == -1
            and qqe_redbar[-1] == -1
            and color_trend[-1] == -1
            and position.short.quantity <= 0.0
        ):
            stop_loss = ema_ha_high[-1]
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
hyper_parameters['rsi_length1'] = [6]
hyper_parameters['rsi_smoothing1'] = [5]
hyper_parameters['QQE_factor1'] = [3]
hyper_parameters['threshold1'] = [3]
hyper_parameters['rsi_length2'] = [6]
hyper_parameters['rsi_smoothing2'] = [5]
hyper_parameters['QQE_factor2'] = [1.61]
hyper_parameters['threshold2'] = [3]
hyper_parameters['bollinger_length'] = [50]
hyper_parameters['QQE_multiplier'] = [0.35]
hyper_parameters['atr_period'] = [9]
hyper_parameters['factor'] = [3.9]
hyper_parameters['ema_length'] = [52]
hyper_parameters['ema_length_smoothing'] = [10]
 
async def start_backtest():
  await permutation.run(hyper_parameters, Strategy)
 
asyncio.run(start_backtest())