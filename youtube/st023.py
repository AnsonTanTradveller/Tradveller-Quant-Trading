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
    # Supertrend
    atr_period = 10
    factor = 3
    # Slow Stochastic
    ss_length = 10
    smooth_k = 6
    smooth_d = 5
    # CCI/StochRsi/EMA
    cci_length = 20
    stoch_rsi_length = 14
    stoch_rsi_smoothk = 6
    stoch_rsi_smoothd = 6
    stoch_rsi_upper = 80
    stoch_rsi_lower = 20
    cci_upper_trigger_line = 100
    cci_lower_trigger_line = -100
    # Swing high/low period
    swing_period = 10
    # RR
    risk_to_reward = 1.5
    qty = 0.1
    btcount = 0

    def __init__(self):
        handler = colorlog.StreamHandler()
        handler.setFormatter(
            colorlog.ColoredFormatter(f"%(log_color)s{Strategy.LOG_FORMAT}")
        )
        file_handler = TimedRotatingFileHandler("y_st023_strategy-test.log", when="h")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(Strategy.LOG_FORMAT))
        super().__init__(log_level=logging.INFO, handlers=[handler, file_handler])

    async def set_param(self, identifier, value):
        logging.info(f"Setting {identifier} to {value}")
        if identifier == "atr_period":
            self.atr_period = int(value)
        elif identifier == "factor":
            self.factor = float(value)
        elif identifier == "ss_length":
            self.ss_length = int(value)
        elif identifier == "smooth_k":
            self.smooth_k = int(value)
        elif identifier == "smooth_d":
            self.smooth_d = int(value)
        elif identifier == "cci_length":
            self.cci_length = int(value)
        elif identifier == "stoch_rsi_length":
            self.stoch_rsi_length = int(value)
        elif identifier == "stoch_rsi_smoothk":
            self.stoch_rsi_smoothk = int(value)
        elif identifier == "stoch_rsi_smoothd":
            self.stoch_rsi_smoothd = int(value)
        else:
            logging.error(f"Could not set {identifier}, not found")

    def calculate_rolling_mean_ad(self, data, window_size):
        rolling_mad = [0.0] * (window_size - 1)
        for i in range(len(data) - window_size + 1):
            window = data[i : i + window_size]
            mean = np.mean(window)
            mad = np.mean(np.abs(window - mean))
            rolling_mad.append(mad)
        return rolling_mad

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
            len(close) < self.atr_period * 3
            or len(close) < self.ss_length * 3
            or len(close) < self.stoch_rsi_length * 3
            or len(close) < self.cci_length * 3
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
        direction = supertrend_data[1]
        supertrend = supertrend_data[0]
        supertrend_signal = supertrend_data[3]
        super_up_trend = np.array(list([0.0] * len(close)))
        super_down_trend = np.array(list([0.0] * len(close)))
        for i in range(0, len(direction)):
            if direction[i] < 0:
                super_up_trend[i] = supertrend[i]
            elif direction[i] > 0:
                super_down_trend[i] = supertrend[i]

        # Stoch
        stoch_data = cybotrade_indicators.stoch(
            high=high,
            low=low,
            close=close,
            k_period=self.ss_length,
            k_slowing_period=self.smooth_k,
            d_period=self.smooth_d,
        )
        stoch = stoch_data[0]
        stoch_ma = stoch_data[1]

        # CCI/StochRsi/EMA
        # CCI
        sma_cci = cybotrade_indicators.sma(real=close, period=self.cci_length)
        for i in range(0, self.cci_length - 1):
            sma_cci = np.insert(sma_cci, 0, 0)
        mad_typical_price = np.array(
            self.calculate_rolling_mean_ad(close, self.cci_length)
        )

        cci = close - sma_cci
        dev = 0.015 * mad_typical_price
        cci_line = []
        for i in range(0, len(cci)):
            if dev[i] == 0.0:
                cci_line.append(0.0)
            else:
                cci_line.append(cci[i] / dev[i])

        # Stoch rsi
        (stochrsi_k, stochrsi_d) = cybotrade_indicators.stoch_rsi(
            close=close,
            smoothk=self.stoch_rsi_smoothk,
            smoothd=self.stoch_rsi_smoothd,
            rsi_period=self.stoch_rsi_length,
            stoch_period=self.stoch_rsi_length,
        )
        sliced_stochrsi_k = stochrsi_k[len(stochrsi_k) - self.cci_length :]
        sliced_cci_line = cci_line[len(cci_line) - self.cci_length :]
        candle_color = np.array(list([0.0] * len(sliced_stochrsi_k)))
        for i in range(0, len(sliced_stochrsi_k)):
            if (
                sliced_cci_line[i] >= self.cci_upper_trigger_line
                and sliced_stochrsi_k[i] >= self.stoch_rsi_upper
            ):
                candle_color[i] = -1  # Yellow (Short)
            elif (
                sliced_cci_line[i] <= self.cci_lower_trigger_line
                and sliced_stochrsi_k[i] <= self.stoch_rsi_lower
            ):
                candle_color[i] = 1  # White (Long)

        position = await strategy.position(symbol=symbol, exchange=Exchange.BybitLinear)
        # wallet_balance = await strategy.get_current_available_balance(symbol=symbol)
        qty = self.qty

        # supertrend up trend
        # CCI/StochRsi/EMA white color bar
        # stoch k > d
        # next candle is green
        if (
            position.long.quantity <= 0.0
            and super_up_trend[-2] > 0.0
            and candle_color[-2] == 1
            and candle_color[-1] != 1
            and stoch[-2] > stoch_ma[-2]
            and close[-1] > open[-1]
        ):
            stop_loss = min(
                min(low[len(low) - self.swing_period : len(low)]), super_up_trend[-2]
            )
            take_profit = close[-1] + (self.risk_to_reward * (close[-1] - stop_loss))
            await strategy.open(side=OrderSide.Buy, quantity=qty, is_hedge_mode=False, take_profit=take_profit, stop_loss=stop_loss, symbol=symbol, exchange=Exchange.BybitLinear, is_post_only=False)
            logging.info(
                f"Placed a buy order with qty {qty} when close: {close[-1]}, tp: {take_profit}, sl: {stop_loss} at time {start_time[-1]}"
            )

        # supertrend down trend > 0
        # CCI/StochRsi/EMA yellow color bar == -1
        # stoch k < d
        # next candle is red
        if (
            position.short.quantity <= 0.0
            and super_down_trend[-2] > 0.0
            and candle_color[-2] == -1
            and candle_color[-1] != -1
            and stoch[-2] < stoch_ma[-2]
            and close[-1] < open[-1]
        ):
            stop_loss = max(
                max(high[len(high) - self.swing_period : len(high)]),
                super_down_trend[-2],
            )
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
    data_count=500,
    # exchange_keys="./z_exchange-keys.json",
    )

permutation = Permutation(config)
hyper_parameters = {}
hyper_parameters['atr_period'] = [10]
hyper_parameters['factor'] = [3]
hyper_parameters['ss_length'] = [10]
hyper_parameters['smooth_k'] = [6]
hyper_parameters['smooth_d'] = [5]
hyper_parameters['cci_length'] = [20]
hyper_parameters['stoch_rsi_length'] = [14]
hyper_parameters['stoch_rsi_smoothk'] = [6]
hyper_parameters['stoch_rsi_smoothd'] = [6]
 
async def start_backtest():
  await permutation.run(hyper_parameters, Strategy)
 
asyncio.run(start_backtest())