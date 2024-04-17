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
from logging.handlers import TimedRotatingFileHandler
from cybotrade.permutation import Permutation
import math
import numpy as np
import talib
import asyncio
import logging
import colorlog
import cybotrade_indicators

class Strategy(BaseStrategy):
    # Indicator params
    # Echo Forecast
    evalution_window = 50
    forecast_window = 50
    # SSL Hybrid
    ssl_ma_length = 200
    ssl_base_channel_multiplier = 0.2
    first_ssl_ma = 0.0
    prev_ssl_ma = 0.0
    current_ssl = 0.0
    ssl_color = 0
    # Volume oscilator
    short_length = 5
    long_length = 10
    # RR
    risk_to_reward = 1.5
    # Swing high/low period
    swing_period = 10
    qty = 0.1
    btcount = 0

    def __init__(self):
        handler = colorlog.StreamHandler()
        handler.setFormatter(
            colorlog.ColoredFormatter(f"%(log_color)s{Strategy.LOG_FORMAT}")
        )
        file_handler = TimedRotatingFileHandler("y_st000_echo_vo_ssl-test.log", when="h")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(Strategy.LOG_FORMAT))
        super().__init__(log_level=logging.INFO, handlers=[handler, file_handler])

    async def set_param(self, identifier, value):
        logging.info(f"Setting {identifier} to {value}")
        if identifier == "evalution_window":
            self.evalution_window = int(value)
        elif identifier == "forecast_window":
            self.forecast_window = int(value)
        elif identifier == "ssl_ma_length":
            self.ssl_ma_length = int(value)
        elif identifier == "ssl_base_channel_multiplier":
            self.ssl_base_channel_multiplier = float(value)
        elif identifier == "short_length":
            self.short_length = int(value)
        elif identifier == "long_length":
            self.long_length = int(value)
        else:
            logging.error(f"Could not set {identifier}, not found")

    def get_mean(self, array):
        total = 0
        for i in range(0, len(array)):
            total += array[i]
        return total / len(array)

    def get_stddev(self, array):
        total = 0
        mean = self.get_mean(array)
        for i in range(0, len(array)):
            minus_mean = math.pow(array[i] - mean, 2)
            total += minus_mean

        return math.sqrt(total / (len(array) - 1))

    def get_covariance(self, array_x, array_y):
        total = 0
        mean_x = self.get_mean(array_x)
        mean_y = self.get_mean(array_y)
        for i in range(0, len(array_y)):
            x = array_x[i] - mean_x
            y = array_y[i] - mean_y
            total += x * y

        return total / (len(array_x) - 1)

    def get_correlation(self, array_x, array_y):
        cov = self.get_covariance(array_x, array_y)
        std_x = self.get_stddev(array_x)
        std_y = self.get_stddev(array_y)

        return cov / (std_x * std_y)

    def calculate_ema(self, data, window):
        ema_values = [0.0]
        multiplier = 2 / (window + 1)  # EMA multiplier
        # Calculate the initial SMA (Simple Moving Average)
        sma = np.mean(data[:window])
        ema_values.append(sma)

        # Calculate the EMA for the remaining data points
        for i in range(window, len(data)):
            ema = (data[i] - ema_values[-1]) * multiplier + ema_values[-1]
            ema_values.append(ema)

        return np.array(ema_values)

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
            len(close) < self.evalution_window * 3
            or len(close) < self.forecast_window * 3
            or len(close) < self.ssl_ma_length * 1
            or len(close) < self.short_length * 3
            or len(close) < self.long_length * 3
        ):
            logging.info("Not enough candles to calculate indicators")
            return

        # Echo forecast
        val = 0.0
        range_length = self.evalution_window + self.forecast_window * 2
        # Slice the last range length close price
        sliced_close = close[len(close) - range_length : len(close)]
        fliped_sliced_close = np.flip(sliced_close)
        # Get the different of close
        d = np.diff(sliced_close)
        d = np.insert(d, 0, 0.0)
        fliped_d = np.flip(d)
        # Get the top and bottom level of forecast window
        top_level = sliced_close.max()
        btm_level = sliced_close.min()
        # Get the reference window data
        ref_window_data = fliped_sliced_close[
            : len(fliped_sliced_close) - self.forecast_window * 2
        ]
        k = [0]
        prev = close[-1]
        current = close[-1]
        future_line = []
        for i in range(0, self.evalution_window):
            eva_window_data = fliped_sliced_close[
                i + self.forecast_window : i + self.forecast_window * 2
            ]
            correlation = self.get_correlation(ref_window_data, eva_window_data)
            if val == 0.0:
                val = correlation
            elif correlation >= val:
                val = correlation
            else:
                val = val
            if val == correlation:
                k.append(i)
            else:
                k.append(k[-1])
        for i in range(0, self.forecast_window):
            current = prev
            e = fliped_d[self.forecast_window + k[-1] + self.forecast_window - i - 1]
            current += e
            future_line.append(current)
            prev = current
        max_future_line = max(future_line)
        min_future_line = min(future_line)
        # Volume Osc
        short_vosc = talib.EMA(volume, self.short_length)
        long_vosc = talib.EMA(volume, self.long_length)
        volume_osc = np.array(list([0.0] * len(close)))
        for i in range(0, len(volume)):
            volume_osc[i] = 100.0 * (short_vosc[i] - long_vosc[i]) / long_vosc[i]

        # SSL Hybrib
        # This McGinley MA Type really depend on when the candles strat
        # For example, if your trading view 1hr BTC candles start from 1/1/2023,
        # then u should start the candles from 1/1 during backtest, if not the SSL ma will not same as the trading view
        # This is because the McGinley is using the first ema after the ma_length given(200), and base on the first ema to
        # calculate the next McGinley MA
        # only get the latest ssl
        # True range. Same as tr(false). It is max(high - low, abs(high - close[1]), abs(low - close[1]))
        tr = np.array(list([0.0] * len(close)))
        for i in range(1, len(close)):
            tr[i] = max(
                max(high[i] - low[i], abs(high[i] - close[i - 1])),
                abs(low[i] - close[i - 1]),
            )
        rangema = cybotrade_indicators.ema(real=tr, period=self.ssl_ma_length)
        if self.first_ssl_ma != 0.0:
            self.current_ssl = self.prev_ssl_ma + (close[-1] - self.prev_ssl_ma) / (
                self.ssl_ma_length * math.pow(close[-1] / self.prev_ssl_ma, 4)
            )
            self.prev_ssl_ma = self.current_ssl
            upperk = self.current_ssl + (rangema[-1] * self.ssl_base_channel_multiplier)
            lowerk = self.current_ssl - (rangema[-1] * self.ssl_base_channel_multiplier)
            if close[-1] > upperk:
                self.ssl_color = 1
            elif close[-1] < lowerk:
                self.ssl_color = -1
            else:
                self.ssl_color = 0
        else:
            ssl_length_close = close[0:self.ssl_ma_length]
            ssl_ema = self.calculate_ema(ssl_length_close, self.ssl_ma_length)
            self.first_ssl_ma = ssl_ema[-1]
            self.current_ssl = ssl_ema[-1]
            self.prev_ssl_ma = ssl_ema[-1]

        position = await strategy.position(symbol=symbol, exchange=Exchange.BybitLinear)
        # wallet_balance = await strategy.get_current_available_balance(symbol=symbol)
        qty = self.qty

        # vosc > 0 and SSL hybrib = blue
        # close above the top level echo forecast
        # 1.5RR, check the maximum of echo forecast future line is larger than TP and minimum is larger than SL
        # SL recent Swing low
        if (
            self.ssl_color == 1
            and volume_osc[-1] > 0
            and close[-1] >= top_level
            and position.long.quantity <= 0.0
            and position.short.quantity <= 0.0
        ):
            stop_loss = min(low[len(low) - self.swing_period : len(low)])
            take_profit = close[-1] + (self.risk_to_reward * (close[-1] - stop_loss))
            if max_future_line > take_profit and min_future_line > stop_loss:
                await strategy.open(side=OrderSide.Buy, quantity=qty, is_hedge_mode=False, take_profit=take_profit, stop_loss=stop_loss, symbol=symbol, exchange=Exchange.BybitLinear, is_post_only=False)
                logging.info(
                    f"Placed a buy order with qty {qty} when close: {close[-1]}, tp: {take_profit}, sl: {stop_loss}, top_level: {top_level}, btm_level: {btm_level}, max_future_line: {max_future_line}, min_future_line: {min_future_line}, self.ssl_color: {self.ssl_color}, self.current_ssl: {self.current_ssl}, volume_osc: {volume_osc[-1]} at time {start_time[-1]}"
                )

        # vosc > 0 and SSL hybrib = red
        # close below the btm level echo forecast
        # 1.5RR, check the maximum of echo forecast future line is smaller than SL and minimum is smaller than TP
        # SL recent Swing High
        if (
            self.ssl_color == -1
            and volume_osc[-1] > 0
            and close[-1] <= btm_level
            and position.short.quantity <= 0.0
            and position.long.quantity <= 0.0
        ):
            stop_loss = max(high[len(high) - self.swing_period : len(high)])
            take_profit = close[-1] - (self.risk_to_reward * (stop_loss - close[-1]))
            if max_future_line < stop_loss and min_future_line < take_profit:
                await strategy.open(side=OrderSide.Sell, quantity=qty, is_hedge_mode=False, take_profit=take_profit, stop_loss=stop_loss, symbol=symbol, exchange=Exchange.BybitLinear, is_post_only=False)
                logging.info(
                    f"Placed a sell order with qty {qty} when close: {close[-1]}, tp: {take_profit}, sl: {stop_loss}, top_level: {top_level}, btm_level: {btm_level}, max_future_line: {max_future_line}, min_future_line: {min_future_line}, self.ssl_color: {self.ssl_color}, self.current_ssl: {self.current_ssl}, volume_osc: {volume_osc[-1]} at time {start_time[-1]}"
                )

config = RuntimeConfig(
    mode=RuntimeMode.Backtest,
    datasource_topics=[],
    active_order_interval=1,
    initial_capital=10000.0,
    candle_topics=["candles-1m-BTC/USDT-bybit"],
    start_time=datetime(2024, 1, 26, 0, 0, 0, tzinfo=timezone.utc),
    end_time=datetime(2024, 1, 30, 0, 0, 0, tzinfo=timezone.utc),
    api_key="YOUR_CYBOTRADE_API_KEY",
    api_secret="YOUR_CYBOTRADE_API_SECRET",
    data_count=700,
    # exchange_keys="./z_exchange-keys.json",
    )
 
permutation = Permutation(config)
hyper_parameters = {}
hyper_parameters['evalution_window'] = [50]
hyper_parameters['forecast_window'] = [50]
hyper_parameters['ssl_ma_length'] = [200]
hyper_parameters['ssl_base_channel_multiplier'] = [0.2]
hyper_parameters['short_length'] = [5]
hyper_parameters['long_length'] = [10]
 
async def start_backtest():
  await permutation.run(hyper_parameters, Strategy)
 
asyncio.run(start_backtest())