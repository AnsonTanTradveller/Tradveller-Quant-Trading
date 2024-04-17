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
    # Rob Hoffman - Overlay Set (RH MA)
    slow_speed_length = 5
    trend_line_1_length = 50
    trend_line_2_length = 89
    fast_primary_trend_line_length = 18
    trend_line_3_length = 144
    mid_line_length = 35
    upper_line_length = 35
    # Rob Hoffman - Inventory Retracement Bar
    inventory_retracement_percent = 0.45
    # RR
    risk_to_reward = 2
    qty = 0.1
    btcount = 0

    def __init__(self):
        handler = colorlog.StreamHandler()
        handler.setFormatter(
            colorlog.ColoredFormatter(f"%(log_color)s{Strategy.LOG_FORMAT}")
        )
        file_handler = TimedRotatingFileHandler("y_st024_rob_hoffman-test.log", when="h")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(Strategy.LOG_FORMAT))
        super().__init__(log_level=logging.INFO, handlers=[handler, file_handler])

    async def set_param(self, identifier, value):
        logging.info(f"Setting {identifier} to {value}")
        if identifier == "slow_speed_length":
            self.slow_speed_length = int(value)
        elif identifier == "trend_line_1_length":
            self.trend_line_1_length = int(value)
        elif identifier == "trend_line_2_length":
            self.trend_line_2_length = int(value)
        elif identifier == "fast_primary_trend_line_length":
            self.fast_primary_trend_line_length = int(value)
        elif identifier == "trend_line_3_length":
            self.trend_line_3_length = int(value)
        elif identifier == "mid_line_length":
            self.mid_line_length = int(value)
        elif identifier == "upper_line_length":
            self.upper_line_length = int(value)
        elif identifier == "inventory_retracement_percent":
            self.inventory_retracement_percent = float(value)
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
            len(close) < self.slow_speed_length * 3
            or len(close) < self.fast_primary_trend_line_length * 3
            or len(close) < self.trend_line_1_length * 3
            or len(close) < self.trend_line_2_length * 3
            or len(close) < self.trend_line_3_length * 3
            or len(close) < self.mid_line_length * 3
            or len(close) < self.upper_line_length * 3
        ):
            logging.info("Not enough candles to calculate indicators")
            return

        # Rob Hoffman - Overlay set
        slow_speed_line = talib.SMA(close, self.slow_speed_length)
        fast_primary_trend_line = talib.EMA(close, self.fast_primary_trend_line_length)
        trend_line_1 = talib.SMA(close, self.trend_line_1_length)
        trend_line_2 = talib.SMA(close, self.trend_line_2_length)
        trend_line_3 = talib.EMA(close, self.trend_line_3_length)
        mid_line = talib.EMA(close, self.mid_line_length)
        # Moving average used in RSI.
        # True range. Same as tr(false). It is max(high - low, abs(high - close[1]), abs(low - close[1]))
        tr = np.array(list([0.0] * len(close)))
        rma = np.array(list([0.0] * len(close)))
        for i in range(1, len(close)):
            tr[i] = max(
                max(high[i] - low[i], abs(high[i] - close[i - 1])),
                abs(low[i] - close[i - 1]),
            )
        # RMA = 1/length * src[i] + (1 - (1/length)) * rma[i-1]
        for i in range(1, len(close)):
            rma[i] = (
                1.0 / self.upper_line_length * tr[i]
                + (1.0 - 1.0 / self.upper_line_length) * rma[i - 1]
            )
        upper_line = mid_line + rma * 0.5

        # Rob Hoffman - Inventory Retracement Bar
        # Candle Range
        candle_range = abs(high - low)
        # Candle Body
        candle_body = abs(close - open)

        # Price Level for Retracement
        x = low + (self.inventory_retracement_percent * candle_range)
        y = high - (self.inventory_retracement_percent * candle_range)
        green_arrow = np.array(list([0.0] * len(close)))
        red_arrow = np.array(list([0.0] * len(close)))
        for i in range(0, len(close)):
            rv = (
                1
                if (
                    candle_body[i]
                    < self.inventory_retracement_percent * candle_range[i]
                )
                else 0
            )
            green_arrow[i] = (
                1
                if (rv == 1 and high[i] > y[i] and close[i] < y[i] and open[i] < y[i])
                else 0
            )
            red_arrow[i] = (
                -1
                if (rv == 1 and low[i] < x[i] and close[i] > x[i] and open[i] > x[i])
                else 0
            )

        position = await strategy.position(symbol=symbol, exchange=Exchange.BybitLinear)
        # wallet_balance = await strategy.get_current_available_balance(symbol=symbol)
        qty = self.qty

        # arror = green
        # close above all the ma, slow_speed_line above fast_primary_trend_line and fast_primary_trend_line above all others ma
        # green candle
        # 1:2 RR / close below fast_primary_trend_line
        if (
            green_arrow[-1] == 1
            and slow_speed_line[-1] > fast_primary_trend_line[-1]
            and close[-1] > slow_speed_line[-1]
            and position.long.quantity <= 0.0
            and fast_primary_trend_line[-1] > trend_line_1[-1]
            and fast_primary_trend_line[-1] > trend_line_2[-1]
            and fast_primary_trend_line[-1] > trend_line_3[-1]
            and fast_primary_trend_line[-1] > mid_line[-1]
            and fast_primary_trend_line[-1] > upper_line[-1]
            and close[-1] > open[-1]
        ):
            stop_loss = min(
                [
                    trend_line_1[-1],
                    trend_line_2[-1],
                    trend_line_3[-1],
                    mid_line[-1],
                    upper_line[-1],
                ]
            )
            take_profit = close[-1] + (self.risk_to_reward * (close[-1] - stop_loss))
            await strategy.open(side=OrderSide.Buy, quantity=qty, is_hedge_mode=False, take_profit=take_profit, stop_loss=stop_loss, symbol=symbol, exchange=Exchange.BybitLinear, is_post_only=False)
            logging.info(
                f"Placed a buy order with qty {qty} when close: {close[-1]}, tp: {take_profit}, sl: {stop_loss} at time {start_time[-1]}"
            )

        # arror = red
        # close below all the ma, slow_speed_line below fast_primary_trend_line and fast_primary_trend_line below all others ma
        # red candle
        # 1:2 RR / close above fast_primary_trend_line
        if (
            red_arrow[-1] == -1
            and slow_speed_line[-1] < fast_primary_trend_line[-1]
            and close[-1] < slow_speed_line[-1]
            and position.short.quantity <= 0.0
            and fast_primary_trend_line[-1] < trend_line_1[-1]
            and fast_primary_trend_line[-1] < trend_line_2[-1]
            and fast_primary_trend_line[-1] < trend_line_3[-1]
            and fast_primary_trend_line[-1] < mid_line[-1]
            and fast_primary_trend_line[-1] < upper_line[-1]
            and close[-1] < open[-1]
        ):
            stop_loss = max(
                [
                    trend_line_1[-1],
                    trend_line_2[-1],
                    trend_line_3[-1],
                    mid_line[-1],
                    upper_line[-1],
                ]
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
    data_count=600,
    # exchange_keys="./z_exchange-keys.json",
    )

permutation = Permutation(config)
hyper_parameters = {}
hyper_parameters['slow_speed_length'] = [5]
hyper_parameters['trend_line_1_length'] = [50]
hyper_parameters['trend_line_2_length'] = [89]
hyper_parameters['fast_primary_trend_line_length'] = [18]
hyper_parameters['trend_line_3_length'] = [144]
hyper_parameters['mid_line_length'] = [35]
hyper_parameters['upper_line_length'] = [35]
hyper_parameters['inventory_retracement_percent'] = [0.45]
 
async def start_backtest():
  await permutation.run(hyper_parameters, Strategy)
 
asyncio.run(start_backtest())