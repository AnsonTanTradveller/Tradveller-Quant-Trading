from os import environ
from datetime import datetime, timezone
from cybotrade.strategy import Strategy as BaseStrategy
from cybotrade.runtime import Runtime
from cybotrade.models import (
    Exchange,
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
    # Smooth HA
    smooth_ha_length1 = 10
    smooth_ha_length2 = 10
    # HA rsi
    rsi_length = 14
    smoothing_length = 7
    over_buy_zone = 20
    over_sell_zone = -20
    # Swing high/low period
    swing_period = 10
    # maximum stoploss percent
    max_stoploss_percentage = 2
    # risk_to_reward
    risk_to_reward = 2.2
    qty = 0.1
    btcount = 0

    def __init__(self):
        handler = colorlog.StreamHandler()
        handler.setFormatter(
            colorlog.ColoredFormatter(f"%(log_color)s{Strategy.LOG_FORMAT}")
        )
        file_handler = TimedRotatingFileHandler("y_ste018_smooth_ha_harsi-test.log", when="h")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(Strategy.LOG_FORMAT))
        super().__init__(log_level=logging.INFO, handlers=[handler, file_handler])

    async def set_param(self, identifier, value):
        logging.info(f"Setting {identifier} to {value}")
        if identifier == "smooth_ha_length1":
            self.smooth_ha_length1 = int(value)
        elif identifier == "smooth_ha_length2":
            self.smooth_ha_length2 = int(value)
        elif identifier == "rsi_length":
            self.rsi_length = int(value)
        elif identifier == "smoothing_length":
            self.smoothing_length = int(value)
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
            len(close) < self.smooth_ha_length1 * 3
            or len(close) < self.smooth_ha_length2 * 3
            or len(close) < self.rsi_length * 3
            or len(close) < self.smoothing_length * 3
        ):
            logging.info("Not enough candles to calculate indicators")
            return

        # Smooth HA
        ema_open = cybotrade_indicators.ema(open, self.smooth_ha_length1)
        ema_high = cybotrade_indicators.ema(high, self.smooth_ha_length1)
        ema_low = cybotrade_indicators.ema(low, self.smooth_ha_length1)
        ema_close = cybotrade_indicators.ema(close, self.smooth_ha_length1)
        ha_ohlc = cybotrade_indicators.ha(
            open=ema_open, high=ema_high, low=ema_low, close=ema_close
        )
        ha_open = ha_ohlc[0]
        ha_high = ha_ohlc[1]
        ha_low = ha_ohlc[2]
        ha_close = ha_ohlc[3]
        ema_ha_open = cybotrade_indicators.ema(ha_open, self.smooth_ha_length2)
        ema_ha_high = cybotrade_indicators.ema(ha_high, self.smooth_ha_length2)
        ema_ha_low = cybotrade_indicators.ema(ha_low, self.smooth_ha_length2)
        ema_ha_close = cybotrade_indicators.ema(ha_close, self.smooth_ha_length2)
        smooth_ha_color = np.array(list([0.0] * len(close)))
        for i in range(0, len(close)):
            smooth_ha_color[i] = 1 if ema_ha_close[i] >= ema_ha_open[i] else -1

        # Smoothed HA RSI
        close_rsi = talib.RSI(close, self.rsi_length) - 50
        y = np.delete(close_rsi, len(close_rsi) - 1)
        open_rsi = np.insert(y, 0, close_rsi[0])
        high_rsi_raw = talib.RSI(high, self.rsi_length) - 50
        low_rsi_raw = talib.RSI(low, self.rsi_length) - 50
        high_rsi = np.array(list([0.0] * len(close)))
        low_rsi = np.array(list([0.0] * len(close)))
        for i in range(0, len(high_rsi_raw)):
            high_rsi[i] = max(high_rsi_raw[i], low_rsi_raw[i])
            low_rsi[i] = min(high_rsi_raw[i], low_rsi_raw[i])

        ha_rsi_close = (open_rsi + high_rsi + low_rsi + close_rsi) / 4
        ha_rsi_open = (open_rsi + close_rsi) / 2
        ha_rsi_high = np.array(list([0.0] * len(close)))
        ha_rsi_low = np.array(list([0.0] * len(close)))
        for i in range(1, len(close)):
            if not np.isnan(ha_rsi_open[i - 1]):
                ha_rsi_open[i] = (
                    ha_rsi_open[i - 1] * self.smoothing_length + ha_rsi_close[i - 1]
                ) / (self.smoothing_length + 1)

        for i in range(0, len(close)):
            ha_rsi_high[i] = max(high_rsi[i], max(ha_rsi_open[i], ha_rsi_close[i]))
            ha_rsi_low[i] = min(low_rsi[i], min(ha_rsi_open[i], ha_rsi_close[i]))

        position = await strategy.position(symbol=symbol, exchange=Exchange.BybitLinear)
        # wallet_balance = await strategy.get_current_available_balance(symbol=symbol)
        qty = self.qty

        # smooth_ha candle is green
        # HA rsi is printed in zero line(which mean close above zero and open below zero) and is green candle and high/low didnt hit oversold/buy
        # stop loss = swing low / smooth_ha candle low
        if (
            smooth_ha_color[-1] == 1
            and ha_rsi_close[-1] > ha_rsi_open[-1]
            and ha_rsi_close[-1] > 0
            and ha_rsi_open[-1] < 0
            and ha_rsi_high[-1] < self.over_buy_zone
            and ha_rsi_low[-1] > self.over_sell_zone
            and position.long.quantity <= 0.0
            and position.short.quantity <= 0.0
        ):
            stop_loss1 = ema_ha_low[-1]
            stop_loss2 = min(low[len(low) - self.swing_period : len(low)])
            stop_loss = 0.0
            if (close[-1] - stop_loss1) / close[
                -1
            ] * 100 < self.max_stoploss_percentage and (close[-1] - stop_loss2) / close[
                -1
            ] * 100 < self.max_stoploss_percentage:
                stop_loss = min(stop_loss1, stop_loss2)
            elif (close[-1] - stop_loss1) / close[
                -1
            ] * 100 < self.max_stoploss_percentage and (close[-1] - stop_loss2) / close[
                -1
            ] * 100 > self.max_stoploss_percentage:
                stop_loss = stop_loss1
            elif (close[-1] - stop_loss1) / close[
                -1
            ] * 100 > self.max_stoploss_percentage and (close[-1] - stop_loss2) / close[
                -1
            ] * 100 < self.max_stoploss_percentage:
                stop_loss = stop_loss2
            else:
                stop_loss = max(stop_loss1, stop_loss2)
            take_profit = close[-1] + (self.risk_to_reward * (close[-1] - stop_loss))
            await strategy.open(side=OrderSide.Buy, quantity=qty, is_hedge_mode=False, take_profit=take_profit, stop_loss=stop_loss, symbol=symbol, exchange=Exchange.BybitLinear, is_post_only=False)
            logging.info(
                f"Placed a buy order with qty {qty} when close: {close[-1]}, tp: {take_profit}, sl: {stop_loss} at time {start_time[-1]}"
            )

        # smooth_ha candle is red
        # HA rsi is printed in zero line(which mean open above zero and close below zero) and is red candle and high/low didnt hit oversold/buy
        # stop loss = swing high / smooth_ha candle high
        if (
            smooth_ha_color[-1] == -1
            and ha_rsi_close[-1] < ha_rsi_open[-1]
            and ha_rsi_close[-1] < 0
            and ha_rsi_open[-1] > 0
            and ha_rsi_high[-1] < self.over_buy_zone
            and ha_rsi_low[-1] > self.over_sell_zone
            and position.long.quantity <= 0.0
            and position.short.quantity <= 0.0
        ):
            stop_loss1 = ema_ha_high[-1]
            stop_loss2 = max(high[len(high) - self.swing_period : len(high)])
            stop_loss = 0.0
            if (stop_loss1 - close[-1]) / close[
                -1
            ] * 100 < self.max_stoploss_percentage and (stop_loss2 - close[-1]) / close[
                -1
            ] * 100 < self.max_stoploss_percentage:
                stop_loss = max(stop_loss1, stop_loss2)
            elif (stop_loss1 - close[-1]) / close[
                -1
            ] * 100 < self.max_stoploss_percentage and (stop_loss2 - close[-1]) / close[
                -1
            ] * 100 > self.max_stoploss_percentage:
                stop_loss = stop_loss1
            elif (stop_loss1 - close[-1]) / close[
                -1
            ] * 100 > self.max_stoploss_percentage and (stop_loss2 - close[-1]) / close[
                -1
            ] * 100 < self.max_stoploss_percentage:
                stop_loss = stop_loss2
            else:
                stop_loss = min(stop_loss1, stop_loss2)
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
    data_count=200,
    # exchange_keys="./z_exchange-keys.json",
    )

permutation = Permutation(config)
hyper_parameters = {}
hyper_parameters['smooth_ha_length1'] = [10]
hyper_parameters['smooth_ha_length2'] = [10]
hyper_parameters['rsi_length'] = [14]
hyper_parameters['smoothing_length'] = [7]
 
async def start_backtest():
  await permutation.run(hyper_parameters, Strategy)
 
asyncio.run(start_backtest())