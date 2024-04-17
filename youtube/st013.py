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
    # Smooth HA
    smooth_ha_length1 = 10
    smooth_ha_length2 = 10
    # ATR
    atr_period1 = 1
    atr_period2 = 14
    qty = 0.1
    btcount = 0

    def __init__(self):
        handler = colorlog.StreamHandler()
        handler.setFormatter(
            colorlog.ColoredFormatter(f"%(log_color)s{Strategy.LOG_FORMAT}")
        )
        file_handler = TimedRotatingFileHandler("y_st013-test.log", when="h")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(Strategy.LOG_FORMAT))
        super().__init__(log_level=logging.INFO, handlers=[handler, file_handler])

    async def set_param(self, identifier, value):
        logging.info(f"Setting {identifier} to {value}")
        if identifier == "smooth_ha_length1":
            self.smooth_ha_length1 = int(value)
        elif identifier == "smooth_ha_length2":
            self.smooth_ha_length2 = int(value)
        elif identifier == "atr_period1":
            self.atr_period1 = int(value)
        elif identifier == "atr_period2":
            self.atr_period2 = int(value)
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
            or len(close) < self.atr_period1 * 3
            or len(close) < self.atr_period2 * 3
        ):
            logging.info("Not enough candles to calculate indicators")
            return

        # ATR
        atr1 = talib.ATR(high, low, close, self.atr_period1)
        atr2 = talib.ATR(high, low, close, self.atr_period2)
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

        position = await strategy.position(symbol=symbol, exchange=Exchange.BybitLinear)
        # wallet_balance = await strategy.get_current_available_balance(symbol=symbol)
        qty = self.qty

        # when have long position, if the candle close below smooth_ha candle then close position
        if position.long.quantity != 0.0 and close[-1] < ema_ha_low[-1]:
            await strategy.close(side=OrderSide.Buy,quantity=position.long.quantity,symbol=symbol, exchange=Exchange.BybitLinear, is_hedge_mode=False)
            logging.info(
                f"Placed a close long position with qty {qty} at time {start_time[-1]}"
            )

        # when have short position, if the candle close above smooth_ha candle then close position
        if position.short.quantity != 0.0 and close[-1] > ema_ha_high[-1]:
            await strategy.close(side=OrderSide.Sell,quantity=position.short.quantity,symbol=symbol, exchange=Exchange.BybitLinear, is_hedge_mode=False)
            logging.info(
                f"Placed a close short position with qty {qty} at time {start_time[-1]}"
            )

        # close totally above smooth_ha candle
        # atr1 > atr2
        if (
            smooth_ha_color[-1] == 1
            and atr1[-1] > atr2[-1]
            and low[-1] > ema_ha_high[-1]
            and position.long.quantity <= 0.0
        ):
            stop_loss = low[-1]
            await strategy.open(side=OrderSide.Buy, quantity=qty, is_hedge_mode=False, take_profit=None, stop_loss=stop_loss, symbol=symbol, exchange=Exchange.BybitLinear, is_post_only=False)
            logging.info(
                f"Placed a buy order with qty {qty} when close: {close[-1]}, sl: {stop_loss} at time {start_time[-1]}"
            )

        # supertrend sell signal and qqe_mod red bar and Trend A-V2 is red line
        if (
            smooth_ha_color[-1] == -1
            and atr1[-1] > atr2[-1]
            and high[-1] < ema_ha_low[-1]
            and position.short.quantity <= 0.0
        ):
            stop_loss = high[-1]
            await strategy.open(side=OrderSide.Sell, quantity=qty, is_hedge_mode=False, take_profit=None, stop_loss=stop_loss, symbol=symbol, exchange=Exchange.BybitLinear, is_post_only=False)
            logging.info(
                f"Placed a sell order with qty {qty} when close: {close[-1]}, sl: {stop_loss} at time {start_time[-1]}"
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
hyper_parameters['atr_period1'] = [1]
hyper_parameters['atr_period2'] = [14]
 
async def start_backtest():
  await permutation.run(hyper_parameters, Strategy)
 
asyncio.run(start_backtest())
