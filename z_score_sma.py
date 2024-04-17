from datetime import datetime, timezone
from logging.handlers import TimedRotatingFileHandler
from cybotrade.strategy import Strategy as BaseStrategy
from cybotrade.runtime import Runtime
from cybotrade.models import (
    Exchange,
    OrderSide,
    RuntimeConfig,
    RuntimeMode,
    Symbol,
)
from cybotrade.permutation import Permutation
import talib
import numpy as np
import asyncio
import logging
import colorlog
import math


RUNTIME_MODE = RuntimeMode.Live

class Strategy(BaseStrategy):
    symbol = [Symbol(base="BTC", quote="USDT")]
    quantity = 0.001
    hedge_mode = True
    sma_length = 50
    z_score_threshold = 0.75

    async def set_param(self, identifier, value):
        logging.info(f"Setting {identifier} to {value}")
        if identifier == "sma":
            self.sma_length = float(value)
        elif identifier == "z_score":
            self.z_score_threshold = float(value)
        else:
            logging.error(f"Could not set {identifier}, not found")

    def convert_ms_to_datetime(self, milliseconds):
        seconds = milliseconds / 1000.0
        return datetime.fromtimestamp(seconds)

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

    def __init__(self):
        handler = colorlog.StreamHandler()
        handler.setFormatter(colorlog.ColoredFormatter(f"%(log_color)s{Strategy.LOG_FORMAT}"))
        file_handler = TimedRotatingFileHandler(filename="z_score_sma.log", when="h", backupCount=10)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(Strategy.LOG_FORMAT))
        super().__init__(log_level=logging.INFO, handlers=[handler, file_handler])

    async def on_candle_closed(self, strategy, topic, symbol):
        # Retrieve list of candle data for corresponding symbol that candle closed.
        candles = self.data_map[topic]
        # high = np.array(list(map(lambda c: float(c["high"]), candles)))
        # low = np.array(list(map(lambda c: float(c["low"]), candles)))
        # volume = np.array(list(map(lambda c: float(c["volume"]), candles)))
        # Retrieve close data from list of candle data.
        close = np.array(list(map(lambda c: float(c["close"]), candles)))
        start_time = np.array(list(map(lambda c: float(c["start_time"]), candles)))
        sma_forty = talib.SMA(close, self.sma_length)
        # logging.info(f"close: {close[-1]}, sma_forty: {sma_forty} ")
        # price_changes = (float(close[-1]) / sma_forty[-1]) - 1.0
        std = self.get_stddev(close[-50:])
        z_score = (close[-1] - sma_forty[-1]) / std

        current_pos = await strategy.position(exchange=Exchange.BybitLinear,symbol=symbol)
        logging.info(f"close: {close[-1]}, sma: {sma_forty[-1]}, std: {std}, z_score: {z_score}, current_pos: {current_pos} at {self.convert_ms_to_datetime(start_time[-1])}")

        if z_score > self.z_score_threshold:
            if current_pos.long.quantity == 0.0:
                try:
                    await strategy.open(exchange=Exchange.BybitLinear,side=OrderSide.Buy, quantity=self.quantity, symbol=symbol, limit=None, take_profit=None, stop_loss=None, is_hedge_mode=self.hedge_mode, is_post_only=False)
                except Exception as e:
                    logging.error(f"Failed to open long: {e}")
        else:
            if current_pos.long.quantity != 0.0:
                try:
                    await strategy.close(exchange=Exchange.BybitLinear, side=OrderSide.Buy, quantity=self.quantity, symbol=symbol, is_hedge_mode=self.hedge_mode)
                except Exception as e:
                    logging.error(f"Failed to close entire position: {e}")
        
        new_pos = await strategy.position(exchange= Exchange.BybitLinear, symbol=symbol)
        logging.info(f"new_pos: {new_pos}")


config = RuntimeConfig(
    mode=RUNTIME_MODE,
    datasource_topics=[],
    candle_topics=["candles-1d-BTC/USDT-bybit"],
    active_order_interval=1,
    initial_capital=10000.0,
    exchange_keys="./credentials.json",
    start_time=datetime(2022, 6, 11, 0, 0, 0, tzinfo=timezone.utc),
    end_time=datetime(2024, 1, 5, 0, 0, 0, tzinfo=timezone.utc),
    data_count=150,
    api_key="test",
    api_secret="notest",
)

permutation = Permutation(config)
hyper_parameters = {}
hyper_parameters["sma"] = [50]
hyper_parameters["z_score"] = [0.75]
# hyper_parameters["sma"] = np.arange(10,60,10)
# hyper_parameters["z_score"] = np.arange(0.7,0.8,0.01)


async def start():
  await permutation.run(hyper_parameters, Strategy)
 
asyncio.run(start())
