from os import environ
from datetime import UTC, datetime, timedelta, timezone
from cybotrade.strategy import Strategy as BaseStrategy
from cybotrade.runtime import Runtime
from cybotrade.models import (
    Exchange,
    Interval,
    OrderSide,
    RuntimeConfig,
    Symbol,
    RuntimeMode,
)
import math
import numpy as np
import asyncio
import logging
import colorlog
from logging.handlers import TimedRotatingFileHandler
from cybotrade.permutation import Permutation
import pandas as pd

class Strategy(BaseStrategy):
    # Indicator params
    holding_time = 432000
    sma_rolling_window = 20
    std_rolling_window = 40
    qty = 0.002
    pair = Symbol(base='BTC', quote='USDT')
    entry_time = datetime.now(tz=timezone.utc).timestamp()

    def __init__(self):
        handler = colorlog.StreamHandler()
        handler.setFormatter(
            colorlog.ColoredFormatter(f"%(log_color)s{Strategy.LOG_FORMAT}")
        )
        file_handler = TimedRotatingFileHandler("btc_netflow_1d.log", when="h", backupCount=50)
        log_level = logging.INFO
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(Strategy.LOG_FORMAT))
        super().__init__(log_level=log_level, handlers=[handler, file_handler])

    async def set_param(self, identifier, value):
        logging.info(f"Setting {identifier} to {value}")
        if identifier == "holding_time":
            self.holding_time = int(value)
        elif identifier == "sma_rolling_window":
            self.sma_rolling_window = int(value)
        elif identifier == "std_rolling_window":
            self.std_rolling_window = int(value)
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

    def get_rolling_mean(self, array, rolling_window):
        arr = [0] * (rolling_window - 1)
        rolling_mean = []
        for i in range(rolling_window, len(array) + 1):
            last_arr = array[i-rolling_window:i]
            rolling_mean.append(self.get_mean(last_arr))
        remove_nan_rolling_mean = np.nan_to_num(rolling_mean)
        return np.concatenate((arr, remove_nan_rolling_mean), axis=0)

    def get_rolling_std(self, array, rolling_window):
        arr = [0] * (rolling_window - 1)
        rolling_std = []
        for i in range(rolling_window, len(array) + 1):
            last_arr = array[i-rolling_window:i]
            rolling_std.append(self.get_stddev(last_arr))
        remove_nan_rolling_std = np.nan_to_num(rolling_std)
        return np.concatenate((arr, remove_nan_rolling_std), axis=0)

    def convert_ms_to_datetime(self, milliseconds):
        seconds = milliseconds / 1000.0
        return datetime.fromtimestamp(seconds, tz=UTC)

    async def on_datasource_interval(self, strategy, topic, data_list):
        # refer to datasource models
        netflow_data = self.data_map[topic]
        netflow_start_time = np.array(list(map(lambda c: float(c["end_time"]), netflow_data)))
        netflow_total = np.array(list(map(lambda c: float(c["netflow_total"]), netflow_data)))
        sma_netflow = self.get_rolling_mean(netflow_total, self.sma_rolling_window)
        std_sma_netflow = self.get_rolling_std(sma_netflow, self.std_rolling_window)
        position = await strategy.position(symbol=self.pair, exchange=Exchange.BybitLinear)
        logging.info(f"current_position: {position}, netflow_total: {netflow_total[-1]}, sma_netflow: {sma_netflow[-1]}, std_sma_netflow: {std_sma_netflow[-1]}, first data : {netflow_data[0]} at {self.convert_ms_to_datetime(netflow_start_time[0])}, last data : {netflow_data[-1]} at {self.convert_ms_to_datetime(netflow_start_time[-1])}, length: {len(netflow_total)}")
        # wallet_balance = await strategy.get_current_available_balance(symbol=symbol)
        qty = self.qty
        if position.long.quantity > 0.0 and netflow_start_time[-1] - self.entry_time >= self.holding_time * 1000:
            try:
                await strategy.close(side=OrderSide.Buy, quantity=qty, symbol=self.pair, exchange=Exchange.BybitLinear, is_hedge_mode=False)
                logging.info(
                    f"Closed long position with qty {qty} when netflow_total: {netflow_total[-1]}, sma_netflow: {sma_netflow[-1]}, std_sma_netflow: {std_sma_netflow[-1]} at time {self.convert_ms_to_datetime(netflow_start_time[-1])}"
                )
            except Exception as e:
                logging.error(f"Failed to close entire position: {e}")
        if (
            sma_netflow[-1] < std_sma_netflow[-1]
            and position.long.quantity <= 0.0
        ):
            self.entry_time = netflow_start_time[-1]
            await strategy.open(side=OrderSide.Buy, quantity=qty, take_profit=None, stop_loss=None, symbol=self.pair, exchange=Exchange.BybitLinear, is_hedge_mode=False, is_post_only=False)
            logging.info(
                f"Placed a buy order with qty {qty} when netflow_total: {netflow_total[-1]}, sma_netflow: {sma_netflow[-1]}, std_sma_netflow: {std_sma_netflow[-1]} at time {self.convert_ms_to_datetime(netflow_start_time[-1])}"
            )

config = RuntimeConfig(
        mode=RuntimeMode.Backtest,
        datasource_topics=[
        "cryptoquant|1d|btc/exchange-flows/netflow?exchange=binance&window=day"
        ],
        active_order_interval=86400,
        initial_capital=1000000.0,
        candle_topics=["candles-1d-BTC/USDT-bybit"],
        start_time=datetime(2020, 5, 10, 0, 0, 0, tzinfo=timezone.utc),
        end_time=datetime(2024, 4, 2, 0, 0, 0, tzinfo=timezone.utc),
        api_key="YOUR_CYBOTRADE_API_KEY",
        api_secret="YOUR_CYBOTRADE_API_SECRET",
        data_count=160,
        exchange_keys="./z_exchange-keys.json",
    )
 
permutation = Permutation(config)
hyper_parameters = {}
hyper_parameters['sma_rolling_window'] = [20] # , 80, 110, 140
hyper_parameters['std_rolling_window'] = [40] # , 80, 110, 140
hyper_parameters['holding_time'] = [432000]
 
async def start_backtest():
  await permutation.run(hyper_parameters, Strategy)
 
asyncio.run(start_backtest())
