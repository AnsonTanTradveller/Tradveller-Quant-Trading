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
    # Volatility oscillator
    vo_length = 100
    # Normalized MACD
    fast_ma_length = 13
    slow_ma_length = 21
    trigger_length = 9
    normalize_length = 50
    # BB band
    bb_length = 34
    bb_std_dev = 2
    # To keep track of the trend of Normalized MACD and RSI
    nmacd_up_trend = False
    nmacd_down_trend = False
    rsi_up_trend = False
    rsi_down_trend = False
    # risk to reward
    risk_to_reward = 1.5
    # previous candle bar to get stoploss(lowest low / highest high)
    prev_bar = 15
    qty = 0.1
    btcount = 0

    def __init__(self):
        handler = colorlog.StreamHandler()
        handler.setFormatter(
            colorlog.ColoredFormatter(f"%(log_color)s{Strategy.LOG_FORMAT}")
        )
        file_handler = TimedRotatingFileHandler("y_st010_nmacd_bb_vo-test.log", when="h")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(Strategy.LOG_FORMAT))
        super().__init__(log_level=logging.INFO, handlers=[handler, file_handler])

    def weightedmovingaverage(self, Data, period):
        weighted = []
        for i in range(len(Data)):
            try:
                total = np.arange(1, period + 1, 1)  # weight matrix
                matrix = Data[i - period + 1 : i + 1]
                matrix = total * matrix  # multiplication
                wma = (matrix.sum()) / (total.sum())  # WMA
                weighted = np.append(weighted, wma)  # add to array
            except ValueError:
                weighted = np.append(weighted, 0.0)
        return weighted

    async def set_param(self, identifier, value):
        logging.info(f"Setting {identifier} to {value}")
        if identifier == "vo_length":
            self.vo_length = int(value)
        elif identifier == "fast_ma_length":
            self.fast_ma_length = int(value)
        elif identifier == "slow_ma_length":
            self.slow_ma_length = int(value)
        elif identifier == "trigger_length":
            self.trigger_length = int(value)
        elif identifier == "normalize_length":
            self.normalize_length = int(value)
        elif identifier == "bb_length":
            self.bb_length = int(value)
        elif identifier == "bb_std_dev":
            self.bb_std_dev = float(value)
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
            len(close) < self.slow_ma_length * 3
            or len(close) < self.fast_ma_length * 3
            or len(close) < self.normalize_length * 3
            or len(close) < self.bb_length * 3
            or len(close) < self.vo_length * 3
            or len(close) < self.trigger_length * 3
        ):
            logging.info("Not enough candles to calculate indicators")
            return

        # bb band
        bb_band_data = cybotrade_indicators.bbands(
            real=close, stddev=self.bb_std_dev, period=self.bb_length
        )
        bb_middle_line = bb_band_data[1]
        # Volatility Oscillator
        vo_spike = close - open
        vo_upper = cybotrade_indicators.stddev(real=vo_spike, period=self.vo_length)
        vo_lower = (
            cybotrade_indicators.stddev(real=vo_spike, period=self.vo_length) * -1
        )

        # Normalized MACD
        sh = talib.EMA(close, self.fast_ma_length)
        lon = talib.EMA(close, self.slow_ma_length)
        ratio = np.array(list([0.0] * len(close)))
        mac = np.array(list([0.0] * len(close)))
        mac_norm = np.array(list([0.0] * len(close)))
        mac_norm_2 = np.array(list([0.0] * len(close)))

        for i in range(0, len(close)):
            ratio[i] = min(sh[i], lon[i]) / max(sh[i], lon[i])
            if sh[i] > lon[i]:
                mac[i] = 2.0 - ratio[i] - 1.0
            else:
                mac[i] = ratio[i] - 1.0

        for i in range(self.normalize_length - 1, len(close)):
            sliced_mac = mac[i - self.normalize_length + 1 : i + 1]
            max_mac = sliced_mac.max()
            min_mac = sliced_mac.min()
            mac_norm[i] = float(
                ((mac[i] - min_mac) / (max_mac - min_mac + 0.000001) * 2.0) - 1.0
            )

        if self.normalize_length < 2:
            mac_norm_2 = mac
        else:
            mac_norm_2 = mac_norm

        wma_mac = self.weightedmovingaverage(mac_norm_2, self.trigger_length)

        position = await strategy.position(symbol=symbol, exchange=Exchange.BybitLinear)
        # wallet_balance = await strategy.get_current_available_balance(symbol=symbol)
        qty = self.qty

        # nmacd fast line cross up slow line and both line below 0 => nmacd up trend
        # nmacd slow line cross down fast line and both line above 0 => nmacd down trend
        if (
            wma_mac[-1] < 0.0
            and mac_norm_2[-1] < 0.0
            and wma_mac[-1] > mac_norm_2[-1]
            and wma_mac[-2] < mac_norm_2[-2]
        ):
            self.nmacd_up_trend = True
            self.nmacd_down_trend = False
        elif (
            wma_mac[-1] > 0.0
            and mac_norm_2[-1] > 0.0
            and wma_mac[-1] < mac_norm_2[-1]
            and wma_mac[-2] > mac_norm_2[-2]
        ):
            self.nmacd_up_trend = False
            self.nmacd_down_trend = True

        # nmacd up trend
        # vo_spike above upper_vo
        # candle close above bb_middle_line
        # stoploss use previous 15 bars lowest low  
        if (
            self.nmacd_up_trend
            and close[-1] > bb_middle_line[-1]
            and vo_spike[-1] > vo_upper[-1]
            and position.long.quantity <= 0.0
            and position.short.quantity <= 0.0
        ):
            stop_loss = low[len(low) - self.prev_bar : len(low) - 1].min()
            take_profit = close[-1] + (self.risk_to_reward * (close[-1] - stop_loss))
            await strategy.open(side=OrderSide.Buy, quantity=qty, is_hedge_mode=False, take_profit=take_profit, stop_loss=stop_loss, symbol=symbol, exchange=Exchange.BybitLinear, is_post_only=False)
            logging.info(
                f"Placed a buy order with qty {qty} when close: {close[-1]}, tp: {take_profit}, sl: {stop_loss} at time {start_time[-1]}"
            )

        # nmacd down trend
        # vo_spike below lower_vo
        # candle close below bb_middle_line
        # stoploss use previous 15 bars highest high
        if (
            self.nmacd_down_trend
            and close[-1] < bb_middle_line[-1]
            and vo_spike[-1] < vo_lower[-1]
            and position.short.quantity <= 0.0
            and position.long.quantity <= 0.0
        ):
            stop_loss = high[len(high) - self.prev_bar : len(high) - 1].max()
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
hyper_parameters['vo_length'] = [100]
hyper_parameters['fast_ma_length'] = [13]
hyper_parameters['slow_ma_length'] = [21]
hyper_parameters['trigger_length'] = [9]
hyper_parameters['normalize_length'] = [50]
hyper_parameters['bb_length'] = [34]
hyper_parameters['bb_std_dev'] = [2]
 
async def start_backtest():
  await permutation.run(hyper_parameters, Strategy)
 
asyncio.run(start_backtest())