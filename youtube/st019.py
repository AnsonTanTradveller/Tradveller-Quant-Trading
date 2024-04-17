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
    # ssl channel length
    ssl_length = 200
    # NSDT HAMA Candles
    high_length = 20
    low_length = 20
    ema_length = 100
    wma_length = 55
    wma_ema_length = 3
    steps = 5
    bull_bear_weak = 0
    bull_strong = 1
    bear_strong = -1
    # To keep track of the trend in SSL
    ssl_up_trend = False
    ssl_down_trend = False
    qty = 0.1
    btcount = 0

    def __init__(self):
        handler = colorlog.StreamHandler()
        handler.setFormatter(
            colorlog.ColoredFormatter(f"%(log_color)s{Strategy.LOG_FORMAT}")
        )
        file_handler = TimedRotatingFileHandler("y_st019_ssl_hama_strategy-test.log", when="h")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(Strategy.LOG_FORMAT))
        super().__init__(log_level=logging.INFO, handlers=[handler, file_handler])

    async def set_param(self, identifier, value):
        logging.info(f"Setting {identifier} to {value}")
        if identifier == "ssl_length":
            self.ssl_length = int(value)
        elif identifier == "high_length":
            self.high_length = int(value)
        elif identifier == "low_length":
            self.low_length = int(value)
        elif identifier == "ema_length":
            self.ema_length = int(value)
        elif identifier == "wma_length":
            self.wma_length = int(value)
        elif identifier == "wma_ema_length":
            self.wma_ema_length = int(value)
        elif identifier == "steps":
            self.steps = int(value)
        else:
            logging.error(f"Could not set {identifier}, not found")

    # function to get the color of NSDT HAMA ma line and candles colors
    def get_color(
        self, ma_arr, center_arr, step, bear_weak, bear_strong, bull_weak, bull_strong
    ):
        qty_adv_dec = [0] * len(ma_arr)
        final_color = [0] * len(ma_arr)
        max_steps = max(1, step)
        # crossover, x > y and prev_y < prev_x => true else false
        x_up = [0] * len(ma_arr)
        # crossunder, x < y and prev_y > prev_x => true else false
        x_down = [0] * len(ma_arr)
        # Difference between current value and previous
        chg = [0] * len(ma_arr)
        for i in range(1, len(ma_arr)):
            if ma_arr[i] > center_arr[i] and ma_arr[i - 1] < center_arr[i - 1]:
                x_up[i] = 1
            else:
                x_up[i] = 0
            if ma_arr[i] < center_arr[i] and ma_arr[i - 1] > center_arr[i - 1]:
                x_down[i] = 1
            else:
                x_down[i] = 0
            chg[i] = ma_arr[i] - ma_arr[i - 1]

        up = [0] * len(ma_arr)
        dn = [0] * len(ma_arr)
        src_bull = [0] * len(ma_arr)
        src_bear = [0] * len(ma_arr)
        for i in range(0, len(ma_arr)):
            if chg[i] > 0:
                up[i] = 1
                dn[i] = 0
            else:
                up[i] = 0
                dn[i] = 1
            if ma_arr[i] > center_arr[i]:
                src_bull[i] = 1
                src_bear[i] = 0
            else:
                src_bull[i] = 0
                src_bear[i] = 1

        for i in range(1, len(ma_arr)):
            if src_bull[i] == 1:
                if x_up[i] == 1:
                    qty_adv_dec[i] = 1
                else:
                    if up[i] == 1:
                        qty_adv_dec[i] = min(max_steps, qty_adv_dec[i - 1] + 1)
                    else:
                        if dn[i] == 1:
                            qty_adv_dec[i] = max(1, qty_adv_dec[i - 1] - 1)
                        else:
                            qty_adv_dec[i] = qty_adv_dec[i - 1]
            elif src_bear[i] == 1:
                if x_down[i] == 1:
                    qty_adv_dec[i] = 1
                else:
                    if dn[i] == 1:
                        qty_adv_dec[i] = min(max_steps, qty_adv_dec[i - 1] + 1)
                    else:
                        if up[i] == 1:
                            qty_adv_dec[i] = max(1, qty_adv_dec[i - 1] - 1)
                        else:
                            qty_adv_dec[i] = qty_adv_dec[i - 1]
            else:
                qty_adv_dec[i] = qty_adv_dec[i - 1]
        # the color will change in few colors in trading view, but the main colors to identify signal are only green or red
        # the color steps are 5, 1 is yellow(based on what color u choose), 2 ,3 ,4 will turn abit into green/red , 5 is equal to green/red
        # Based on my understanding, only 5 which is the max steps only consider as Green or Red, even 4 in trading view seem like so near to green or red
        # green = 1, red = -1 , others color = 0
        for i in range(1, len(ma_arr)):
            if src_bull[i] == 1:
                if qty_adv_dec[i] == max_steps:
                    final_color[i] = bull_strong
                else:
                    final_color[i] = bull_weak
            elif src_bear[i] == 1:
                if qty_adv_dec[i] == max_steps:
                    final_color[i] = bear_strong
                else:
                    final_color[i] = bear_weak
            else:
                final_color[i] = final_color[i - 1]

        return final_color

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
            len(close) < self.ssl_length * 3
            or len(close) < self.ema_length * 3
            or len(close) < self.high_length * 3
            or len(close) < self.low_length * 3
            or len(close) < self.wma_ema_length * 3
            or len(close) < self.wma_length * 3
        ):
            logging.info("Not enough candles to calculate indicators")
            return

        # SSL channel
        sma_high = cybotrade_indicators.sma(real=high, period=self.ssl_length)
        sma_low = cybotrade_indicators.sma(real=low, period=self.ssl_length)
        # using cybotrade_indicators original indicator from tulinb will need to unshift to get back the ori length of array
        for i in range(0, self.ssl_length - 1):
            sma_high = np.insert(sma_high, 0, 0)
            sma_low = np.insert(sma_low, 0, 0)
        hlv = [0]
        ssl_down = []
        ssl_up = []
        sma_high = sma_high.tolist()
        sma_low = sma_low.tolist()
        for i in range(1, len(close)):
            if close[i] > sma_high[i]:
                hlv.append(1)
            elif close[i] < sma_low[i]:
                hlv.append(-1)
            else:
                hlv.append(hlv[-1])

        for i in range(0, len(close)):
            if hlv[i] < 0:
                ssl_down.append(sma_high[i])
                ssl_up.append(sma_low[i])
            else:
                ssl_down.append(sma_low[i])
                ssl_up.append(sma_high[i])

        # NSDT HAMA candles
        # get ema of high and low
        # Using HA candles formula but using normal candles to calculate
        new_high = np.array(list([0.0] * len(close)))
        new_low = np.array(list([0.0] * len(close)))
        for i in range(0, len(close)):
            new_high[i] = max(high[i], close[i])
            new_low[i] = min(low[i], close[i])
        ema_high = talib.EMA(new_high, self.high_length)
        ema_low = talib.EMA(new_low, self.low_length)
        # MA line = normal close EMA
        ema_line = cybotrade_indicators.ema(real=close, period=self.ema_length)

        # WMA and EMA to get the colors for MA line and HAMA candles
        # the color will change in few colors in trading view, but the main colors to identify signal are only green and red, yellow mean no signal
        # Based on my understanding, only 5 which is the max steps only consider as Green or Red, even 4 in trading view seem like so near to green or red
        # the color steps are 5, 1 is yellow(based on what color u choose), 2 ,3 ,4 will turn abit into green/red , 5 is equal to green/red
        # green = 1, red = -1 , others color = 0
        color_wma = talib.WMA(close, self.wma_length)
        color_ema = talib.EMA(color_wma, self.wma_ema_length)
        final_color = self.get_color(
            color_wma,
            color_ema,
            self.steps,
            self.bull_bear_weak,
            self.bear_strong,
            self.bull_bear_weak,
            self.bull_strong,
        )

        position = await strategy.position(symbol=symbol, exchange=Exchange.BybitLinear)
        # wallet_balance = await strategy.get_current_available_balance(symbol=symbol)
        qty = self.qty

        if ssl_up[-1] > ssl_down[-1] and ssl_up[-2] < ssl_down[-2]:
            self.ssl_up_trend = True
            self.ssl_down_trend = False
            if position.short.quantity != 0.0:
                await strategy.close(side=OrderSide.Sell,quantity=position.short.quantity,symbol=symbol, exchange=Exchange.BybitLinear, is_hedge_mode=False)
                logging.info(
                    f"Placed a close short position with qty {qty} at time {start_time[-1]}"
                )

        elif ssl_up[-1] < ssl_down[-1] and ssl_up[-2] > ssl_down[-2]:
            self.ssl_up_trend = False
            self.ssl_down_trend = True
            if position.long.quantity != 0.0:
                await strategy.close(side=OrderSide.Buy,quantity=position.long.quantity,symbol=symbol, exchange=Exchange.BybitLinear, is_hedge_mode=False)
                logging.info(
                    f"Placed a close long position with qty {qty} at time {start_time[-1]}"
                )
                
        if (
            final_color[-1] == 1  # line color = Green
            and close[-1] > ema_line[-1]  # close above ema line
            and self.ssl_up_trend  # SSL up trend
            and close[-1] > ema_high[-1]  # close above the HAMA candle high
            and close[-1] > open[-1]  # Green candle
            and position.long.quantity <= 0.0
        ):
            stop_loss = ssl_down[-1]
            take_profit = None
            await strategy.open(side=OrderSide.Buy, quantity=qty, is_hedge_mode=False, take_profit=take_profit, stop_loss=stop_loss, symbol=symbol, exchange=Exchange.BybitLinear, is_post_only=False)
            logging.info(
                f"Placed a buy order with qty {qty} when close: {close[-1]}, tp: {take_profit}, sl: {stop_loss} at time {start_time[-1]}"
            )
            self.ssl_up_trend = False

        # supertrend sell signal and qqe_mod red bar and Trend A-V2 is red line
        if (
            final_color[-1] == -1  # Red color line
            and close[-1] < ema_line[-1]  # close below ema line
            and self.ssl_down_trend  # SSL down trend
            and close[-1] < ema_low[-1]  # close below HAMA candle low
            and close[-1] < open[-1]  # red candle
            and position.short.quantity <= 0.0
        ):
            stop_loss = ssl_down[-1]
            take_profit = None
            await strategy.open(side=OrderSide.Sell, quantity=qty, is_hedge_mode=False, take_profit=take_profit, stop_loss=stop_loss, symbol=symbol, exchange=Exchange.BybitLinear, is_post_only=False)
            logging.info(
                f"Placed a sell order with qty {qty} when close: {close[-1]}, tp: {take_profit}, sl: {stop_loss} at time {start_time[-1]}"
            )
            self.ssl_down_trend = False

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
    data_count=800,
    # exchange_keys="./z_exchange-keys.json",
    )

permutation = Permutation(config)
hyper_parameters = {}
hyper_parameters['ssl_length'] = [200]
hyper_parameters['high_length'] = [20]
hyper_parameters['low_length'] = [20]
hyper_parameters['ema_length'] = [100]
hyper_parameters['wma_length'] = [55]
hyper_parameters['wma_ema_length'] = [3]
hyper_parameters['steps'] = [5]
 
async def start_backtest():
  await permutation.run(hyper_parameters, Strategy)
 
asyncio.run(start_backtest())