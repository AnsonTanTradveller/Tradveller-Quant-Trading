from os import environ
from datetime import datetime, timezone, timedelta
from cybotrade.strategy import Strategy as BaseStrategy
from cybotrade.runtime import Runtime
from cybotrade.models import (
    DatahubConfig,
    Direction,
    Exchange,
    Interval,
    OrderParams,
    OrderSide,
    StopParams,
    OrderSize,
    OrderSizeUnit,
    RuntimeConfig,
    RuntimeMode,
    Symbol,
)
import numpy as np
import asyncio
import logging
import colorlog
import random
import requests
from binance import Client
import json
import collections
import time
import pandas as pd
import matplotlib.pyplot as plt

def fetch_kline_price_data():
    API_Library='v1/klines'
    ticker = 'BTCUSDT'
    time_interval= '1m'
    start_time = datetime(2023,12,21) #获取数据开始的时间，YYYY/M/D
    end_time = datetime.now() #数据最新更新的时间

    price_data=[]
    
    
    while start_time < end_time:
        start_time_2 = int(start_time.timestamp() * 1000)
        url = 'https://fapi.binance.com/fapi/'+str(API_Library)+'?symbol='+str(ticker)+'&interval='+str(time_interval)+'&limit=1500&startTime='+str(start_time_2)
        print(start_time)
        resp = requests.get(url)
        resp = json.loads(resp.content.decode())  

        price_data.extend(resp)
        
        start_time = start_time + timedelta(minutes=1500)
    
    price_data = pd.DataFrame(price_data)

    price_data[0] = pd.to_datetime(price_data[0], unit="ms")
    price_data[6] = pd.to_datetime(price_data[6], unit="ms")

    #项目命名Columns Rename
    price_data.columns = ['Time', 'Open','High', 'Low', 'Close', 'Volume','Close Time','Ignore','Ignore','Ignore','Ignore','Ignore']
    price_data = price_data.set_index('Time')

    #price_data = price_data[price_data.index >= datetime(2020, 12, 1)]

    extracted_df = price_data['Close']
   
    
    return extracted_df


price_request = fetch_kline_price_data() 

price_df = pd.DataFrame(price_request)
