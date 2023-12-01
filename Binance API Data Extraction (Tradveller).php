import requests
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np


df = pd.DataFrame()
data =[]

#用户设置参数区 User Settings Panel ##

API_Library='v1/klines' # https://binance-docs.github.io/apidocs/futures/en/#market-data-endpoints, Binance API 链接文档
ticker = 'BTCUSDT' #交易对code
time_interval='1d' #1m,3m,5m,15m,30m，1h,2h,4h,6h,8h,12h,1d,3d,1w,1M

start_time = datetime(2020,1,1) #获取数据开始的时间，YYYY/M/D
end_time = datetime.now() #数据最新更新的时间

###### User Panel End ######


while start_time < end_time:
    print(start_time)
    start_time_2 = int(start_time.timestamp() * 1000)
    url = 'https://fapi.binance.com/fapi/'+str(API_Library)+'?symbol='+str(ticker)+'&interval='+str(time_interval)+'&limit=1500&startTime='+str(start_time_2)
    resp = requests.get(url)
    resp = json.loads(resp.content.decode())
    data.append(resp)
    start_time = start_time + timedelta(minutes=1500)



df = pd.DataFrame(data)
combined_rows = []
for _, row in df.iterrows():
    combined_row = []
    for cell in row:
       
        combined_row.extend(cell if cell is not None else [np.nan, np.nan, np.nan])
    combined_rows.append(combined_row)


split_rows = [row[i:i + 12] for row in combined_rows for i in range(0, len(row), 12)]


new_df = pd.DataFrame(split_rows)

new_df[0] = pd.to_datetime(new_df[0], unit="ms")
new_df[6] = pd.to_datetime(new_df[6], unit="ms")

#项目命名Columns Rename
new_df.columns = ['Time', 'Open','High', 'Low', 'Close', 'Volume','Close Time','Ignore','Ignore','Ignore','Ignore','Ignore']
new_df = new_df.set_index('Time')


print("DataFrame with concatenated arrays:")
print(new_df)


#若要将数据导成 .csv档，需把下一行的 ‘#’ 删除，并在'()'内重新命名档案。
#new_df = new_df.to_csv('Sure Win Easy.csv')
