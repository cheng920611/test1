# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 15:30:38 2024

@author: user
"""

import numpy as np
import datetime
import pandas as pd
import streamlit as st 
import streamlit.components.v1 as stc 
import plotly.graph_objects as go
from plotly.subplots import make_subplots

html_temp = """
<div style="background-color:#FFA500;padding:10px;border-radius:10px">
<h1 style="color:white;text-align:center;">金融期末報告(聯電) </h1>
<h2 style="color:white;text-align:center;">資科三B 411001241 周威丞 </h2>
</div>
"""
stc.html(html_temp)

# Load and preprocess the data
df_original = pd.read_excel("2303.xlsx")
df_original.to_pickle('2303.pkl')
df_original = pd.read_pickle('2303.pkl')
df_original = df_original.drop('Unnamed: 0', axis=1)

st.subheader("選擇開始與結束的日期, 區間:2020-01-01 至 2024-06-20")
start_date = st.date_input('選擇開始日期', value=datetime.date(2020, 1, 1), min_value=datetime.date(2020, 1, 1), max_value=datetime.date(2024, 6, 20))
end_date = st.date_input('選擇結束日期', value=datetime.date(2024, 6, 20), min_value=datetime.date(2020, 1, 1), max_value=datetime.date(2024, 6, 20))
start_date = datetime.datetime.combine(start_date, datetime.datetime.min.time())
end_date = datetime.datetime.combine(end_date, datetime.datetime.min.time())

df = df_original[(df_original['time'] >= start_date) & (df_original['time'] <= end_date)]
st.dataframe(df)

# Convert data to dictionary and numpy arrays
KBar_dic = df.to_dict()
KBar_dic['open'] = np.array(list(KBar_dic['open'].values()))
KBar_dic['time'] = np.array([i.to_pydatetime() for i in list(KBar_dic['time'].values())])
KBar_dic['low'] = np.array(list(KBar_dic['low'].values()))
KBar_dic['high'] = np.array(list(KBar_dic['high'].values()))
KBar_dic['close'] = np.array(list(KBar_dic['close'].values()))
KBar_dic['volume'] = np.array(list(KBar_dic['volume'].values()))
KBar_dic['amount'] = np.array(list(KBar_dic['amount'].values()))
KBar_dic['product'] = np.repeat('tsmc', KBar_dic['open'].size)

# Class to manage K-Bar data
class KBar:
    def __init__(self, start_date, duration):
        self.start_date = start_date
        self.duration = duration
        self.TAKBar = {'time': [], 'open': [], 'high': [], 'low': [], 'close': [], 'volume': []}
        
    def AddPrice(self, time, open_price, close_price, low_price, high_price, volume):
        if not self.TAKBar['time'] or time >= self.TAKBar['time'][-1] + datetime.timedelta(minutes=self.duration):
            self.TAKBar['time'].append(time)
            self.TAKBar['open'].append(open_price)
            self.TAKBar['high'].append(high_price)
            self.TAKBar['low'].append(low_price)
            self.TAKBar['close'].append(close_price)
            self.TAKBar['volume'].append(volume)
        else:
            self.TAKBar['high'][-1] = max(self.TAKBar['high'][-1], high_price)
            self.TAKBar['low'][-1] = min(self.TAKBar['low'][-1], low_price)
            self.TAKBar['close'][-1] = close_price
            self.TAKBar['volume'][-1] += volume

st.subheader("設定一根 K 棒的時間長度 (分鐘)")
cycle_duration = st.number_input(
    '輸入一根 K 棒的時間長度 (單位:分鐘, 一日=1440分鐘)', 
    min_value=1, 
    max_value=1440, 
    value=1440
)
st.write(f"你設定的一根 K 棒的時間長度為: {cycle_duration} 分鐘")

# Create KBar object and add prices
kbar = KBar(start_date, cycle_duration)
for i in range(KBar_dic['time'].size):
    kbar.AddPrice(KBar_dic['time'][i], KBar_dic['open'][i], KBar_dic['close'][i], KBar_dic['low'][i], KBar_dic['high'][i], KBar_dic['volume'][i])

# Update KBar_dic with aggregated data
KBar_dic = {
    'time': np.array(kbar.TAKBar['time']),
    'open': np.array(kbar.TAKBar['open']),
    'high': np.array(kbar.TAKBar['high']),
    'low': np.array(kbar.TAKBar['low']),
    'close': np.array(kbar.TAKBar['close']),
    'volume': np.array(kbar.TAKBar['volume']),
    'product': np.repeat('tsmc', len(kbar.TAKBar['time']))
}

# Convert to DataFrame for analysis
KBar_df = pd.DataFrame(KBar_dic)

# Moving Average strategy
st.subheader("設定計算長移動平均線(MA)的 K 棒數目(整數, 例如 10)")
LongMAPeriod = st.number_input('輸入長移動平均線的 K 棒數目', min_value=0, max_value=100, value=10)

st.subheader("設定計算短移動平均線(MA)的 K 棒數目(整數, 例如 2)")
ShortMAPeriod = st.number_input('輸入短移動平均線的 K 棒數目', min_value=0, max_value=100, value=2)

KBar_df['MA_long'] = KBar_df['close'].rolling(window=LongMAPeriod).mean()
KBar_df['MA_short'] = KBar_df['close'].rolling(window=ShortMAPeriod).mean()
last_nan_index_MA = KBar_df['MA_long'][::-1].index[KBar_df['MA_long'][::-1].apply(pd.isna)][0]

# RSI strategy
st.subheader("設定計算長RSI的 K 棒數目(整數, 例如 10)")
LongRSIPeriod = st.slider('選擇一個整數', 0, 1000, 10)
st.subheader("設定計算短RSI的 K 棒數目(整數, 例如 2)")
ShortRSIPeriod = st.slider('選擇一個整數', 0, 1000, 2)

def calculate_rsi(df, period=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

KBar_df['RSI_long'] = calculate_rsi(KBar_df, LongRSIPeriod)
KBar_df['RSI_short'] = calculate_rsi(KBar_df, ShortRSIPeriod)
KBar_df['RSI_Middle'] = np.array([50] * len(KBar_dic['time']))
last_nan_index_RSI = KBar_df['RSI_long'][::-1].index[KBar_df['RSI_long'][::-1].apply(pd.isna)][0]

# Bollinger Bands strategy
def Calculate_Bollinger_Bands(df, period=20, num_std_dev=2):
    df['SMA'] = df['close'].rolling(window=period).mean()
    df['Standard_Deviation'] = df['close'].rolling(window=period).std()
    df['Upper_Band'] = df['SMA'] + (df['Standard_Deviation'] * num_std_dev)
    df['Lower_Band'] = df['SMA'] - (df['Standard_Deviation'] * num_std_dev)
    return df

with st.expander("設定布林通道(Bollinger Band)相關參數:"):
    period = st.slider('設定計算布林通道(Bollinger Band)上中下三通道之K棒週期數目(整數, 例如 20)', 0, 100, 20, key='BB_period')
    num_std_dev = st.slider('設定計算布林通道(Bollinger Band)上中(或下中)通道之帶寬(例如 2 代表上中通道寬度為2倍的標準差)', 0, 100, 2, key='BB_heigh')

KBar_df = Calculate_Bollinger_Bands(KBar_df, period, num_std_dev)
last_nan_index_BB = KBar_df['SMA'][::-1].index[KBar_df['SMA'][::-1].apply(pd.isna)][0]

# Convert column names to title case
KBar_df.columns = [i[0].upper() + i[1:] for i in KBar_df.columns]

# Plotting
st.subheader("畫圖")

# Plot K-line chart with Moving Averages
with st.expander("K線圖, 移動平均線"):
    fig1 = make_subplots(specs=[[{"secondary_y": True}]])
    fig1.add_trace(go.Candlestick(x=KBar_df['Time'],
                                  open=KBar_df['Open'], high=KBar_df['High'],
                                  low=KBar_df['Low'], close=KBar_df['Close'], name='K線'),
                   secondary_y=True)
    fig1.add_trace(go.Bar(x=KBar_df['Time'], y=KBar_df['Volume'], name='成交量', marker=dict(color='black')), secondary_y=False)
    fig1.add_trace(go.Scatter(x=KBar_df['Time'][last_nan_index_MA+1:], y=KBar_df['MA_long'][last_nan_index_MA+1:], mode='lines', line=dict(color='orange', width=2), name=f'{LongMAPeriod}-根 K棒 移動平均線'), secondary_y=True)
    fig1.add_trace(go.Scatter(x=KBar_df['Time'][last_nan_index_MA+1:], y=KBar_df['MA_short'][last_nan_index_MA+1:], mode='lines', line=dict(color='pink', width=2), name=f'{ShortMAPeriod}-根 K棒 移動平均線'), secondary_y=True)
    fig1.layout.yaxis2.showgrid = True
    st.plotly_chart(fig1, use_container_width=True)

# Plot K-line chart with RSI
with st.expander("K線圖, 長短 RSI"):
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    fig2.add_trace(go.Candlestick(x=KBar_df['Time'],
                                  open=KBar_df['Open'], high=KBar_df['High'],
                                  low=KBar_df['Low'], close=KBar_df['Close'], name='K線'),
                   secondary_y=True)
    fig2.add_trace(go.Bar(x=KBar_df['Time'], y=KBar_df['Volume'], name='成交量', marker=dict(color='black')), secondary_y=False)
    fig2.add_trace(go.Scatter(x=KBar_df['Time'][last_nan_index_RSI+1:], y=KBar_df['RSI_long'][last_nan_index_RSI+1:], mode='lines', line=dict(color='green', width=2), name=f'{LongRSIPeriod}-根 K棒 RSI'), secondary_y=True)
    fig2.add_trace(go.Scatter(x=KBar_df['Time'][last_nan_index_RSI+1:], y=KBar_df['RSI_short'][last_nan_index_RSI+1:], mode='lines', line=dict(color='yellow', width=2), name=f'{ShortRSIPeriod}-根 K棒 RSI'), secondary_y=True)
    fig2.add_trace(go.Scatter(x=KBar_df['Time'][last_nan_index_RSI+1:], y=KBar_df['RSI_Middle'][last_nan_index_RSI+1:], mode='lines', line=dict(color='grey', width=2), name='50中線'), secondary_y=True)
    fig2.layout.yaxis2.showgrid = True
    st.plotly_chart(fig2, use_container_width=True)

# Plot K-line chart with Bollinger Bands
with st.expander("K線圖,布林通道"):
    fig3 = make_subplots(specs=[[{"secondary_y": True}]])
    fig3.add_trace(go.Candlestick(x=KBar_df['Time'],
                    open=KBar_df['Open'], high=KBar_df['High'],
                    low=KBar_df['Low'], close=KBar_df['Close'], name='K線'),
                    secondary_y=True)    
    fig3.add_trace(go.Scatter(x=KBar_df['Time'][last_nan_index_BB+1:], y=KBar_df['SMA'][last_nan_index_BB+1:], mode='lines',line=dict(color='black', width=2), name='布林通道中軌道'), 
                  secondary_y=False)
    fig3.add_trace(go.Scatter(x=KBar_df['Time'][last_nan_index_BB+1:], y=KBar_df['Upper_Band'][last_nan_index_BB+1:], mode='lines',line=dict(color='red', width=2), name='布林通道上軌道'), 
                  secondary_y=False)
    fig3.add_trace(go.Scatter(x=KBar_df['Time'][last_nan_index_BB+1:], y=KBar_df['Lower_Band'][last_nan_index_BB+1:], mode='lines',line=dict(color='blue', width=2), name='布林通道下軌道'), 
                  secondary_y=False)
    
    fig3.layout.yaxis2.showgrid = True

    st.plotly_chart(fig3, use_container_width=True)
