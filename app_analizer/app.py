import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import talib
import requests

# Function to fetch live market data (example using Yahoo Finance API)
def fetch_market_data(symbol, interval='1d', limit=100):
    url = f'https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval={interval}&range=1mo'
    response = requests.get(url).json()
    try:
        timestamps = response['chart']['result'][0]['timestamp']
        ohlc = response['chart']['result'][0]['indicators']['quote'][0]
        df = pd.DataFrame(ohlc, index=pd.to_datetime(timestamps, unit='s'))
        df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
        return df.dropna()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Function to detect double tops and bottoms
def detect_double_top_bottom(df):
    peaks = talib.MAX(df['High'], timeperiod=10)
    troughs = talib.MIN(df['Low'], timeperiod=10)
    return peaks, troughs

# Function to detect head and shoulders
def detect_head_and_shoulders(df):
    df['SMA50'] = talib.SMA(df['Close'], timeperiod=50)
    df['SMA200'] = talib.SMA(df['Close'], timeperiod=200)
    df['H&S'] = (df['SMA50'] > df['SMA200']) & (df['Close'].shift(1) > df['Close']) & (df['Close'].shift(-1) > df['Close'])
    return df['H&S']

# Function to plot candlestick chart
def plot_candlestick(df, title="Candlestick Chart"):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Candlesticks'))
    peaks, troughs = detect_double_top_bottom(df)
    fig.add_trace(go.Scatter(x=df.index, y=peaks, mode='markers', marker=dict(color='red', size=8), name='Double Tops'))
    fig.add_trace(go.Scatter(x=df.index, y=troughs, mode='markers', marker=dict(color='green', size=8), name='Double Bottoms'))
    st.plotly_chart(fig)

# Streamlit UI
def main():
    st.title("Real-Time Market Pattern Recognition")
    symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, TSLA, BTC-USD)", "AAPL")
    if st.button("Analyze Chart"):
        df = fetch_market_data(symbol)
        if df is not None:
            plot_candlestick(df, title=f"{symbol} Candlestick Chart")
            if detect_head_and_shoulders(df).any():
                st.warning("Head & Shoulders pattern detected!")

if __name__ == "__main__":
    main()
