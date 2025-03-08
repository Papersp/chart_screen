import requests
import os
from tensorflow.keras.models import load_model
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

# ✅ Improved version: Load ML model (downloads if missing)
def load_ml_model():
    model_path = "pattern_model.h5"
    github_url = "https://raw.githubusercontent.com/Papersp/chart_screen/main/app_analizer/pattern_model.h5"

    # Check if the model file exists locally
    if not os.path.exists(model_path):
        # Download the model file from GitHub
        response = requests.get(github_url)
        if response.status_code == 200:
            with open(model_path, "wb") as f:
                f.write(response.content)
        else:
            st.error("Failed to download ML model. Make sure the file exists on GitHub.")
            return None

    # Load the model
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading ML model: {e}")
        return None

# Fetch market data from yfinance
def fetch_yfinance_data(symbol, period='1mo', interval='1d'):
    df = yf.download(symbol, period=period, interval=interval)
    df.reset_index(inplace=True)
    return df

# Fetch market data from Binance API
def fetch_binance_data(symbol, interval='1h', limit=100):
    url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}'
    response = requests.get(url).json()
    df = pd.DataFrame(response, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', '_', '_', '_', '_', '_', '_'])
    df = df[['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']]
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# Placeholder for Alpha Vantage / OANDA integration
def fetch_forex_data(symbol):
    return pd.DataFrame()

# Detect patterns using ML model
def detect_patterns(df, model):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df[['Open', 'High', 'Low', 'Close', 'Volume']].values)
    predictions = model.predict(np.expand_dims(data_scaled, axis=0))
    return predictions

# Plot candlestick chart
def plot_candlestick(df, title="Candlestick Chart"):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df['timestamp'] if 'timestamp' in df else df['Date'],
        open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='Candlesticks'
    ))
    st.plotly_chart(fig)

# Streamlit UI
def main():
    st.title("Financial Market Visualizer")
    symbol = st.text_input("Enter Symbol (e.g., AAPL, BTCUSDT, EURUSD)", "AAPL")
    data_source = st.selectbox("Select Data Source", ["Yahoo Finance", "Binance", "Forex (Alpha Vantage/OANDA)"])
    interval = st.selectbox("Select Timeframe", ["1d", "1h", "5m"])
    model = load_ml_model()
    
    if st.button("Analyze Chart"):
        if data_source == "Yahoo Finance":
            df = fetch_yfinance_data(symbol)
        elif data_source == "Binance":
            df = fetch_binance_data(symbol, interval)
        else:
            df = fetch_forex_data(symbol)
        
        if not df.empty:
            plot_candlestick(df, title=f"{symbol} Candlestick Chart")
            if model:
                patterns = detect_patterns(df, model)
                st.write("Pattern Predictions:", patterns)

if __name__ == "__main__":
    main()
