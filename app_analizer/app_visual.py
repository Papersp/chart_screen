import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from datetime import datetime, timedelta
import json
import os
from io import BytesIO
from scipy.signal import find_peaks
import ta
from sklearn.linear_model import LinearRegression

# App title and configuration
st.set_page_config(
    page_title="Financial Market Visualizer",
    page_icon="📈",
    layout="wide"
)

# Initialize session state variables if they don't exist
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = pd.DataFrame(columns=['Symbol', 'Shares', 'Purchase Price', 'Purchase Date'])
if 'saved_configs' not in st.session_state:
    st.session_state.saved_configs = {}
if 'chart_tickers' not in st.session_state:
    st.session_state.chart_tickers = ['AAPL']
if 'detected_patterns' not in st.session_state:
    st.session_state.detected_patterns = {}

# Sidebar for app navigation
st.sidebar.title("Financial Market Visualizer")
app_mode = st.sidebar.selectbox("Select Mode", ["Chart Analysis", "Multi-Chart View", "Portfolio Tracker", "Correlation Analysis"])

# Helper Functions
def calculate_technical_indicators(df):
    """Calculate technical indicators for a dataframe with OHLC data"""
    # Moving Averages
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    # Bollinger Bands
    df['MA20_std'] = df['Close'].rolling(window=20).std()
    df['upper_band'] = df['MA20'] + (df['MA20_std'] * 2)
    df['lower_band'] = df['MA20'] - (df['MA20_std'] * 2)
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Add volume analysis
    df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
    
    return df

def download_chart(fig):
    """Generate a download link for a Plotly figure"""
    buffer = BytesIO()
    fig.write_image(buffer, format="png", width=1200, height=800)
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="chart.png">Download Chart as PNG</a>'
    return href

def export_html(fig):
    """Export a Plotly figure as HTML"""
    html_str = fig.to_html(full_html=False, include_plotlyjs='cdn')
    b64 = base64.b64encode(html_str.encode()).decode()
    href = f'<a href="data:text/html;base64,{b64}" download="chart.html">Download Chart as HTML</a>'
    return href

def calculate_financial_ratios(ticker):
    """Calculate key financial ratios for a stock"""
    stock = yf.Ticker(ticker)
    info = stock.info
    
    # Basic financial information
    ratios = {}
    
    # Market ratios
    try:
        ratios['Market Cap'] = info.get('marketCap', 'N/A')
        ratios['P/E Ratio'] = info.get('trailingPE', 'N/A') 
        ratios['Forward P/E'] = info.get('forwardPE', 'N/A')
        ratios['PEG Ratio'] = info.get('pegRatio', 'N/A')
        ratios['Price to Book'] = info.get('priceToBook', 'N/A')
        ratios['Price to Sales'] = info.get('priceToSalesTrailing12Months', 'N/A')
        
        # Dividend information
        ratios['Dividend Yield'] = info.get('dividendYield', 'N/A')
        if ratios['Dividend Yield'] != 'N/A':
            ratios['Dividend Yield'] = f"{ratios['Dividend Yield'] * 100:.2f}%"
        
        # Profitability ratios
        ratios['Profit Margin'] = info.get('profitMargins', 'N/A')
        if ratios['Profit Margin'] != 'N/A':
            ratios['Profit Margin'] = f"{ratios['Profit Margin'] * 100:.2f}%"
        
        ratios['Operating Margin'] = info.get('operatingMargins', 'N/A')
        if ratios['Operating Margin'] != 'N/A':
            ratios['Operating Margin'] = f"{ratios['Operating Margin'] * 100:.2f}%"
        
        # Return ratios
        ratios['ROA'] = info.get('returnOnAssets', 'N/A')
        if ratios['ROA'] != 'N/A':
            ratios['ROA'] = f"{ratios['ROA'] * 100:.2f}%"
            
        ratios['ROE'] = info.get('returnOnEquity', 'N/A')
        if ratios['ROE'] != 'N/A':
            ratios['ROE'] = f"{ratios['ROE'] * 100:.2f}%"
        
        # Growth
        ratios['EPS Growth'] = info.get('earningsGrowth', 'N/A')
        if ratios['EPS Growth'] != 'N/A':
            ratios['EPS Growth'] = f"{ratios['EPS Growth'] * 100:.2f}%"
        
        ratios['Revenue Growth'] = info.get('revenueGrowth', 'N/A')
        if ratios['Revenue Growth'] != 'N/A':
            ratios['Revenue Growth'] = f"{ratios['Revenue Growth'] * 100:.2f}%"
            
    except Exception as e:
        st.error(f"Error retrieving financial ratios: {e}")
        
    return ratios

def load_portfolio_data(uploaded_file):
    """Load portfolio data from CSV or Excel file"""
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(uploaded_file)
    else:
        st.error("Unsupported file format. Please upload a CSV or Excel file.")
        return None
    
    # Verify the required columns
    required_cols = ['Symbol', 'Shares', 'Purchase Price', 'Purchase Date']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.error(f"Missing required columns: {', '.join(missing_cols)}")
        return None
    
    return df

def save_config(config_name, config_data):
    """Save the current configuration"""
    st.session_state.saved_configs[config_name] = config_data
    st.success(f"Configuration '{config_name}' saved successfully!")

def load_config(config_name):
    """Load a saved configuration"""
    return st.session_state.saved_configs.get(config_name)

def share_config(config_name):
    """Generate a shareable link for a saved configuration"""
    if config_name in st.session_state.saved_configs:
        config_data = st.session_state.saved_configs[config_name]
        config_json = json.dumps(config_data)
        config_b64 = base64.b64encode(config_json.encode()).decode()
        share_text = f"Use this configuration in Financial Market Visualizer: {config_b64}"
        st.text_area("Share this configuration", share_text, height=100)
    else:
        st.error("Configuration not found!")

# Pattern Recognition Functions
def find_peaks_troughs(df, window=10, prominence=0.5):
    """Find peaks and troughs in price data for pattern recognition"""
    # Scale prominence to the data
    price_range = df['High'].max() - df['Low'].min()
    scaled_prominence = price_range * prominence / 100
    
    # Find peaks (highs)
    peaks, _ = find_peaks(df['High'].values, distance=window, prominence=scaled_prominence)
    peak_prices = df['High'].iloc[peaks]
    peak_dates = df.index[peaks]
    
    # Find troughs (lows)
    inverted = -df['Low'].values
    troughs, _ = find_peaks(inverted, distance=window, prominence=scaled_prominence)
    trough_prices = df['Low'].iloc[troughs]
    trough_dates = df.index[troughs]
    
    return {
        'peak_dates': peak_dates,
        'peak_prices': peak_prices,
        'trough_dates': trough_dates,
        'trough_prices': trough_prices
    }

def detect_horizontal_supports_resistances(df, pts, tolerance=0.02):
    """Detect horizontal support and resistance levels"""
    horizontal_levels = []
    
    # Check peaks for resistance levels
    for i in range(len(pts['peak_dates'])):
        price = pts['peak_prices'].iloc[i]
        date = pts['peak_dates'][i]
        
        # Find other peaks at similar price levels
        similar_peaks = [j for j in range(len(pts['peak_dates'])) 
                         if abs(pts['peak_prices'].iloc[j] - price) / price < tolerance 
                         and j != i]
        
        if len(similar_peaks) >= 1:  # At least 2 peaks at similar level
            # Check if this level is already added
            if not any(abs(level['price'] - price) / price < tolerance for level in horizontal_levels):
                horizontal_levels.append({
                    'type': 'resistance',
                    'price': price,
                    'start_date': df.index[0],
                    'end_date': df.index[-1],
                    'strength': len(similar_peaks) + 1  # Number of touches
                })
    
    # Check troughs for support levels
    for i in range(len(pts['trough_dates'])):
        price = pts['trough_prices'].iloc[i]
        date = pts['trough_dates'][i]
        
        # Find other troughs at similar price levels
        similar_troughs = [j for j in range(len(pts['trough_dates'])) 
                         if abs(pts['trough_prices'].iloc[j] - price) / price < tolerance 
                         and j != i]
        
        if len(similar_troughs) >= 1:  # At least 2 troughs at similar level
            # Check if this level is already added
            if not any(abs(level['price'] - price) / price < tolerance for level in horizontal_levels):
                horizontal_levels.append({
                    'type': 'support',
                    'price': price,
                    'start_date': df.index[0],
                    'end_date': df.index[-1],
                    'strength': len(similar_troughs) + 1  # Number of touches
                })
    
    return horizontal_levels

def detect_trend_sr_lines(df, pts, min_points=3):
    """Detect trend lines for support and resistance"""
    trend_lines = []
    
    # Function to fit line to points and check quality
    def fit_line(dates, prices, is_support=True):
        if len(dates) < min_points:
            return None
        
        # Convert dates to numeric format for regression
        x = np.array([(date - df.index[0]).total_seconds() for date in dates]).reshape(-1, 1)
        y = np.array(prices)
        
        # Fit linear regression
        model = LinearRegression().fit(x, y)
        
        # Calculate R-squared to measure fit quality
        y_pred = model.predict(x)
        r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
        
        # Only add lines with good fit
        if r2 > 0.7:
            # Calculate start and end points
            start_x = (df.index[0] - df.index[0]).total_seconds()
            end_x = (df.index[-1] - df.index[0]).total_seconds()
            
            start_y = model.predict([[start_x]])[0]
            end_y = model.predict([[end_x]])[0]
            
            return {
                'type': 'support' if is_support else 'resistance',
                'start_date': df.index[0],
                'start_price': start_y,
                'end_date': df.index[-1],
                'end_price': end_y,
                'slope': model.coef_[0],
                'r2': r2,
                'strength': len(dates)  # Number of touches
            }
        return None
    
    # Try different combinations of peak points for resistance
    for i in range(len(pts['peak_dates'])):
        for j in range(i+1, len(pts['peak_dates'])):
            # Use two points to establish potential line
            dates = [pts['peak_dates'][i], pts['peak_dates'][j]]
            prices = [pts['peak_prices'].iloc[i], pts['peak_prices'].iloc[j]]
            
            # Calculate line equation
            slope = (prices[1] - prices[0]) / (dates[1] - dates[0]).total_seconds()
            intercept = prices[0] - slope * (dates[0] - df.index[0]).total_seconds()
            
            # Find other peaks near this line
            for k in range(len(pts['peak_dates'])):
                if k != i and k != j:
                    x = (pts['peak_dates'][k] - df.index[0]).total_seconds()
                    expected_y = slope * x + intercept
                    actual_y = pts['peak_prices'].iloc[k]
                    
                    # If point is close to line, add it
                    if abs(expected_y - actual_y) / actual_y < 0.02:
                        dates.append(pts['peak_dates'][k])
                        prices.append(pts['peak_prices'].iloc[k])
            
            # Try to fit line to collected points
            line = fit_line(dates, prices, False)
            if line:
                trend_lines.append(line)
    
    # Try different combinations of trough points for support
    for i in range(len(pts['trough_dates'])):
        for j in range(i+1, len(pts['trough_dates'])):
            # Use two points to establish potential line
            dates = [pts['trough_dates'][i], pts['trough_dates'][j]]
            prices = [pts['trough_prices'].iloc[i], pts['trough_prices'].iloc[j]]
            
            # Calculate line equation
            slope = (prices[1] - prices[0]) / (dates[1] - dates[0]).total_seconds()
            intercept = prices[0] - slope * (dates[0] - df.index[0]).total_seconds()
            
            # Find other troughs near this line
            for k in range(len(pts['trough_dates'])):
                if k != i and k != j:
                    x = (pts['trough_dates'][k] - df.index[0]).total_seconds()
                    expected_y = slope * x + intercept
                    actual_y = pts['trough_prices'].iloc[k]
                    
                    # If point is close to line, add it
                    if abs(expected_y - actual_y) / actual_y < 0.02:
                        dates.append(pts['trough_dates'][k])
                        prices.append(pts['trough_prices'].iloc[k])
            
            # Try to fit line to collected points
            line = fit_line(dates, prices, True)
            if line:
                trend_lines.append(line)
    
    return trend_lines

def detect_channels(df, trend_lines, tolerance=0.05):
    """Detect price channels using parallel trend lines"""
    channels = []
    
    # Compare support and resistance trend lines
    supports = [line for line in trend_lines if line['type'] == 'support']
    resistances = [line for line in trend_lines if line['type'] == 'resistance']
    
    for support in supports:
        for resistance in resistances:
            # Check if slopes are similar (parallel lines)
            if abs(support['slope'] - resistance['slope']) / abs(support['slope']) < tolerance:
                # Check if resistance is above support
                if (resistance['start_price'] > support['start_price'] and 
                    resistance['end_price'] > support['end_price']):
                    
                    direction = "up" if support['slope'] > 0 else "down"
                    
                    channels.append({
                        'type': f'channel_{direction}',
                        'support': support,
                        'resistance': resistance,
                        'slope': support['slope'],
                        'height': resistance['start_price'] - support['start_price'],
                        'start_date': max(support['start_date'], resistance['start_date']),
                        'end_date': min(support['end_date'], resistance['end_date'])
                    })
    
    return channels

def detect_wedges_triangles(df, trend_lines):
    """Detect wedges and triangles patterns using converging trend lines"""
    patterns = []
    
    # Compare support and resistance trend lines
    supports = [line for line in trend_lines if line['type'] == 'support']
    resistances = [line for line in trend_lines if line['type'] == 'resistance']
    
    for support in supports:
        for resistance in resistances:
            # Calculate intersection point
            if support['slope'] == resistance['slope']:
                continue  # Parallel lines don't intersect
                
            # Calculate time of intersection
            t_intersect = (support['start_price'] - resistance['start_price']) / (resistance['slope'] - support['slope'])
            intersect_date = df.index[0] + timedelta(seconds=t_intersect)
            
            # Check if intersection is within reasonable future timeframe (not too far)
            max_future = df.index[-1] + timedelta(days=30)
            if intersect_date < max_future and intersect_date > df.index[-1]:
                # Determine pattern type based on slopes
                if support['slope'] > 0 and resistance['slope'] < 0:
                    pattern_type = 'wedge_up' if resistance['start_price'] > support['start_price'] else 'triangle_ascending'
                elif support['slope'] < 0 and resistance['slope'] > 0:
                    pattern_type = 'wedge_down' if resistance['start_price'] > support['start_price'] else 'triangle_descending'
                else:
                    # Both lines have same sign slope
                    if support['slope'] > 0 and resistance['slope'] > 0:
                        pattern_type = 'wedge_up' if support['slope'] > resistance['slope'] else 'wedge_down'
                    else:  # Both negative
                        pattern_type = 'wedge_down' if support['slope'] < resistance['slope'] else 'wedge_up'
                
                patterns.append({
                    'type': pattern_type,
                    'support': support,
                    'resistance': resistance,
                    'intersect_date': intersect_date,
                    'start_date': max(support['start_date'], resistance['start_date']),
                    'end_date': df.index[-1]
                })
    
    return patterns

def detect_double_patterns(df, pts, window=20):
    """Detect double tops and bottoms"""
    patterns = []
    
    # Detect double tops
    for i in range(len(pts['peak_dates']) - 1):
        for j in range(i + 1, len(pts['peak_dates'])):
            date1 = pts['peak_dates'][i]
            date2 = pts['peak_dates'][j]
            price1 = pts['peak_prices'].iloc[i]
            price2 = pts['peak_prices'].iloc[j]
            
            # Check if dates are within range but not too close
            time_diff = (date2 - date1).days
            if 5 <= time_diff <= window:
                # Check if prices are similar
                if abs(price2 - price1) / price1 < 0.03:
                    # Check for a valley in between
                    between_idx = df.index[(df.index > date1) & (df.index < date2)]
                    if len(between_idx) > 0:
                        between_min = df.loc[between_idx, 'Low'].min()
                        if (price1 - between_min) / price1 > 0.03:  # At least 3% drop between peaks
                            patterns.append({
                                'type': 'double_top',
                                'date1': date1,
                                'date2': date2,
                                'price1': price1,
                                'price2': price2,
                                'valley_date': df.loc[between_idx, 'Low'].idxmin(),
                                'valley_price': between_min
                            })
    
    # Detect double bottoms
    for i in range(len(pts['trough_dates']) - 1):
        for j in range(i + 1, len(pts['trough_dates'])):
            date1 = pts['trough_dates'][i]
            date2 = pts['trough_dates'][j]
            price1 = pts['trough_prices'].iloc[i]
            price2 = pts['trough_prices'].iloc[j]
            
            # Check if dates are within range but not too close
            time_diff = (date2 - date1).days
            if 5 <= time_diff <= window:
                # Check if prices are similar
                if abs(price2 - price1) / price1 < 0.03:
                    # Check for a peak in between
                    between_idx = df.index[(df.index > date1) & (df.index < date2)]
                    if len(between_idx) > 0:
                        between_max = df.loc[between_idx, 'High'].max()
                        if (between_max - price1) / price1 > 0.03:  # At least 3% rise between troughs
                            patterns.append({
                                'type': 'double_bottom',
                                'date1': date1,
                                'date2': date2,
                                'price1': price1,
                                'price2': price2,
                                'peak_date': df.loc[between_idx, 'High'].idxmax(),
                                'peak_price': between_max
                            })
    
    return patterns

def detect_head_shoulders(df, pts, tolerance=0.05):
    """Detect head and shoulders patterns"""
    patterns = []
    
    # Head and shoulders pattern (bearish)
    for i in range(len(pts['peak_dates']) - 2):
        # Look for 3 consecutive peaks with the middle one higher
        date_left = pts['peak_dates'][i]
        price_left = pts['peak_prices'].iloc[i]
        
        for j in range(i + 1, len(pts['peak_dates']) - 1):
            date_head = pts['peak_dates'][j]
            price_head = pts['peak_prices'].iloc[j]
            
            # Head should be higher than left shoulder
            if price_head <= price_left:
                continue
                
            for k in range(j + 1, len(pts['peak_dates'])):
                date_right = pts['peak_dates'][k]
                price_right = pts['peak_prices'].iloc[k]
                
                # Right shoulder should be similar to left
                if abs(price_right - price_left) / price_left > tolerance:
                    continue
                    
                # Find neckline (connecting troughs between shoulders and head)
                between_left_head = df.index[(df.index > date_left) & (df.index < date_head)]
                between_head_right = df.index[(df.index > date_head) & (df.index < date_right)]
                
                if len(between_left_head) > 0 and len(between_head_right) > 0:
                    trough_left = df.loc[between_left_head, 'Low'].min()
                    trough_right = df.loc[between_head_right, 'Low'].min()
                    
                    date_trough_left = df.loc[between_left_head, 'Low'].idxmin()
                    date_trough_right = df.loc[between_head_right, 'Low'].idxmin()
                    
                    # Neckline should be somewhat horizontal
                    neckline_slope = (trough_right - trough_left) / (date_trough_right - date_trough_left).total_seconds()
                    
                    if abs(neckline_slope) < 0.0001:  # Very low slope
                        patterns.append({
                            'type': 'head_shoulders',
                            'left_shoulder_date': date_left,
                            'left_shoulder_price': price_left,
                            'head_date': date_head,
                            'head_price': price_head,
                            'right_shoulder_date': date_right,
                            'right_shoulder_price': price_right,
                            'neckline_left_date': date_trough_left,
                            'neckline_left_price': trough_left,
                            'neckline_right_date': date_trough_right,
                            'neckline_right_price': trough_right,
                            'neckline_slope': neckline_slope
                        })
    
    # Inverse head and shoulders pattern (bullish)
    for i in range(len(pts['trough_dates']) - 2):
        # Look for 3 consecutive troughs with the middle one lower
        date_left = pts['trough_dates'][i]
        price_left = pts['trough_prices'].iloc[i]
        
        for j in range(i + 1, len(pts['trough_dates']) - 1):
            date_head = pts['trough_dates'][j]
            price_head = pts['trough_prices'].iloc[j]
            
            # Head should be lower than left shoulder
            if price_head >= price_left:
                continue
                
            for k in range(j + 1, len(pts['trough_dates'])):
                date_right = pts['trough_dates'][k]
                price_right = pts['trough_prices'].iloc[k]
                
                # Right shoulder should be similar to left
                if abs(price_right - price_left) / price_left > tolerance:
                    continue
                    
                # Find neckline (connecting peaks between shoulders and head)
                between_left_head = df.index[(df.index > date_left) & (df.index < date_head)]
                between_head_right = df.index[(df.index > date_head) & (df.index < date_right)]
                
                if len(between_left_head) > 0 and len(between_head_right) > 0:
                    peak_left = df.loc[between_left_head, 'High'].max()
                    peak_right = df.loc[between_head_right, 'High'].max()
                    
                    date_peak_left = df.loc[between_left_head, 'High'].idxmax()
                    date_peak_right = df.loc[between_head_right, 'High'].idxmax()
                    
                    # Neckline should be somewhat horizontal
                    neckline_slope = (peak_right - peak_left) / (date_peak_right - date_peak_left).total_seconds()
                    
                    if abs(neckline_slope) < 0.0001:  # Very low slope
                        patterns.append({
                            'type': 'inverse_head_shoulders',
                            'left_shoulder_date': date_left,
                            'left_shoulder_price': price_left,
                            'head_date': date_head,
                            'head_price': price_head,
                            'right_shoulder_date': date_right,
                            'right_shoulder_price': price_right,
                            'neckline_left_date': date_peak_left,
                            'neckline_left_price': peak_left,
                            'neckline_right_date': date_peak_right,
                            'neckline_right_price': peak_right,
                            'neckline_slope': neckline_slope
                        })
    
    return patterns

def analyze_patterns(df):
    """Main function to detect and analyze all chart patterns"""
    patterns = {}
    
    # Find significant price points
    pts = find_peaks_troughs(df)
    
    # Detect horizontal support/resistance
    patterns['horizontal_sr'] = detect_horizontal_supports_resistances(df, pts)
    
    # Detect trend lines
    trend_lines = detect_trend_sr_lines(df, pts)
    patterns['trend_sr'] = trend_lines
    
    # Detect channels
    patterns['channels'] = detect_channels(df, trend_lines)
    
    # Detect wedges and triangles
    patterns['wedges_triangles'] = detect_wedges_triangles(df, trend_lines)
    
    # Detect double tops/bottoms
    patterns['double_patterns'] = detect_double_patterns(df, pts)
    
    # Detect head and shoulders
    patterns['head_shoulders'] = detect_head_shoulders(df, pts)
    
    return patterns

def draw_patterns_on_chart(fig, df, patterns):
    """Add pattern visualization to the chart"""
    # Color scheme for patterns
    colors = {
        'support': 'green',
        'resistance': 'red',
        'channel_up': 'rgba(0, 128, 0, 0.3)',  # Semi-transparent green
        'channel_down': 'rgba(255, 0, 0, 0.3)',  # Semi-transparent red
        'wedge_up': 'rgba(0, 128, 255, 0.3)',  # Semi-transparent blue
        'wedge_down': 'rgba(255, 165, 0, 0.3)',  # Semi-transparent orange
        'triangle_ascending': 'rgba(153, 51, 255, 0.3)',  # Semi-transparent purple
        'triangle_descending': 'rgba(255, 102, 102, 0.3)',  # Semi-transparent light red
        'double_top': 'rgba(255, 0, 0, 0.7)',  # Stronger red
        'double_bottom': 'rgba(0, 128, 0, 0.7)',  # Stronger green
        'head_shoulders': 'rgba(139, 0, 0, 0.7)',  # Dark red
        'inverse_head_shoulders': 'rgba(0, 100,
