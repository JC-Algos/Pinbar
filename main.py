# pinbar_detector.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from io import BytesIO

st.set_page_config(layout="wide", page_title="Pinbar Pattern Detection Tool")

# ---------- 1.  TICKER LISTS ----------
WORLD_TICKERS = ["^GSPC","^NDX","^RUT","^HSI","3032.HK","^STOXX50E","^BSESN","^KS11",
                 "^TWII","000300.SS","^N225","HYG","AGG","EEM","GDX","XLE","XME","AAXJ","IBB",
                 "DBA","TLT","EFA","EWZ","EWG","EWJ","EWY","EWT","EWQ","EWA","EWC","EWH",
                 "EWS","EIDO","EPHE","THD","INDA","KWEB","QQQ","SPY","IWM","VNQ","GLD","SLV",
                 "USO","UNG","VEA","VWO","VTI","VXUS"][:150]

US_TICKERS = ["AAPL", "ABBV", "ABNB", "ABSV", "ABT", "ACN", "ADBE", "ADP", "ADSK", "ALGN", 
              "AMAT", "AMD", "AMGN", "AMZN", "AMT", "ANET", "APA", "ARGO", "ARM", "AS", 
              "ASML", "AVGO", "BA", "BAC", "BDX", "BLK", "BKNG", "BMRG", "BMY", "BRK-B", 
              "CAT", "CB", "CCL", "CDNS", "CF", "CHTR", "CME", "COP", "COST", "CRM", 
              "CRWD", "CSCO", "CSX", "CVS", "CVX", "DDOG", "DE", "DHR", "DIS", "DLTR", 
              "DVN", "DXCM", "EOG", "EXM", "F", "FANG", "FCX", "FDD", "FTNT", "FUTU", 
              "G", "GE", "GILD", "GIS", "GM", "GOOGL", "GS", "HAL", "HD", "HON", "HSY", 
              "IBM", "ICE", "IDXX", "INTC", "INTU", "ISRG", "ITW", "IWM", "JNJ", "JPM", 
              "JPU", "KD", "KHC", "KMB", "KMI", "KO", "LEN", "LIAT", "LLY", "LMT", "LOW", 
              "LRCX", "MA", "MAR", "MCD", "MDLZ", "META", "MMM", "MRK", "MRO", "MSFT", 
              "MU", "NEE", "NFLX", "NKE", "NOW", "NRG", "NVO", "NVDA", "NXTR", "ORCL", 
              "ORLY", "OXY", "PANW", "PEP", "PFE", "PG", "PGR", "PLTR", "PM", "PSX", 
              "QCOM", "REGN", "RTX", "SBUX", "SLB", "SMH", "SNOW", "SPGI", "TGT", "TJX", 
              "TMO", "TRV", "TSLA", "TSM", "TTD", "TTWO", "TXN", "TEAM", "ULTA", "UNH", 
              "UNP", "UPS", "V", "VLO", "VMO", "VST", "VZ", "WDH", "WMB", "WMT", "WRTC", 
              "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY", 
              "XOM", "YUM", "ZS", "BX", "COIN",
              # Additional tickers from original code...
              "GOOG", "TMUS", "AZN", "LIN", "SHOP", "PDD", "CMCSA", "APP", "MELI", "VRTX", 
              "SNYS", "KLAC", "MSTR", "ADI", "CEG", "DASH", "CTAS", "TRI", "MRVL", "PYPL", 
              "WDAY", "AEP", "MNST", "ROP", "AXON", "NXPI", "FAST", "PAYX", "PCAR", "KDP", 
              "CCEP", "ROST", "CPRT", "BKR", "EXC", "XEL", "CSGP", "EA", "MCHP", "VRSK", 
              "CTSH", "GEHC", "WBD", "ODFL", "LULU", "ON", "CDW", "GFS", "BIIB",
              "WFC", "AXP", "MS", "T", "UBER", "SCHW", "BSX", "SYK", "C", "GEV", 
              "ETN", "MMC", "APH", "MDT", "KKR", "PLD", "WELL", "MO", "SO", "TT", 
              "WM", "HCA", "FI", "DUK", "EQIX", "SHW", "MCK", "ELV", "MCO", "PH", 
              "AJG", "CI", "TDG", "AON", "RSG", "DELL", "APO", "COF", "ZTS", "ECL", 
              "RCL", "GD", "CL", "HWM", "CMG", "PNC", "NOC", "MSI", "USB", "EMR", 
              "JCI", "BK", "APD", "AZO", "SPG", "DLR", "CARR", "HLT", "NEM", "NSC", 
              "AFL", "COR", "ALL", "MET", "PWR", "PSA", "TFC", "FDX", "GWW", "OKE", 
              "O", "AIG", "SRE", "AMP", "MPC", "NDAQ"]

HK_TICKERS = ["0001.HK","0002.HK","0003.HK","0005.HK","0006.HK","0011.HK","0012.HK","0016.HK","0017.HK","0019.HK",
              "0020.HK","0027.HK","0066.HK","0101.HK","0144.HK","0168.HK","0175.HK","0177.HK","0220.HK","0241.HK",
              "0267.HK","0268.HK","0285.HK","0288.HK","0291.HK","0300.HK","0316.HK","0317.HK","0322.HK","0338.HK",
              "0358.HK","0386.HK","0388.HK","0390.HK","0489.HK","0552.HK","0598.HK","0636.HK","0669.HK","0688.HK",
              "0696.HK","0700.HK","0728.HK","0753.HK","0762.HK","0763.HK","0772.HK","0788.HK","0799.HK","0806.HK",
              "0811.HK","0823.HK","0836.HK","0857.HK","0868.HK","0883.HK","0902.HK","0914.HK","0916.HK","0921.HK",
              "0939.HK","0941.HK","0956.HK","0960.HK","0968.HK","0981.HK","0991.HK","0992.HK","0998.HK","1024.HK",
              "1033.HK","1038.HK","1044.HK","1055.HK","1066.HK","1071.HK","1072.HK","1088.HK","1093.HK","1099.HK",
              "1109.HK","1113.HK","1133.HK","1138.HK","1157.HK","1171.HK","1177.HK","1186.HK","1209.HK","1211.HK",
              "1288.HK","1299.HK","1316.HK","1336.HK","1339.HK","1347.HK","1359.HK","1378.HK","1398.HK","1515.HK",
              "1618.HK","1658.HK","1766.HK","1772.HK","1776.HK","1787.HK","1800.HK","1801.HK","1810.HK","1816.HK",
              "1818.HK","1833.HK","1860.HK","1876.HK","1880.HK","1886.HK","1898.HK","1918.HK","1919.HK","1928.HK",
              "1929.HK","1958.HK","1988.HK","1997.HK","2007.HK","2013.HK","2015.HK","2018.HK","2020.HK","2196.HK",
              "2202.HK","2208.HK","2238.HK","2252.HK","2269.HK","2313.HK","2318.HK","2319.HK","2328.HK","2331.HK",
              "2333.HK","2382.HK","2386.HK","2388.HK","2400.HK","2480.HK","2498.HK","2533.HK","2600.HK","2601.HK",
              "2607.HK","2611.HK","2628.HK","2688.HK","2689.HK","2696.HK","2727.HK","2799.HK","2845.HK","2866.HK",
              "2880.HK","2883.HK","2899.HK","3191.HK","3311.HK","3319.HK","3323.HK","3328.HK","3330.HK","3606.HK",
              "3618.HK","3690.HK","3698.HK","3800.HK","3866.HK","3880.HK","3888.HK","3898.HK","3900.HK","3908.HK",
              "3931.HK","3968.HK","3969.HK","3988.HK","3993.HK","3996.HK","6030.HK","6060.HK","6078.HK","6098.HK",
              "6099.HK","6139.HK","6160.HK","6178.HK","6181.HK","6618.HK","6655.HK","6680.HK","6682.HK","6690.HK",
              "6699.HK","6806.HK","6808.HK","6818.HK","6862.HK","6865.HK","6869.HK","6881.HK","6886.HK","6955.HK",
              "6963.HK","6969.HK","6990.HK","9600.HK","9601.HK","9618.HK","9626.HK","9633.HK","9666.HK","9668.HK",
              "9676.HK","9696.HK","9698.HK","9699.HK","9801.HK","9863.HK","9868.HK","9880.HK","9888.HK","9889.HK",
              "9901.HK","9922.HK","9923.HK","9961.HK","9988.HK","9992.HK","9995.HK","9999.HK"]

UNIVERSE_MAP = {
    "World": {"tickers": WORLD_TICKERS},
    "US":    {"tickers": US_TICKERS},
    "HK":    {"tickers": HK_TICKERS}
}

# ---------- 2.  DATA FETCHING ----------
@st.cache_data(ttl=3600)
def fetch_data(universe):
    cfg = UNIVERSE_MAP[universe]
    tickers = cfg["tickers"]
    end = datetime.today()
    
    # Weekly data - resample to week ending Friday
    weekly_start = end - timedelta(weeks=50)
    weekly_data = yf.download(tickers, start=weekly_start, end=end, progress=False)
    weekly_ohlc = {
        'Open': weekly_data['Open'].resample('W-FRI').first(),
        'High': weekly_data['High'].resample('W-FRI').max(), 
        'Low': weekly_data['Low'].resample('W-FRI').min(),
        'Close': weekly_data['Close'].resample('W-FRI').last()
    }
    
    # Daily data
    daily_start = end - timedelta(days=100)
    daily_data = yf.download(tickers, start=daily_start, end=end, progress=False)
    daily_ohlc = {
        'Open': daily_data['Open'],
        'High': daily_data['High'],
        'Low': daily_data['Low'], 
        'Close': daily_data['Close']
    }
    
    # Clean data - forward fill then backward fill
    for timeframe in [weekly_ohlc, daily_ohlc]:
        for key in timeframe:
            timeframe[key] = timeframe[key].fillna(method='ffill').fillna(method='bfill')
            timeframe[key] = timeframe[key].dropna(axis=1, how="all")
    
    return weekly_ohlc, daily_ohlc

# ---------- 3.  PINBAR DETECTION FUNCTIONS ----------
def calculate_rsi(close_prices, period=9):
    """Calculate RSI manually"""
    try:
        if len(close_prices) < period + 1:
            return np.nan
        
        # Calculate price changes
        delta = close_prices.diff()
        
        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate initial averages
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Calculate RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not np.isnan(rsi.iloc[-1]) else np.nan
    except:
        return np.nan

def calculate_atr(high_prices, low_prices, close_prices, period=9):
    """Calculate ATR manually"""
    try:
        if len(high_prices) < period + 1:
            return np.nan
        
        # Calculate True Range components
        hl = high_prices - low_prices
        hc = abs(high_prices - close_prices.shift(1))
        lc = abs(low_prices - close_prices.shift(1))
        
        # True Range is the maximum of the three
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        
        # ATR is the moving average of True Range
        atr = tr.rolling(window=period).mean()
        
        return atr.iloc[-1] if not np.isnan(atr.iloc[-1]) else np.nan
    except:
        return np.nan

def detect_pinbar(ticker, ohlc_data, rsi_period, atr_period, rsi_bullish_threshold, rsi_bearish_threshold):
    """
    Detect pinbar patterns for a single ticker
    
    Returns: dict with signal info or None if no valid signal
    """
    try:
        # Get the latest candle data
        if ticker not in ohlc_data['Open'].columns:
            return None
            
        # Get last valid values for each OHLC component
        open_val = ohlc_data['Open'][ticker].dropna().iloc[-1] if not ohlc_data['Open'][ticker].dropna().empty else np.nan
        high_val = ohlc_data['High'][ticker].dropna().iloc[-1] if not ohlc_data['High'][ticker].dropna().empty else np.nan
        low_val = ohlc_data['Low'][ticker].dropna().iloc[-1] if not ohlc_data['Low'][ticker].dropna().empty else np.nan
        close_val = ohlc_data['Close'][ticker].dropna().iloc[-1] if not ohlc_data['Close'][ticker].dropna().empty else np.nan
        
        # Check if we have valid OHLC data
        if any(np.isnan([open_val, high_val, low_val, close_val])):
            return None
        
        # Calculate candle components
        body_top = max(open_val, close_val)
        body_bottom = min(open_val, close_val)
        upper_wick = high_val - body_top
        lower_wick = body_bottom - low_val
        candle_range = high_val - low_val
        
        # Skip if candle range is zero (no movement)
        if candle_range == 0:
            return None
        
        # Calculate RSI and ATR
        close_series = ohlc_data['Close'][ticker].dropna()
        high_series = ohlc_data['High'][ticker].dropna() 
        low_series = ohlc_data['Low'][ticker].dropna()
        
        rsi = calculate_rsi(close_series, rsi_period)
        atr = calculate_atr(high_series, low_series, close_series, atr_period)
        
        # Check if we have valid RSI and ATR
        if np.isnan(rsi) or np.isnan(atr) or atr == 0:
            return None
        
        # Check if candle range meets ATR requirement (>= 1 ATR)
        if candle_range < atr:
            return None
        
        # Bullish Pinbar Detection
        if (lower_wick >= 0.5 * candle_range and  # Lower wick at least 50% of candle
            lower_wick >= 2 * upper_wick and      # Lower wick at least 2x upper wick
            rsi <= rsi_bullish_threshold):        # RSI oversold
            
            return {
                'ticker': ticker,
                'signal': 'Bullish',
                'upper_wick': round(upper_wick, 2),
                'lower_wick': round(lower_wick, 2),
                'rsi': round(rsi, 2),
                'atr': round(atr, 2),
                'candle_range': round(candle_range, 2)
            }
        
        # Bearish Pinbar Detection  
        elif (upper_wick >= 0.5 * candle_range and  # Upper wick at least 50% of candle
              upper_wick >= 2 * lower_wick and      # Upper wick at least 2x lower wick
              rsi >= rsi_bearish_threshold):        # RSI overbought
            
            return {
                'ticker': ticker,
                'signal': 'Bearish', 
                'upper_wick': round(upper_wick, 2),
                'lower_wick': round(lower_wick, 2),
                'rsi': round(rsi, 2),
                'atr': round(atr, 2),
                'candle_range': round(candle_range, 2)
            }
        
        return None
        
    except Exception as e:
        st.sidebar.write(f"Error processing {ticker}: {str(e)}")
        return None

# ---------- 4.  USER INTERFACE ----------
st.title("📍 Pinbar Pattern Detection Tool")

# Sidebar controls
st.sidebar.title("Settings")
universe = st.sidebar.radio("Choose Universe", list(UNIVERSE_MAP.keys()), index=0)

# Technical indicator periods
rsi_period = st.sidebar.number_input("RSI Period", min_value=3, max_value=50, value=9)
atr_period = st.sidebar.number_input("ATR Period", min_value=3, max_value=50, value=9)

# RSI thresholds
rsi_bullish_threshold = st.sidebar.number_input("RSI Bullish Threshold (≤)", min_value=10, max_value=50, value=30)
rsi_bearish_threshold = st.sidebar.number_input("RSI Bearish Threshold (≥)", min_value=50, max_value=90, value=70)

# Add refresh button
if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

# Fetch data
try:
    weekly_data, daily_data = fetch_data(universe)
    tickers = list(set(weekly_data['Close'].columns) & set(daily_data['Close'].columns))
    
    if not tickers:
        st.error("No valid tickers found in the selected universe.")
        st.stop()
        
except Exception as e:
    st.error(f"Error fetching data: {str(e)}")
    st.stop()

# ---------- 5.  PATTERN DETECTION ----------
st.subheader(f"🔍 Pinbar Detection Results - {universe} Universe")

# Process each ticker for both timeframes
weekly_results = []
daily_results = []

progress_bar = st.progress(0)
total_tickers = len(tickers)

for i, ticker in enumerate(tickers):
    progress_bar.progress((i + 1) / total_tickers)
    
    # Weekly pinbar detection
    weekly_signal = detect_pinbar(
        ticker, weekly_data, rsi_period, atr_period, 
        rsi_bullish_threshold, rsi_bearish_threshold
    )
    if weekly_signal:
        weekly_results.append(weekly_signal)
    
    # Daily pinbar detection  
    daily_signal = detect_pinbar(
        ticker, daily_data, rsi_period, atr_period,
        rsi_bullish_threshold, rsi_bearish_threshold
    )
    if daily_signal:
        daily_results.append(daily_signal)

progress_bar.empty()

# ---------- 6.  DISPLAY RESULTS ----------
# Create two columns for side-by-side display
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Weekly Pinbar Signals")
    if weekly_results:
        weekly_df = pd.DataFrame(weekly_results)
        # Sort by signal type (Bullish first, then Bearish)
        weekly_df = weekly_df.sort_values(['signal', 'ticker'])
        
        # Style the dataframe
        def color_signal(val):
            if val == 'Bullish':
                return 'background-color: #90EE90'  # Light green
            elif val == 'Bearish':
                return 'background-color: #FFB6C1'  # Light red
            return ''
        
        styled_weekly = weekly_df.style.applymap(color_signal, subset=['signal'])
        st.dataframe(styled_weekly, use_container_width=True, height=400)
        st.write(f"📈 **{len(weekly_results)} Weekly Pinbar signals found**")
    else:
        st.info("No weekly pinbar signals found with current criteria.")

with col2:
    st.subheader("📊 Daily Pinbar Signals") 
    if daily_results:
        daily_df = pd.DataFrame(daily_results)
        # Sort by signal type (Bullish first, then Bearish)
        daily_df = daily_df.sort_values(['signal', 'ticker'])
        
        styled_daily = daily_df.style.applymap(color_signal, subset=['signal'])
        st.dataframe(styled_daily, use_container_width=True, height=400)
        st.write(f"📈 **{len(daily_results)} Daily Pinbar signals found**")
    else:
        st.info("No daily pinbar signals found with current criteria.")

# ---------- 7.  SUMMARY STATISTICS ----------
st.subheader("📈 Summary Statistics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    weekly_bullish = len([r for r in weekly_results if r['signal'] == 'Bullish'])
    st.metric("Weekly Bullish", weekly_bullish)

with col2:
    weekly_bearish = len([r for r in weekly_results if r['signal'] == 'Bearish']) 
    st.metric("Weekly Bearish", weekly_bearish)

with col3:
    daily_bullish = len([r for r in daily_results if r['signal'] == 'Bullish'])
    st.metric("Daily Bullish", daily_bullish)

with col4:
    daily_bearish = len([r for r in daily_results if r['signal'] == 'Bearish'])
    st.metric("Daily Bearish", daily_bearish)

# ---------- 8.  EXCEL EXPORT ----------
if weekly_results or daily_results:
    st.subheader("💾 Export Results")
    
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        workbook = writer.book
        
        # Add timestamp
        timestamp_format = workbook.add_format({'bold': True, 'font_size': 12})
        current_time = (datetime.now() + timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S")
        
        # Define formats
        bullish_format = workbook.add_format({'bg_color': '#90EE90', 'border': 1})
        bearish_format = workbook.add_format({'bg_color': '#FFB6C1', 'border': 1}) 
        header_format = workbook.add_format({'bold': True, 'bg_color': '#D3D3D3', 'border': 1})
        border_format = workbook.add_format({'border': 1})
        
        # Export weekly data
        if weekly_results:
            weekly_df = pd.DataFrame(weekly_results)
            weekly_df.to_excel(writer, sheet_name="Weekly Pinbars", index=False, startrow=2)
            worksheet = writer.sheets['Weekly Pinbars']
            
            # Add timestamp and formatting
            worksheet.write(0, 0, f"Weekly Pinbar Analysis - {current_time} (GMT+8)", timestamp_format)
            
            # Format headers
            for col in range(len(weekly_df.columns)):
                worksheet.write(2, col, weekly_df.columns[col], header_format)
            
            # Format data rows
            for row_num in range(len(weekly_df)):
                excel_row = row_num + 3
                signal = weekly_df.iloc[row_num]['signal']
                format_to_use = bullish_format if signal == 'Bullish' else bearish_format
                
                for col in range(len(weekly_df.columns)):
                    worksheet.write(excel_row, col, weekly_df.iloc[row_num, col], 
                                   format_to_use if col == 1 else border_format)
            
            # Auto-adjust columns
            worksheet.set_column('A:G', 15)
        
        # Export daily data
        if daily_results:
            daily_df = pd.DataFrame(daily_results)
            daily_df.to_excel(writer, sheet_name="Daily Pinbars", index=False, startrow=2)
            worksheet = writer.sheets['Daily Pinbars']
            
            # Add timestamp and formatting
            worksheet.write(0, 0, f"Daily Pinbar Analysis - {current_time} (GMT+8)", timestamp_format)
            
            # Format headers
            for col in range(len(daily_df.columns)):
                worksheet.write(2, col, daily_df.columns[col], header_format)
            
            # Format data rows  
            for row_num in range(len(daily_df)):
                excel_row = row_num + 3
                signal = daily_df.iloc[row_num]['signal']
                format_to_use = bullish_format if signal == 'Bullish' else bearish_format
                
                for col in range(len(daily_df.columns)):
                    worksheet.write(excel_row, col, daily_df.iloc[row_num, col],
                                   format_to_use if col == 1 else border_format)
            
            # Auto-adjust columns
            worksheet.set_column('A:G', 15)
    
    buffer.seek(0)
    st.download_button(
        "📥 Download Excel Report",
        data=buffer,
        file_name=f"Pinbar_Analysis_{universe}_{datetime.now():%Y%m%d_%H%M%S}.xlsx", 
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# ---------- 9.  CRITERIA EXPLANATION ----------
st.subheader("ℹ️ Pinbar Detection Criteria")

with st.expander("📋 Detection Rules"):
    st.write("""
    **Bullish Pinbar Criteria:**
    - Lower wick ≥ 50% of total candle range (High - Low)
    - Lower wick ≥ 2× upper wick size
    - RSI ≤ threshold (default: 30)
    - Candle range (High - Low) ≥ 1 ATR
    
    **Bearish Pinbar Criteria:**  
    - Upper wick ≥ 50% of total candle range (High - Low)
    - Upper wick ≥ 2× lower wick size
    - RSI ≥ threshold (default: 70)
    - Candle range (High - Low) ≥ 1 ATR
    
    **Technical Indicators:**
    - RSI: Relative Strength Index for momentum
    - ATR: Average True Range for volatility filter
    - Both indicators use configurable periods (default: 9)
    """)

# Footer
st.markdown("---")
st.markdown("🔧 **Pinbar Pattern Detection Tool** | Built with Streamlit & Yahoo Finance")
