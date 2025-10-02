import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import yfinance as yf
import numpy as np

# Page config
st.set_page_config(
    page_title="Portfolio Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for holdings
if 'holdings' not in st.session_state:
    st.session_state.holdings = []
    
    # Auto-load from holdings.csv if it exists
    import os
    if os.path.exists('holdings.csv'):
        try:
            df = pd.read_csv('holdings.csv')
            required_columns = ['symbol', 'buy_price', 'quantity']
            
            if all(col in df.columns for col in required_columns):
                for _, row in df.iterrows():
                    st.session_state.holdings.append({
                        "symbol": str(row['symbol']).strip(),
                        "buy_price": float(row['buy_price']),
                        "quantity": int(row['quantity']),
                        "date_added": datetime.now().strftime("%Y-%m-%d")
                    })
        except Exception as e:
            pass

if 'current_page' not in st.session_state:
    st.session_state.current_page = "Portfolio"

# Custom CSS
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .profit {
        color: #00c853;
    }
    .loss {
        color: #ff1744;
    }
    </style>
""", unsafe_allow_html=True)

def format_indian_currency(amount):
    """Format currency in Indian numbering system"""
    amount = float(amount)
    s = f"{amount:,.0f}"
    # Convert to Indian style (lakhs and crores)
    if amount >= 10000000:  # 1 crore
        return f"₹{amount/10000000:.2f} Cr"
    elif amount >= 100000:  # 1 lakh
        return f"₹{amount/100000:.2f} L"
    else:
        # For smaller amounts, use thousands separator
        s = str(int(amount))
        if len(s) <= 3:
            return f"₹{s}"
        last3 = s[-3:]
        rest = s[:-3]
        formatted = ""
        while rest:
            if len(rest) <= 2:
                formatted = rest + "," + formatted
                break
            else:
                formatted = rest[-2:] + "," + formatted
                rest = rest[:-2]
        return f"₹{formatted}{last3}"

# Sidebar Navigation
st.sidebar.title("📊 Portfolio Dashboard")
page = st.sidebar.radio(
    "Navigation",
    ["Portfolio", "Technical Analysis", "News & Announcements", "Market Sentiment"]
)

# Real API functions using yfinance
def validate_ticker(symbol):
    """Validate if ticker exists"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        if 'regularMarketPrice' in info or 'currentPrice' in info or 'previousClose' in info:
            return True
        return False
    except Exception as e:
        return False

@st.cache_data(ttl=60)
def get_current_price(symbol):
    """Get real current price from Yahoo Finance"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        price = (info.get('regularMarketPrice') or 
                info.get('currentPrice') or 
                info.get('previousClose'))
        
        if price:
            return float(price)
        
        hist = ticker.history(period='1d')
        if not hist.empty:
            return float(hist['Close'].iloc[-1])
        
        return None
    except Exception as e:
        return None

def get_pe_ratio(symbol):
    """Get real P/E ratio"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        pe = info.get('trailingPE') or info.get('forwardPE')
        return round(float(pe), 2) if pe else None
    except:
        return None

def get_forward_pe(symbol):
    """Get forward P/E ratio"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        fpe = info.get('forwardPE')
        return round(float(fpe), 2) if fpe else None
    except:
        return None

def get_market_cap_category(symbol):
    """Categorize stocks by market cap"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        market_cap = info.get('marketCap', 0)
        
        if market_cap >= 200000000000:
            return "Large Cap"
        elif market_cap >= 50000000000:
            return "Mid Cap"
        else:
            return "Small Cap"
    except:
        return "Unknown"

def get_next_earnings_date(symbol):
    """Get next earnings date"""
    try:
        ticker = yf.Ticker(symbol)
        calendar = ticker.calendar
        if calendar is not None and 'Earnings Date' in calendar:
            earnings_date = pd.to_datetime(calendar['Earnings Date'][0])
            return earnings_date
    except:
        pass
    return datetime.now() + timedelta(days=30)

def get_200_ema(symbol):
    """Calculate 200-day EMA"""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period='1y')
        if len(hist) >= 200:
            ema_200 = hist['Close'].ewm(span=200, adjust=False).mean().iloc[-1]
            return round(float(ema_200), 2)
    except:
        pass
    return None

# ==================== PORTFOLIO PAGE ====================
if page == "Portfolio":
    st.title("📈 Portfolio Management")
    
    # CSV Upload Section
    st.subheader("📁 Upload Holdings from CSV")
    
    with st.expander("Upload CSV File", expanded=False):
        st.markdown("""
        **CSV Format Required:**
        - Columns: `symbol`, `buy_price`, `quantity`
        - Example:
        ```
        symbol,buy_price,quantity
        RELIANCE.NS,2400.50,100
        TCS.NS,3500.00,50
        INFY.NS,1450.75,75
        ```
        """)
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                required_columns = ['symbol', 'buy_price', 'quantity']
                if not all(col in df.columns for col in required_columns):
                    st.error(f"❌ CSV must contain columns: {', '.join(required_columns)}")
                else:
                    st.dataframe(df, use_container_width=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("✅ Import All Holdings", type="primary"):
                            success_count = 0
                            error_count = 0
                            errors = []
                            
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            for idx, row in df.iterrows():
                                symbol = str(row['symbol']).strip()
                                buy_price = float(row['buy_price'])
                                quantity = int(row['quantity'])
                                
                                status_text.text(f"Validating {symbol}... ({idx+1}/{len(df)})")
                                progress_bar.progress((idx + 1) / len(df))
                                
                                existing_symbols = [h['symbol'] for h in st.session_state.holdings]
                                if symbol in existing_symbols:
                                    errors.append(f"{symbol} - Already exists")
                                    error_count += 1
                                    continue
                                
                                is_valid = validate_ticker(symbol)
                                
                                if is_valid:
                                    st.session_state.holdings.append({
                                        "symbol": symbol,
                                        "buy_price": buy_price,
                                        "quantity": quantity,
                                        "date_added": datetime.now().strftime("%Y-%m-%d")
                                    })
                                    success_count += 1
                                else:
                                    errors.append(f"{symbol} - Invalid ticker")
                                    error_count += 1
                            
                            progress_bar.empty()
                            status_text.empty()
                            
                            if success_count > 0:
                                st.success(f"✅ Successfully imported {success_count} holdings!")
                            if error_count > 0:
                                st.warning(f"⚠️ {error_count} holdings failed:")
                                for error in errors:
                                    st.text(f"  - {error}")
                            
                            if success_count > 0:
                                st.rerun()
                    
                    with col2:
                        sample_csv = """symbol,buy_price,quantity
RELIANCE.NS,2400.50,100
TCS.NS,3500.00,50
INFY.NS,1450.75,75
HDFCBANK.NS,1600.00,60"""
                        st.download_button(
                            label="📥 Download Sample CSV",
                            data=sample_csv,
                            file_name="portfolio_sample.csv",
                            mime="text/csv"
                        )
                        
            except Exception as e:
                st.error(f"❌ Error reading CSV: {str(e)}")
    
    # Add Holdings Section
    st.subheader("Add New Holding Manually")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        symbol = st.text_input("Stock Symbol (e.g., RELIANCE.NS)", key="symbol_input")
    with col2:
        buy_price = st.number_input("Buy Price (₹)", min_value=0.0, step=0.01, key="buy_price")
    with col3:
        quantity = st.number_input("Quantity", min_value=1, step=1, key="quantity")
    with col4:
        st.write("")
        st.write("")
        if st.button("Add Holding", type="primary"):
            if symbol and buy_price > 0 and quantity > 0:
                with st.spinner(f"Validating {symbol}..."):
                    is_valid = validate_ticker(symbol)
                    
                    if is_valid:
                        existing_symbols = [h['symbol'] for h in st.session_state.holdings]
                        if symbol in existing_symbols:
                            st.error(f"❌ {symbol} already exists in your portfolio!")
                        else:
                            st.session_state.holdings.append({
                                "symbol": symbol,
                                "buy_price": buy_price,
                                "quantity": quantity,
                                "date_added": datetime.now().strftime("%Y-%m-%d")
                            })
                            st.success(f"✅ Added {quantity} shares of {symbol}")
                            st.rerun()
                    else:
                        st.error(f"❌ Invalid ticker symbol: {symbol}. Please check and try again.")
    
    # Portfolio Overview
    if st.session_state.holdings:
        st.subheader("📊 Portfolio Overview")
        
        portfolio_data = []
        total_investment = 0
        total_current_value = 0
        
        with st.spinner("Fetching latest prices..."):
            for holding in st.session_state.holdings:
                current_price = get_current_price(holding['symbol'])
                
                if current_price is None:
                    st.warning(f"⚠️ Could not fetch price for {holding['symbol']}")
                    continue
                
                investment = holding['buy_price'] * holding['quantity']
                current_value = current_price * holding['quantity']
                pnl = current_value - investment
                pnl_percent = (pnl / investment) * 100
                
                total_investment += investment
                total_current_value += current_value
                
                portfolio_data.append({
                    "Symbol": holding['symbol'],
                    "Quantity": holding['quantity'],
                    "Buy Price": f"₹{holding['buy_price']:.2f}",
                    "Current Price": f"₹{current_price:.0f}",
                    "Investment": format_indian_currency(investment),
                    "Current Value": format_indian_currency(current_value),
                    "P&L": format_indian_currency(pnl),
                    "P&L %": f"{pnl_percent:.2f}%",
                    "P/E": f"{get_pe_ratio(holding['symbol']):.2f}" if get_pe_ratio(holding['symbol']) else "N/A",
                    "Forward P/E": f"{get_forward_pe(holding['symbol']):.2f}" if get_forward_pe(holding['symbol']) else "N/A",
                })
        
        total_pnl = total_current_value - total_investment
        total_pnl_percent = (total_pnl / total_investment * 100) if total_investment > 0 else 0
        
        # Calculate weighted average P/E
        total_pe_weighted = 0
        total_weight = 0
        for holding in st.session_state.holdings:
            current_price = get_current_price(holding['symbol'])
            pe = get_pe_ratio(holding['symbol'])
            if current_price and pe:
                value = current_price * holding['quantity']
                total_pe_weighted += pe * value
                total_weight += value
        
        weighted_avg_pe = (total_pe_weighted / total_weight) if total_weight > 0 else 0
        
        col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 1, 1.5])
        with col1:
            st.metric("Total Investment", format_indian_currency(total_investment))
        with col2:
            st.metric("Current Value", format_indian_currency(total_current_value))
        with col3:
            st.metric("Total P&L", format_indian_currency(total_pnl), f"{total_pnl_percent:.2f}%")
        with col4:
            st.metric("Holdings", len(st.session_state.holdings))
        with col5:
            st.metric("Weighted Avg P/E", f"{weighted_avg_pe:.2f}" if weighted_avg_pe > 0 else "N/A")
        
        if portfolio_data:
            # Create DataFrame and add color styling to P&L columns
            df = pd.DataFrame(portfolio_data)
            
            def color_pnl(val):
                try:
                    # Remove currency symbols and commas, then convert to float
                    num = float(val.replace('₹', '').replace(',', '').replace('%', ''))
                    if num >= 0:
                        return 'background-color: rgba(0, 200, 83, 0.3); color: #00c853; font-weight: bold'
                    else:
                        return 'background-color: rgba(255, 23, 68, 0.3); color: #ff1744; font-weight: bold'
                except:
                    return ''
            
            styled_df = df.style.applymap(color_pnl, subset=['P&L', 'P&L %'])
            
            st.dataframe(
                styled_df,
                use_container_width=True,
                hide_index=True
            )
        
        st.subheader("📊 Asset Allocation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            cap_allocation = {}
            for holding in st.session_state.holdings:
                cap_type = get_market_cap_category(holding['symbol'])
                current_price = get_current_price(holding['symbol'])
                if current_price:
                    value = current_price * holding['quantity']
                    cap_allocation[cap_type] = cap_allocation.get(cap_type, 0) + value
            
            if cap_allocation:
                fig_cap = px.pie(
                    values=list(cap_allocation.values()),
                    names=list(cap_allocation.keys()),
                    title="Market Cap Allocation",
                    hole=0.4
                )
                st.plotly_chart(fig_cap, use_container_width=True)
        
        with col2:
            holdings_value = []
            for holding in st.session_state.holdings:
                current_price = get_current_price(holding['symbol'])
                if current_price:
                    value = current_price * holding['quantity']
                    holdings_value.append({
                        "Symbol": holding['symbol'],
                        "Value": value
                    })
            
            if holdings_value:
                fig_holdings = px.bar(
                    pd.DataFrame(holdings_value),
                    x="Symbol",
                    y="Value",
                    title="Holdings by Value (₹)"
                )
                st.plotly_chart(fig_holdings, use_container_width=True)
        
        st.subheader("📅 Upcoming Earnings Calendar")
        
        earnings_data = []
        for holding in st.session_state.holdings:
            earnings_date = get_next_earnings_date(holding['symbol'])
            days_until = (earnings_date - datetime.now()).days
            
            earnings_data.append({
                "Symbol": holding['symbol'],
                "Earnings Date": earnings_date.strftime("%Y-%m-%d"),
                "Days Until": days_until
            })
        
        earnings_df = pd.DataFrame(earnings_data).sort_values("Days Until")
        st.dataframe(earnings_df, use_container_width=True, hide_index=True)
        
        st.subheader("⚙️ Manage Portfolio")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔄 Refresh Prices", type="secondary"):
                st.cache_data.clear()
                st.rerun()
        with col2:
            if st.button("💾 Export to CSV", type="secondary"):
                export_data = []
                for holding in st.session_state.holdings:
                    export_data.append({
                        "symbol": holding['symbol'],
                        "buy_price": holding['buy_price'],
                        "quantity": holding['quantity']
                    })
                export_df = pd.DataFrame(export_data)
                csv = export_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download holdings.csv",
                    data=csv,
                    file_name="holdings.csv",
                    mime="text/csv"
                )
        with col3:
            if st.button("🗑️ Clear All Holdings", type="secondary"):
                st.session_state.holdings = []
                st.rerun()
    
    else:
        st.info("👆 Add your first holding to get started!")
        st.markdown("""
        **Example Indian Stock Symbols:**
        - RELIANCE.NS (Reliance Industries)
        - TCS.NS (Tata Consultancy Services)
        - INFY.NS (Infosys)
        - HDFCBANK.NS (HDFC Bank)
        - ITC.NS (ITC Limited)
        """)

# ==================== TECHNICAL ANALYSIS PAGE ====================
elif page == "Technical Analysis":
    st.title("📉 Technical Analysis")
    
    if not st.session_state.holdings:
        st.warning("Please add holdings first in the Portfolio page")
    else:
        symbols = [h['symbol'] for h in st.session_state.holdings]
        selected_stock = st.selectbox("Select Stock", symbols)
        
        with st.spinner(f"Loading data for {selected_stock}..."):
            current_price = get_current_price(selected_stock)
            ema_200 = get_200_ema(selected_stock)
            
            if current_price and ema_200:
                distance = current_price - ema_200
                distance_percent = (distance / ema_200) * 100
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Current Price", f"₹{current_price:.2f}")
                with col2:
                    st.metric("200 EMA", f"₹{ema_200:.2f}")
                with col3:
                    st.metric("Distance (₹)", f"₹{distance:.2f}")
                with col4:
                    st.metric("Distance (%)", f"{distance_percent:.2f}%", 
                             f"{'Above' if distance >= 0 else 'Below'} EMA")
                
                st.subheader("📈 Price Chart with 200 EMA")
                
                try:
                    ticker = yf.Ticker(selected_stock)
                    hist = ticker.history(period='1y')
                    
                    if not hist.empty:
                        hist['EMA_200'] = hist['Close'].ewm(span=200, adjust=False).mean()
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=hist.index, 
                            y=hist['Close'], 
                            mode='lines', 
                            name='Price', 
                            line=dict(color='blue')
                        ))
                        fig.add_trace(go.Scatter(
                            x=hist.index, 
                            y=hist['EMA_200'], 
                            mode='lines', 
                            name='200 EMA', 
                            line=dict(color='orange', dash='dash')
                        ))
                        
                        fig.update_layout(
                            title=f"{selected_stock} - Price vs 200 EMA",
                            xaxis_title="Date",
                            yaxis_title="Price (₹)",
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("No historical data available")
                except Exception as e:
                    st.error(f"Error loading chart: {str(e)}")
            else:
                st.error("Could not fetch data for this stock")
        
        # EMA Analysis Table for All Holdings
        st.markdown("---")
        st.subheader("📊 All Holdings - EMA Analysis")
        
        with st.spinner("Calculating EMA for all holdings..."):
            ema_analysis = []
            for holding in st.session_state.holdings:
                symbol = holding['symbol']
                curr_price = get_current_price(symbol)
                ema_val = get_200_ema(symbol)
                
                if curr_price and ema_val:
                    dist_rupees = curr_price - ema_val
                    dist_percent = (dist_rupees / ema_val) * 100
                    pos = "Above EMA" if dist_percent >= 0 else "Below EMA"
                    
                    ema_analysis.append({
                        "Stock": symbol,
                        "Current Price": f"₹{curr_price:.2f}",
                        "200 EMA": f"₹{ema_val:.2f}",
                        "Distance (₹)": f"₹{dist_rupees:.2f}",
                        "Distance (%)": f"{dist_percent:.2f}%",
                        "Position": pos
                    })
        
        if ema_analysis:
            ema_df = pd.DataFrame(ema_analysis)
            
            # Sort by Distance (%) in descending order
            # Extract numeric values for sorting
            ema_df['Distance_Sort'] = ema_df['Distance (%)'].str.replace('%', '').astype(float)
            ema_df = ema_df.sort_values('Distance_Sort', ascending=False)
            ema_df = ema_df.drop('Distance_Sort', axis=1)
            
            # Apply color styling to Distance (%)
            def color_distance_percent(val):
                try:
                    num = float(val.replace('%', ''))
                    if num >= 0:
                        return 'background-color: rgba(0, 200, 83, 0.3); color: #00c853; font-weight: bold'
                    else:
                        return 'background-color: rgba(255, 23, 68, 0.3); color: #ff1744; font-weight: bold'
                except:
                    return ''
            
            styled_df = ema_df.style.applymap(color_distance_percent, subset=['Distance (%)'])
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
            
            above_ema = len([x for x in ema_analysis if "Above" in x['Position']])
            below_ema = len([x for x in ema_analysis if "Below" in x['Position']])
            
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Above 200 EMA", f"{above_ema} stocks")
            with col2:
                st.metric("Below 200 EMA", f"{below_ema} stocks")
            with col3:
                bullish_percent = (above_ema / len(ema_analysis) * 100) if ema_analysis else 0
                st.metric("Portfolio Strength", f"{bullish_percent:.1f}%")
        else:
            st.warning("Unable to calculate EMA for holdings")

# ==================== NEWS & ANNOUNCEMENTS PAGE ====================
elif page == "News & Announcements":
    st.title("📰 News & Announcements")
    
    if not st.session_state.holdings:
        st.warning("Please add holdings first in the Portfolio page")
    else:
        symbols = [h['symbol'] for h in st.session_state.holdings]
        
        col1, col2 = st.columns([3, 1])
        with col1:
            selected_filter = st.selectbox("Filter by Stock", ["All Stocks"] + symbols)
        with col2:
            time_filter = st.selectbox("Time Period", ["Today", "This Week", "This Month"])
        
        st.info("For real news integration, you can use NewsAPI or similar services")
        
        st.subheader("📊 Company Information")
        
        for symbol in symbols:
            if selected_filter == "All Stocks" or selected_filter == symbol:
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    
                    with st.expander(f"🏢 {info.get('longName', symbol)}", expanded=False):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            market_cap = info.get('marketCap', 0) / 10000000
                            st.metric("Market Cap", f"₹{market_cap:,.0f} Cr")
                        with col2:
                            st.metric("P/E Ratio", f"{info.get('trailingPE', 'N/A')}")
                        with col3:
                            st.metric("52W High", f"₹{info.get('fiftyTwoWeekHigh', 'N/A')}")
                        
                        st.markdown(f"**Sector:** {info.get('sector', 'N/A')}")
                        st.markdown(f"**Industry:** {info.get('industry', 'N/A')}")
                        st.markdown(f"**Website:** {info.get('website', 'N/A')}")
                except:
                    st.warning(f"Could not fetch info for {symbol}")

# ==================== MARKET SENTIMENT PAGE ====================
elif page == "Market Sentiment":
    st.title("📊 Market Sentiment & Indicators")
    
    try:
        nifty = yf.Ticker("^NSEI")
        nifty_data = nifty.history(period='1d')
        
        if not nifty_data.empty:
            nifty_price = nifty_data['Close'].iloc[-1]
            nifty_change = ((nifty_data['Close'].iloc[-1] - nifty_data['Open'].iloc[-1]) / nifty_data['Open'].iloc[-1]) * 100
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Nifty 50", f"{nifty_price:,.0f}", f"{nifty_change:+.2f}%")
            with col2:
                vix = yf.Ticker("^INDIAVIX")
                vix_data = vix.history(period='1d')
                if not vix_data.empty:
                    st.metric("India VIX", f"{vix_data['Close'].iloc[-1]:.2f}")
            with col3:
                st.metric("Market Status", "Open" if datetime.now().hour < 15 else "Closed")
            with col4:
                st.metric("Your Holdings", len(st.session_state.holdings))
    except:
        st.warning("Could not fetch market data")
    
    st.info("For FII/DII flow data, you would need to integrate with NSE/BSE APIs or paid data providers")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("""
**Quick Tips:**
- Add holdings to track your portfolio
- All prices are fetched in real-time
- Invalid tickers will be rejected
- Click Refresh to update prices
""")
st.sidebar.markdown("**Data Source:** Yahoo Finance")