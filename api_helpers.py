"""
API Integration Helpers
Replace mock functions with real API calls here
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf

# ============== CONFIGURATION ==============
# Add your API keys here
ALPHA_VANTAGE_KEY = "YOUR_API_KEY"
FMP_KEY = "YOUR_FMP_KEY"
STOCKINSIGHTS_KEY = "YOUR_STOCKINSIGHTS_KEY"

# ============== STOCK PRICE DATA ==============

def get_current_price_yahoo(symbol):
    """Get current stock price using Yahoo Finance (FREE)"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period='1d')
        if not data.empty:
            return data['Close'].iloc[-1]
        return None
    except Exception as e:
        print(f"Error fetching price for {symbol}: {e}")
        return None

def get_stock_info_yahoo(symbol):
    """Get comprehensive stock info from Yahoo Finance"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        return {
            'current_price': info.get('currentPrice', 0),
            'pe_ratio': info.get('trailingPE', 0),
            'forward_pe': info.get('forwardPE', 0),
            'market_cap': info.get('marketCap', 0),
            'day_high': info.get('dayHigh', 0),
            'day_low': info.get('dayLow', 0),
            'volume': info.get('volume', 0),
            '52_week_high': info.get('fiftyTwoWeekHigh', 0),
            '52_week_low': info.get('fiftyTwoWeekLow', 0),
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown')
        }
    except Exception as e:
        print(f"Error fetching info for {symbol}: {e}")
        return None

# ============== MARKET CAP CATEGORIZATION ==============

def get_market_cap_category_from_api(symbol):
    """Categorize stock based on actual market cap"""
    try:
        info = get_stock_info_yahoo(symbol)
        if info:
            market_cap = info['market_cap']
            
            # Indian market categorization (in INR)
            if market_cap >= 1_00_000_00_00_000:  # ≥ 1 Lakh Crore
                return "Large Cap"
            elif market_cap >= 25_000_00_00_000:  # 25,000 Cr to 1 Lakh Cr
                return "Mid Cap"
            else:
                return "Small Cap"
        return "Unknown"
    except Exception as e:
        print(f"Error categorizing {symbol}: {e}")
        return "Unknown"

# ============== EARNINGS DATA ==============

def get_next_earnings_date_yahoo(symbol):
    """Get next earnings date from Yahoo Finance"""
    try:
        ticker = yf.Ticker(symbol)
        calendar = ticker.calendar
        
        if calendar is not None and 'Earnings Date' in calendar:
            earnings_date = calendar['Earnings Date'][0]
            if isinstance(earnings_date, pd.Timestamp):
                return earnings_date.to_pydatetime()
        
        # Fallback: estimate based on quarterly pattern
        return datetime.now() + timedelta(days=30)
    except Exception as e:
        print(f"Error fetching earnings date for {symbol}: {e}")
        return datetime.now() + timedelta(days=30)

def get_earnings_summary_fmp(symbol):
    """Get earnings summary from Financial Modeling Prep API"""
    try:
        # Remove .NS or .BO suffix for FMP
        clean_symbol = symbol.replace('.NS', '').replace('.BO', '')
        
        url = f"https://financialmodelingprep.com/api/v3/earnings-surprises/{clean_symbol}"
        params = {'apikey': FMP_KEY}
        
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            if data:
                latest = data[0]
                return {
                    'date': latest.get('date'),
                    'eps_actual': latest.get('actualEarningResult'),
                    'eps_estimate': latest.get('estimatedEarning'),
                    'revenue': latest.get('revenue'),
                    'surprise_percent': latest.get('surprisePercent')
                }
        return None
    except Exception as e:
        print(f"Error fetching earnings for {symbol}: {e}")
        return None

# ============== TECHNICAL ANALYSIS ==============

def calculate_200_ema(symbol, period='2y'):
    """Calculate 200-day EMA"""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period)
        
        if len(hist) >= 200:
            ema_200 = hist['Close'].ewm(span=200, adjust=False).mean().iloc[-1]
            return ema_200
        return None
    except Exception as e:
        print(f"Error calculating 200 EMA for {symbol}: {e}")
        return None

def get_technical_indicators(symbol):
    """Calculate multiple technical indicators"""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period='1y')
        
        if len(hist) < 50:
            return None
        
        # Calculate indicators
        close = hist['Close']
        
        # Moving Averages
        sma_50 = close.rolling(window=50).mean().iloc[-1]
        sma_200 = close.rolling(window=200).mean().iloc[-1] if len(hist) >= 200 else None
        ema_50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
        ema_200 = close.ewm(span=200, adjust=False).mean().iloc[-1] if len(hist) >= 200 else None
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return {
            'sma_50': sma_50,
            'sma_200': sma_200,
            'ema_50': ema_50,
            'ema_200': ema_200,
            'rsi': rsi.iloc[-1],
            'current_price': close.iloc[-1]
        }
    except Exception as e:
        print(f"Error calculating indicators for {symbol}: {e}")
        return None

def get_price_history(symbol, period='1y'):
    """Get historical price data for charts"""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period)
        return hist
    except Exception as e:
        print(f"Error fetching history for {symbol}: {e}")
        return None

# ============== MARKET SENTIMENT DATA ==============

def get_nse_indices():
    """Get Nifty 50 and other indices data"""
    try:
        nifty = yf.Ticker("^NSEI")
        info = nifty.info
        
        return {
            'nifty_price': info.get('regularMarketPrice', 0),
            'nifty_change': info.get('regularMarketChange', 0),
            'nifty_change_percent': info.get('regularMarketChangePercent', 0),
            'nifty_pe': info.get('trailingPE', 0)
        }
    except Exception as e:
        print(f"Error fetching NSE indices: {e}")
        return None

def get_india_vix():
    """Get India VIX data"""
    try:
        vix = yf.Ticker("^INDIAVIX")
        hist = vix.history(period='1d')
        if not hist.empty:
            return hist['Close'].iloc[-1]
        return None
    except Exception as e:
        print(f"Error fetching India VIX: {e}")
        return None

def get_fii_dii_data():
    """
    Get FII/DII flow data
    Note: This requires web scraping NSE India website
    or paid API access. Mock implementation provided.
    """
    # This is a placeholder - implement web scraping or use paid API
    # NSE publishes this data daily at: 
    # https://www.nseindia.com/reports/fii-dii-data
    
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    
    # Mock data - replace with actual scraping/API
    import numpy as np
    fii_data = np.random.randint(-2000, 3000, 30)
    dii_data = np.random.randint(-1000, 2500, 30)
    
    return pd.DataFrame({
        'date': dates,
        'fii_flow': fii_data,
        'dii_flow': dii_data,
        'net_flow': fii_data + dii_data
    })

def scrape_nse_fii_dii():
    """
    Scrape FII/DII data from NSE India
    IMPORTANT: Add proper headers and cookies for NSE
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        
        # NSE API endpoint (may change, verify current endpoint)
        url = "https://www.nseindia.com/api/fiidiiTradeReact"
        
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers)  # Get cookies
        
        response = session.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            return data
        return None
    except Exception as e:
        print(f"Error scraping FII/DII data: {e}")
        return None

# ============== NEWS & ANNOUNCEMENTS ==============

def get_stock_news_yahoo(symbol, num_articles=10):
    """Get news articles for a stock from Yahoo Finance"""
    try:
        ticker = yf.Ticker(symbol)
        news = ticker.news
        
        articles = []
        for item in news[:num_articles]:
            articles.append({
                'title': item.get('title'),
                'publisher': item.get('publisher'),
                'link': item.get('link'),
                'publish_time': datetime.fromtimestamp(item.get('providerPublishTime'))
            })
        return articles
    except Exception as e:
        print(f"Error fetching news for {symbol}: {e}")
        return []

def get_earnings_transcript_summary(symbol):
    """
    Get earnings transcript summary
    Requires StockInsights API or similar service
    """
    try:
        # Example using StockInsights API
        # Replace with actual API endpoint
        url = f"https://api.stockinsights.ai/v1/earnings/{symbol}/latest"
        headers = {'Authorization': f'Bearer {STOCKINSIGHTS_KEY}'}
        
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            return {
                'summary': data.get('summary'),
                'highlights': data.get('key_highlights'),
                'revenue': data.get('revenue'),
                'eps': data.get('eps'),
                'guidance': data.get('guidance'),
                'sentiment': data.get('sentiment'),
                'date': data.get('earnings_date')
            }
        return None
    except Exception as e:
        print(f"Error fetching earnings transcript: {e}")
        return None

def get_company_announcements_nse(symbol):
    """Get corporate announcements from NSE"""
    try:
        # Remove .NS suffix
        clean_symbol = symbol.replace('.NS', '')
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        url = f"https://www.nseindia.com/api/corporate-announcements?index=equities&symbol={clean_symbol}"
        
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        print(f"Error fetching announcements: {e}")
        return None

# ============== HISTORICAL P/E DATA ==============

def get_nifty_historical_pe():
    """
    Get historical Nifty P/E data
    Note: This data is available from NSE reports
    """
    try:
        # This requires downloading NSE historical P/E reports
        # or using a paid data provider
        
        # Mock data for demonstration
        dates = pd.date_range(end=datetime.now(), periods=365*5, freq='D')
        pe_values = 21.5 + np.random.randn(len(dates)) * 2
        
        return pd.DataFrame({
            'date': dates,
            'pe_ratio': pe_values
        })
    except Exception as e:
        print(f"Error fetching historical P/E: {e}")
        return None

# ============== DATA PERSISTENCE ==============

def save_holdings_to_file(holdings, filename='portfolio_holdings.json'):
    """Save holdings to JSON file"""
    import json
    try:
        with open(filename, 'w') as f:
            json.dump(holdings, f, indent=2, default=str)
        return True
    except Exception as e:
        print(f"Error saving holdings: {e}")
        return False

def load_holdings_from_file(filename='portfolio_holdings.json'):
    """Load holdings from JSON file"""
    import json
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []
    except Exception as e:
        print(f"Error loading holdings: {e}")
        return []

# ============== BATCH OPERATIONS ==============

def update_all_stock_prices(holdings):
    """Update prices for all holdings in batch"""
    updated_holdings = []
    
    for holding in holdings:
        price = get_current_price_yahoo(holding['symbol'])
        if price:
            holding['current_price'] = price
            holding['last_updated'] = datetime.now().isoformat()
        updated_holdings.append(holding)
    
    return updated_holdings

def get_portfolio_summary(holdings):
    """Calculate comprehensive portfolio summary"""
    total_investment = 0
    total_current_value = 0
    holdings_data = []
    
    for holding in holdings:
        info = get_stock_info_yahoo(holding['symbol'])
        if info:
            current_price = info['current_price']
            investment = holding['buy_price'] * holding['quantity']
            current_value = current_price * holding['quantity']
            pnl = current_value - investment
            pnl_percent = (pnl / investment * 100) if investment > 0 else 0
            
            total_investment += investment
            total_current_value += current_value
            
            holdings_data.append({
                'symbol': holding['symbol'],
                'quantity': holding['quantity'],
                'buy_price': holding['buy_price'],
                'current_price': current_price,
                'investment': investment,
                'current_value': current_value,
                'pnl': pnl,
                'pnl_percent': pnl_percent,
                'pe_ratio': info['pe_ratio'],
                'forward_pe': info['forward_pe'],
                'sector': info['sector']
            })
    
    return {
        'holdings': holdings_data,
        'total_investment': total_investment,
        'total_current_value': total_current_value,
        'total_pnl': total_current_value - total_investment,
        'total_pnl_percent': ((total_current_value - total_investment) / total_investment * 100) if total_investment > 0 else 0
    }

# ============== USAGE EXAMPLES ==============

if __name__ == "__main__":
    # Example usage
    print("Testing API functions...")
    
    # Test stock price
    symbol = "RELIANCE.NS"
    price = get_current_price_yahoo(symbol)
    print(f"Current price of {symbol}: ₹{price}")
    
    # Test stock info
    info = get_stock_info_yahoo(symbol)
    print(f"P/E Ratio: {info['pe_ratio']}")
    
    # Test 200 EMA
    ema = calculate_200_ema(symbol)
    print(f"200 EMA: ₹{ema}")
    
    # Test market cap category
    category = get_market_cap_category_from_api(symbol)
    print(f"Market Cap Category: {category}")
    
    # Test technical indicators
    indicators = get_technical_indicators(symbol)
    if indicators:
        print(f"RSI: {indicators['rsi']:.2f}")
    
    # Test news
    news = get_stock_news_yahoo(symbol, num_articles=3)
    print(f"Found {len(news)} news articles")
    
    print("\nAll tests completed!")