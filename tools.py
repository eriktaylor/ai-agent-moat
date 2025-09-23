# tools.py

import yfinance as yf
from langchain.tools import tool
import requests

@tool
def get_stock_info(ticker: str) -> str:
    """
    Fetches key financial information for a given stock ticker using the yfinance library.
    Uses a multi-step fallback approach to ensure robustness against API changes.
    Returns a formatted string of the data or an error message if critical data is unavailable.
    """
    try:
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36'
        })
        
        stock = yf.Ticker(ticker, session=session)
        
        # --- NEW ROBUST DATA FETCHING LOGIC ---
        data = {}

        # 1. Use fast_info for the most critical, real-time data
        fi = getattr(stock, "fast_info", {})
        data["Current Price"] = fi.get("lastPrice")
        data["Previous Close"] = fi.get("previousClose")
        data["Open"] = fi.get("open")
        data["Day Low"] = fi.get("dayLow")
        data["Day High"] = fi.get("dayHigh")
        data["Volume"] = fi.get("volume")
        data["Market Cap"] = fi.get("marketCap")

        # 2. Use history() as a fallback for price if fast_info fails
        if data.get("Current Price") is None:
            try:
                hist = stock.history(period="1d")
                if not hist.empty:
                    data["Current Price"] = float(hist["Close"].iloc[-1])
                    if data.get("Open") is None:
                         data["Open"] = float(hist["Open"].iloc[-1])
            except Exception:
                pass # Ignore if history fails

        # 3. Use get_info() for less critical, supplemental data (can be slow/flaky)
        try:
            info = stock.get_info()
            data["Company Name"] = info.get('longName')
            data["Symbol"] = info.get('symbol')
            data["Trailing P/E"] = info.get('trailingPE')
            data["Forward P/E"] = info.get('forwardPE')
            data["Trailing EPS"] = info.get('trailingEps')
            data["52 Week High"] = info.get('fiftyTwoWeekHigh')
            data["52 Week Low"] = info.get('fiftyTwoWeekLow')
            # Use get_info as a fallback for data that might also be in fast_info
            if data.get("Market Cap") is None: data["Market Cap"] = info.get('marketCap')
            if data.get("Volume") is None: data["Volume"] = info.get('volume')
        except Exception:
            pass # get_info() can fail, but we can proceed without it
        # --- END OF NEW LOGIC ---

        # Validate that we have at least a price; otherwise, the analysis is not useful.
        if data.get("Current Price") is None:
            return f"Error: Could not retrieve current price for ticker '{ticker}'. The symbol may be invalid or delisted."

        # Define the desired order and format the data into a clean, readable string
        key_order = [
            "Company Name", "Symbol", "Current Price", "Previous Close", "Open",
            "Day Low", "Day High", "Market Cap", "Trailing P/E", "Forward P/E",
            "Trailing EPS", "Volume", "52 Week High", "52 Week Low"
        ]
        
        report = []
        for key in key_order:
            value = data.get(key)
            report.append(f"{key}: {value if value is not None else 'N/A'}")

        return "\n".join(report)

    except Exception as e:
        return f"Error: An unexpected exception occurred while fetching data for ticker '{ticker}'. Details: {e}"

@tool
def scrape_website(url: str) -> str:
    """
    A placeholder for a website scraping tool.
    In a real implementation, this would use libraries like BeautifulSoup or Selenium.
    """
    return "Scraping functionality is not implemented in this demo."