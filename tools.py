import yfinance as yf
from langchain.tools import tool
import requests  # <-- ADD THIS IMPORT

@tool
def get_stock_info(ticker: str) -> str:
    """
    Fetches key financial information for a given stock ticker using the yfinance library.
    Returns a formatted string of the data or an error message if the ticker is invalid or data is unavailable.
    """
    try:
        # --- THIS IS THE FIX ---
        # Create a session with a browser-like User-Agent to avoid being blocked.
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36'
        })
        
        # Pass the session to the Ticker object.
        stock = yf.Ticker(ticker, session=session)
        # --- END OF FIX ---

        info = stock.info

        # Check if essential data is present. 'regularMarketPrice' is a good indicator of a valid ticker.
        if not info or 'regularMarketPrice' not in info or info['regularMarketPrice'] is None:
            return f"Error: Could not retrieve valid data for ticker '{ticker}'. It may be an invalid symbol."

        # Define the data points we want to extract
        key_metrics = {
            "Company Name": info.get('longName'),
            "Symbol": info.get('symbol'),
            "Current Price": info.get('regularMarketPrice'),
            "Previous Close": info.get('previousClose'),
            "Open": info.get('open'),
            "Day Low": info.get('dayLow'),
            "Day High": info.get('dayHigh'),
            "Market Cap": info.get('marketCap'),
            "Trailing P/E": info.get('trailingPE'),
            "Forward P/E": info.get('forwardPE'),
            "Trailing EPS": info.get('trailingEps'),
            "Volume": info.get('volume'),
            "52 Week High": info.get('fiftyTwoWeekHigh'),
            "52 Week Low": info.get('fiftyTwoWeekLow'),
        }

        # Format the data into a clean, readable string
        report = []
        for key, value in key_metrics.items():
            if value is not None:
                report.append(f"{key}: {value}")
            else:
                report.append(f"{key}: N/A")

        return "\n".join(report)

    except Exception as e:
        # Catch any other exceptions from yfinance or network issues
        return f"Error: An exception occurred while trying to fetch data for ticker '{ticker}'. Details: {e}"

# You can keep other tools like scrape_website here if you have them
@tool
def scrape_website(url: str) -> str:
    """
    A placeholder for a website scraping tool.
    In a real implementation, this would use libraries like BeautifulSoup or Selenium.
    """
    return "Scraping functionality is not implemented in this demo."
