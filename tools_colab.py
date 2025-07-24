import requests
from bs4 import BeautifulSoup
import yfinance as yf
from langchain.agents import tool
import fitz  # PyMuPDF
import io

# Upgraded scraper tool with PDF handling
@tool
def scrape_website(url: str) -> str:
    """Scrapes text from HTML websites and extracts text from PDF files."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status() # Raise an exception for bad status codes

        content_type = response.headers.get('content-type', '')

        if 'application/pdf' in content_type or url.lower().endswith('.pdf'):
            # It's a PDF, use PyMuPDF to extract text
            with fitz.open(stream=io.BytesIO(response.content), filetype='pdf') as doc:
                text = "".join(page.get_text() for page in doc)
            return text[:8000] # Return a larger chunk for detailed PDFs
        elif 'text/html' in content_type:
            # It's HTML, use BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            text = ' '.join(p.get_text() for p in soup.find_all('p'))
            if len(text) < 200:
                return f"Error: HTML content from {url} is too short."
            return text[:4000]
        else:
            return f"Error: Unsupported content type '{content_type}' at {url}"

    except requests.RequestException as e:
        return f"Error: Could not access the URL. {e}"
    except Exception as e:
        return f"Error: An unexpected error occurred while processing {url}. {e}"

# Tool now fetches EPS
@tool
def get_stock_info(ticker: str) -> str:
    """Fetches key financial information for a given stock ticker using Yahoo Finance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        market_cap = info.get('marketCap', 'N/A')
        trailing_pe = info.get('trailingPE', 'N/A')
        forward_pe = info.get('forwardPE', 'N/A')
        eps = info.get('trailingEps', 'N/A')
        long_business_summary = info.get('longBusinessSummary', 'N/A')
        return f"### KEY FINANCIAL DATA ###\nMarket Cap: {market_cap}\nTrailing P/E: {trailing_pe}\nForward P/E: {forward_pe}\nTrailing EPS: {eps}\nBusiness Summary: {long_business_summary}\n### END FINANCIAL DATA ###"
    except Exception as e:
        return f"Error fetching stock info for {ticker}: {e}"
