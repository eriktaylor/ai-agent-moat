# data_manager.py

"""
Handles all data fetching, caching, and loading operations for the stock analysis pipeline.
"""

import os
import pandas as pd
import yfinance as yf
from tqdm import tqdm
from datetime import datetime, timedelta
import config  # Import the configuration file

class DataManager:
    """
    A class to manage financial data, including fetching from yfinance and caching.
    """
    def __init__(self):
        """
        Initializes the DataManager and ensures the data directory exists.
        """
        # Ensure the directory for caching data exists.
        os.makedirs(config.DATA_DIR, exist_ok=True)
        print(f"Data directory is set to: {config.DATA_DIR}")

    def _is_cache_stale(self, path, max_age_days):
        """
        Checks if a cached file is older than the specified maximum age.
        """
        if not os.path.exists(path):
            return True  # File doesn't exist, so it's "stale"

        file_mod_time = datetime.fromtimestamp(os.path.getmtime(path))
        file_age = datetime.now() - file_mod_time
        print(f"Cache file '{os.path.basename(path)}' is {file_age} old.")
        return file_age > timedelta(days=max_age_days)

    def get_sp500_tickers(self):
        """
        Fetches the list of S&P 500 tickers from the SSGA website.
        """
        print("Fetching S&P 500 tickers...")
        try:
            url = 'https://www.ssga.com/us/en/intermediary/etfs/library-content/products/fund-data/etfs/us/holdings-daily-us-en-spy.xlsx'
            df = pd.read_excel(url, engine='openpyxl', skiprows=4).dropna(subset=['Ticker'])
            tickers = [str(t).replace(' ', '-').replace('.', '-') for t in df['Ticker'].tolist() if isinstance(t, str)]
            print(f"✅ Successfully fetched {len(tickers)} S&P 500 tickers.")
            return tickers
        except Exception as e:
            print(f"❌ Error fetching S&P 500 tickers: {e}")
            return []

    def load_or_fetch_price_data(self):
        """
        Loads price data from cache if fresh, otherwise downloads and caches it.
        """
        if not self._is_cache_stale(config.PRICE_DATA_PATH, config.CACHE_MAX_AGE_DAYS):
            print(f"✅ Loading fresh cached price data from {config.PRICE_DATA_PATH}...")
            return pd.read_csv(config.PRICE_DATA_PATH, parse_dates=['Date'], index_col='Date')

        print("⏳ Price data cache is stale or missing. Downloading new data...")
        tickers = self.get_sp500_tickers()
        if not tickers:
            return None

        df = yf.download(tickers, period=config.YFINANCE_PERIOD, auto_adjust=False).stack(level=1).rename_axis(['Date', 'Ticker']).reset_index()
        print(f"💾 Saving new price data to {config.PRICE_DATA_PATH}...")
        df.to_csv(config.PRICE_DATA_PATH, index=False)
        return pd.read_csv(config.PRICE_DATA_PATH, parse_dates=['Date'], index_col='Date')

    def load_or_fetch_fundamentals(self, tickers):
        """
        Loads fundamental data from cache if fresh, otherwise downloads it.
        """
        if not self._is_cache_stale(config.FUNDAMENTAL_DATA_PATH, config.CACHE_MAX_AGE_DAYS):
            print(f"✅ Loading fresh cached fundamental data from {config.FUNDAMENTAL_DATA_PATH}...")
            return pd.read_csv(config.FUNDAMENTAL_DATA_PATH, index_col='Ticker')

        print("⏳ Fundamental data cache is stale or missing. Downloading new data...")
        data = [{'Ticker': t, **yf.Ticker(t).info} for t in tqdm(tickers, desc="Fetching Fundamentals")]
        df = pd.DataFrame(data).set_index('Ticker')

        # Select only the columns we need to avoid errors with complex objects
        # and ensure consistency.
        required_cols = ['trailingPE', 'forwardPE', 'priceToBook', 'enterpriseToEbitda', 'profitMargins']
        df = df[[col for col in required_cols if col in df.columns]]

        print(f"💾 Saving new fundamental data to {config.FUNDAMENTAL_DATA_PATH}...")
        df.to_csv(config.FUNDAMENTAL_DATA_PATH)
        return df

    def load_or_fetch_spy_data(self):
        """
        Loads SPY market data from cache if fresh, otherwise downloads it.
        """
        if not self._is_cache_stale(config.SPY_DATA_PATH, config.CACHE_MAX_AGE_DAYS):
            print(f"✅ Loading fresh cached SPY data from {config.SPY_DATA_PATH}...")
            return pd.read_csv(config.SPY_DATA_PATH, parse_dates=['Date'], index_col='Date')

        print("⏳ SPY data cache is stale or missing. Downloading new data...")
        df = yf.download('SPY', period=config.YFINANCE_PERIOD, auto_adjust=True).stack(level=1).rename_axis(['Date', 'Ticker']).reset_index()
        print(f"💾 Saving new SPY data to {config.SPY_DATA_PATH}...")
        df.to_csv(config.SPY_DATA_PATH)
        return pd.read_csv(config.SPY_DATA_PATH, parse_dates=['Date'], index_col='Date')

    def get_all_data(self):
        """
        Main method to orchestrate the loading of all required data.
        """
        print("\n--- 📊 Loading All Financial Data ---")
        price_df = self.load_or_fetch_price_data()

        if price_df is None or price_df.empty:
            print("❌ Could not load the main price data. Aborting.")
            return None, None, None

        available_tickers = price_df['Ticker'].unique().tolist()
        fundamentals_df = self.load_or_fetch_fundamentals(available_tickers)
        spy_df = self.load_or_fetch_spy_data()

        print("\n--- ✅ All data loaded successfully! ---")
        print(f"Price Data Shape:      {price_df.shape}")
        print(f"Fundamental Data Shape:  {fundamentals_df.shape}")
        print(f"SPY Market Data Shape:   {spy_df.shape}")

        return price_df, fundamentals_df, spy_df

