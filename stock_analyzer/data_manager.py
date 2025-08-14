# stock_analyzer/data_manager.py

import os
import pandas as pd
import yfinance as yf
from tqdm import tqdm
from datetime import datetime, timedelta
import config

class DataManager:
    """
    A class to manage financial data, including fetching from yfinance and caching.
    """
    def __init__(self):
        os.makedirs(config.DATA_DIR, exist_ok=True)
        print(f"Data directory is set to: {config.DATA_DIR}")

    def _is_cache_stale(self, path, max_age_days):
        """
        Checks if the data inside the cached file is older than the max age.
        This now checks the content of the CSV, not just the file's modification time.
        """
        if not os.path.exists(path):
            print(f"Cache file '{os.path.basename(path)}' not found. Marked as stale.")
            return True

        try:
            # Read only the last row to be efficient
            df = pd.read_csv(path, usecols=['Date'])
            if df.empty or 'Date' not in df.columns:
                 print(f"Cache file '{os.path.basename(path)}' is empty or has no 'Date' column. Marked as stale.")
                 return True

            last_date_in_file = pd.to_datetime(df['Date'].iloc[-1])
            data_age = datetime.now() - last_date_in_file
            
            print(f"Latest data in '{os.path.basename(path)}' is from {last_date_in_file.date()} ({data_age.days} days old).")
            
            # Data is stale if it's older than the max age, accounting for weekends
            # (e.g., data from Friday is not stale on Monday).
            return data_age > timedelta(days=max_age_days)

        except Exception as e:
            print(f"Error reading cache file '{os.path.basename(path)}': {e}. Marked as stale.")
            return True

    def get_sp500_tickers(self):
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

        print("⏳ Price data cache is stale. Downloading new data...")
        tickers = self.get_sp500_tickers()
        if not tickers: return None
        df = yf.download(tickers, period=config.YFINANCE_PERIOD, auto_adjust=False).stack(level=1).rename_axis(['Date', 'Ticker']).reset_index()
        print(f"💾 Saving new price data to {config.PRICE_DATA_PATH}...")
        df.to_csv(config.PRICE_DATA_PATH, index=False)
        return pd.read_csv(config.PRICE_DATA_PATH, parse_dates=['Date'], index_col='Date')

    def load_or_fetch_fundamentals(self, tickers):
        """
        Loads fundamental data from cache if fresh, otherwise downloads it.
        The staleness of fundamentals is tied to the price data's staleness.
        """
        # We don't need a separate date check for fundamentals; if price data is stale,
        # we should refresh fundamentals too.
        if not self._is_cache_stale(config.FUNDAMENTAL_DATA_PATH, config.CACHE_MAX_AGE_DAYS):
             print(f"✅ Loading fresh cached fundamental data from {config.FUNDAMENTAL_DATA_PATH}...")
             return pd.read_csv(config.FUNDAMENTAL_DATA_PATH, index_col='Ticker')

        print("⏳ Fundamental data cache is stale. Downloading new data...")
        data = [{'Ticker': t, **yf.Ticker(t).info} for t in tqdm(tickers, desc="Fetching Fundamentals")]
        df = pd.DataFrame(data).set_index('Ticker')
        required_cols = ['trailingPE', 'forwardPE', 'priceToBook', 'enterpriseToEbitda', 'profitMargins']
        df = df[[col for col in required_cols if col in df.columns]]
        print(f"💾 Saving new fundamental data to {config.FUNDAMENTAL_DATA_PATH}...")
        df.to_csv(config.FUNDAMENTAL_DATA_PATH)
        return df

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
        
        if self._is_cache_stale(config.SPY_DATA_PATH, config.CACHE_MAX_AGE_DAYS):
            print("⏳ Refreshing SPY data...")
            spy_df = yf.download('SPY', period=config.YFINANCE_PERIOD, auto_adjust=True)
            spy_df.to_csv(config.SPY_DATA_PATH)
        else:
            print(f"✅ Loading fresh cached SPY data from {config.SPY_DATA_PATH}...")
            spy_df = pd.read_csv(config.SPY_DATA_PATH, parse_dates=['Date'], index_col='Date')

        print("\n--- ✅ All data loaded successfully! ---")
        return price_df, fundamentals_df, spy_df
