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

    def _is_data_stale(self, path, max_age_days):
        """
        Checks if the data inside a time-series CSV is older than the max age.
        """
        if not os.path.exists(path):
            print(f"Cache file '{os.path.basename(path)}' not found. Marked as stale.")
            return True
        try:
            # Efficiently read only the 'Date' column to check the last date
            df = pd.read_csv(path, usecols=['Date'])
            if df.empty:
                print(f"Cache file '{os.path.basename(path)}' is empty. Marked as stale.")
                return True
            last_date_in_file = pd.to_datetime(df['Date'].iloc[-1])
            data_age = datetime.now() - last_date_in_file
            print(f"Latest data in '{os.path.basename(path)}' is from {last_date_in_file.date()} ({data_age.days} days old).")
            return data_age > timedelta(days=max_age_days)
        except (ValueError, KeyError):
            print(f"Could not determine data age from '{os.path.basename(path)}'. Using file modification time instead.")
            file_mod_time = datetime.fromtimestamp(os.path.getmtime(path))
            file_age = datetime.now() - file_mod_time
            return file_age > timedelta(days=max_age_days)

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

    def get_all_data(self):
        """
        Main method to orchestrate loading of all data. It now uses a dependency-based
        refresh logic and correctly sets the index on the returned DataFrames.
        """
        print("\n--- 📊 Loading All Financial Data ---")
        
        price_data_is_stale = self._is_data_stale(config.PRICE_DATA_PATH, config.CACHE_MAX_AGE_DAYS)
        
        if price_data_is_stale:
            print("⏳ Price data cache is stale. Downloading new data...")
            tickers = self.get_sp500_tickers()
            if not tickers: return None, None, None
            price_df = yf.download(tickers, period=config.YFINANCE_PERIOD, auto_adjust=False).stack(level=1).rename_axis(['Date', 'Ticker']).reset_index()
            print(f"💾 Saving new price data to {config.PRICE_DATA_PATH}...")
            price_df.to_csv(config.PRICE_DATA_PATH, index=False)
        else:
            print(f"✅ Loading fresh cached price data from {config.PRICE_DATA_PATH}...")
            price_df = pd.read_csv(config.PRICE_DATA_PATH)

        if price_df is None or price_df.empty:
            print("❌ Could not load the main price data. Aborting.")
            return None, None, None
        
        # --- THIS IS THE FIX ---
        # Ensure the 'Date' column is converted to datetime and set as the index
        # for the price_df DataFrame, regardless of its source.
        price_df['Date'] = pd.to_datetime(price_df['Date'])
        price_df.set_index('Date', inplace=True)
        # --- END OF FIX ---
            
        available_tickers = price_df['Ticker'].unique().tolist()
        
        if price_data_is_stale:
            print("⏳ Fundamental data cache is being refreshed because price data was stale...")
            data = [{'Ticker': t, **yf.Ticker(t).info} for t in tqdm(available_tickers, desc="Fetching Fundamentals")]
            fundamentals_df = pd.DataFrame(data).set_index('Ticker')
            required_cols = ['trailingPE', 'forwardPE', 'priceToBook', 'enterpriseToEbitda', 'profitMargins']
            fundamentals_df = fundamentals_df[[col for col in required_cols if col in fundamentals_df.columns]]
            print(f"💾 Saving new fundamental data to {config.FUNDAMENTAL_DATA_PATH}...")
            fundamentals_df.to_csv(config.FUNDAMENTAL_DATA_PATH)
        else:
            print(f"✅ Loading fresh cached fundamental data from {config.FUNDAMENTAL_DATA_PATH}...")
            fundamentals_df = pd.read_csv(config.FUNDAMENTAL_DATA_PATH, index_col='Ticker')
        
        if self._is_data_stale(config.SPY_DATA_PATH, config.CACHE_MAX_AGE_DAYS):
            print("⏳ Refreshing SPY data...")
            spy_df = yf.download('SPY', period=config.YFINANCE_PERIOD, auto_adjust=True)
            spy_df.to_csv(config.SPY_DATA_PATH)
        else:
            print(f"✅ Loading fresh cached SPY data from {config.SPY_DATA_PATH}...")
            spy_df = pd.read_csv(config.SPY_DATA_PATH, parse_dates=['Date'], index_col='Date')

        print("\n--- ✅ All data loaded successfully! ---")
        return price_df, fundamentals_df, spy_df
