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
        Checks if a file is older than the max age or malformed.
        """
        if not os.path.exists(path):
            return True
        try:
            # Check file modification time as a primary, simple check.
            file_mod_time = datetime.fromtimestamp(os.path.getmtime(path))
            file_age = datetime.now() - file_mod_time
            if file_age > timedelta(days=max_age_days):
                return True
            
            # Secondary check for empty files.
            df = pd.read_csv(path)
            return df.empty
        except Exception:
            # If any error occurs reading the file, treat it as stale.
            return True

    def _clean_spy_data(self, df_raw):
        """
        A consistent cleaning function to handle both clean (from yfinance)
        and malformed (from old cache file) SPY DataFrames.
        """
        # The malformed CSV, when read naively, has a 'Price' column from its junk header.
        # A clean DataFrame has 'Date' as its first column when read from a fresh CSV.
        if 'Price' in df_raw.columns:
            print("🔧 Malformed DataFrame detected. Applying corrective formatting...")
            
            # The actual data starts at row index 3 of the malformed file.
            clean_df = df_raw.iloc[3:].copy()
            
            # Manually provide the correct column names.
            clean_df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
            
            # Convert 'Date' column to datetime objects and set it as the index.
            clean_df['Date'] = pd.to_datetime(clean_df['Date'])
            clean_df.set_index('Date', inplace=True)
            
            return clean_df
        else:
            # The DataFrame is already clean (from yfinance or a healed cache file).
            # We just need to ensure the 'Date' column is the index.
            if 'Date' in df_raw.columns:
                df_raw['Date'] = pd.to_datetime(df_raw['Date'])
                df_raw.set_index('Date', inplace=True)
            return df_raw

    def get_sp500_tickers(self):
        """
        Fetches the current list of S&P 500 tickers.
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

    def get_all_data(self):
        """
        Main method to orchestrate loading of all data with universal post-processing.
        """
        print("\n--- 📊 Loading All Financial Data ---")
        
        price_data_is_stale = self._is_data_stale(config.PRICE_DATA_PATH, config.CACHE_MAX_AGE_DAYS)
        
        if price_data_is_stale:
            print("⏳ Price data cache is stale. Downloading new data...")
            tickers = self.get_sp500_tickers()
            if not tickers: return None, None, None
            price_df = yf.download(tickers, period=config.YFINANCE_PERIOD, auto_adjust=False).stack(level=1).rename_axis(['Date', 'Ticker']).reset_index()
            price_df.to_csv(config.PRICE_DATA_PATH, index=False)
        else:
            print(f"✅ Loading fresh cached price data from {config.PRICE_DATA_PATH}...")
            price_df = pd.read_csv(config.PRICE_DATA_PATH)

        if price_df is None or price_df.empty:
            print("❌ Could not load the main price data. Aborting.")
            return None, None, None
        
        price_df['Date'] = pd.to_datetime(price_df['Date'])
        price_df.set_index('Date', inplace=True)
            
        available_tickers = price_df['Ticker'].unique().tolist()
        
        if price_data_is_stale:
            print("⏳ Fundamental data cache is being refreshed because price data was stale...")
            data = [{'Ticker': t, **yf.Ticker(t).info} for t in tqdm(available_tickers, desc="Fetching Fundamentals")]
            fundamentals_df = pd.DataFrame(data).set_index('Ticker')
            required_cols = ['trailingPE', 'forwardPE', 'priceToBook', 'enterpriseToEbitda', 'profitMargins']
            fundamentals_df = fundamentals_df[[col for col in required_cols if col in fundamentals_df.columns]]
            fundamentals_df.to_csv(config.FUNDAMENTAL_DATA_PATH)
        else:
            print(f"✅ Loading fresh cached fundamental data from {config.FUNDAMENTAL_DATA_PATH}...")
            fundamentals_df = pd.read_csv(config.FUNDAMENTAL_DATA_PATH, index_col='Ticker')
        
        # --- START: UNIVERSAL LOGIC FOR SPY DATA ---
        if self._is_data_stale(config.SPY_DATA_PATH, config.CACHE_MAX_AGE_DAYS):
            print("⏳ Refreshing SPY data...")
            # 1. ACQUIRE DATA: Download fresh data from yfinance.
            spy_df_raw = yf.download('SPY', period=config.YFINANCE_PERIOD, auto_adjust=True)
            # Save it cleanly for next time.
            spy_df_raw.to_csv(config.SPY_DATA_PATH, index=True)
        else:
            print(f"✅ Loading fresh cached SPY data from {config.SPY_DATA_PATH}...")
            # 1. ACQUIRE DATA: Load the file naively from cache, making no assumptions.
            spy_df_raw = pd.read_csv(config.SPY_DATA_PATH)

        # 2. UNIVERSAL CLEANING: Apply the same cleaning function regardless of the source.
        spy_df = self._clean_spy_data(spy_df_raw)

        # 3. UNIVERSAL TYPE CONVERSION: This runs on the now-clean DataFrame.
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            if col in spy_df.columns:
                spy_df[col] = pd.to_numeric(spy_df[col], errors='coerce')
        # --- END: UNIVERSAL LOGIC ---

        print("\n--- ✅ All data loaded successfully! ---")
        return price_df, fundamentals_df, spy_df
