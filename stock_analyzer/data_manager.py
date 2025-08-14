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
        print('before',df_raw)
        clean_df = df_raw.iloc[3:].copy()
        # Manually provide the correct column names.
        print('after',clean_df)
        clean_df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
        clean_df['Date'] = pd.to_datetime(clean_df['Date'])
    
        numeric_cols = ['Close', 'High', 'Low', 'Open', 'Volume']
        for col in numeric_cols:
            # Correctly apply to the clean_df
            clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce')
        
        return clean_df
    
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
        #price_df.set_index('Date', inplace=True)
            
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
            #fundamentals_df = pd.read_csv(config.FUNDAMENTAL_DATA_PATH, index_col='Ticker')
            fundamentals_df = pd.read_csv(config.FUNDAMENTAL_DATA_PATH)
        
        # --- START: UNIVERSAL LOGIC FOR SPY DATA ---
        # First, acquire the raw data from either the cache or a fresh download.
        if self._is_data_stale(config.SPY_DATA_PATH, config.CACHE_MAX_AGE_DAYS):
            print("⏳ Refreshing SPY data...")
            # 1. ACQUIRE DATA: Download fresh data.
            spy_df_raw = yf.download('SPY', period=config.YFINANCE_PERIOD, auto_adjust=True)
            # 2. CLEAN: Prepare the clean DataFrame.
            spy_df = spy_df_raw.copy()
            spy_df.reset_index(inplace=True) # Turns the 'Date' index into a 'Date' column.
            
            # At this point, spy_df has 6 columns: ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            
            # 3. SAVE: Save the clean, consistently formatted data.
            spy_df.to_csv(config.SPY_DATA_PATH, index=False) # Use index=False
        else:
            print(f"✅ Loading cached SPY data from {config.SPY_DATA_PATH}...")
            # 1. ACQUIRE DATA: Load the raw file from the cache.
            spy_df_raw = pd.read_csv(config.SPY_DATA_PATH)

        # --- UNIVERSAL TYPE CONVERSION ---
        # This now runs on a DataFrame that is guaranteed to be clean.
        #spy_df['Date'] = pd.to_datetime(spy_df['Date'])
        #numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        #for col in numeric_cols:
        #    if col in spy_df.columns:
        #        spy_df[col] = pd.to_numeric(spy_df[col], errors='coerce')

        print("\n--- ✅ All data loaded successfully! ---")
        return price_df, fundamentals_df, spy_df
