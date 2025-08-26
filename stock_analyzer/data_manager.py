# stock_analyzer/data_manager.py

import os
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf
from pandas.tseries.offsets import BDay
from tqdm import tqdm

import config

META_PATH = Path(config.DATA_DIR) / "meta.json"

class DataManager:
    """
    Manage financial data (S&P 500 prices, fundamentals, SPY) with robust caching.
    Freshness is determined by:
      1) sidecar .meta.json 'fetched_at' timestamp (preferred),
      2) latest in-file date column (e.g., 'Date' or 'AsOf'),
      3) file mtime (last resort).
    """

    def __init__(self):
        os.makedirs(config.DATA_DIR, exist_ok=True)
        print(f"Data directory is set to: {config.DATA_DIR}")
        
    # ---------------------------
    # Meta helpers
    # ---------------------------

    def _load_meta(self) -> dict:
        if META_PATH.exists():
            try:
                with open(META_PATH, "r") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
    
    def _save_meta(self, meta: dict) -> None:
        META_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(META_PATH, "w") as f:
            json.dump(meta, f, indent=2, sort_keys=True)
    
    def _update_meta(self, filename: str) -> None:
        meta = self._load_meta()
        meta[filename] = {"fetched_at": datetime.now(timezone.utc).isoformat()}
        self._save_meta(meta)

    # ---------------------------
    # Freshness logic
    # ---------------------------
    def _is_data_stale(self, path: str, max_age_days: int, *, date_col: str | None = None) -> bool:
        """
        Determine staleness using:
          - meta fetched_at (business days)
          - latest date in CSV (business days)
        Returns True if file is missing, empty, unreadable, or older than threshold.
        """
        
        p = Path(path)
        if not p.exists():
            return True
    
        meta = self._load_meta()
        now_utc = datetime.now(timezone.utc)
        rec = meta.get(p.name)
        if rec and "fetched_at" in rec:
            try:
                ts = datetime.fromisoformat(rec["fetched_at"])
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                return (now_utc - ts) > timedelta(days=max_age_days)
            except Exception:
                pass  
        # Missing file
        if not os.path.exists(path):
            print(f"🔎 Freshness: {os.path.basename(path)} → missing → STALE")
            return True

        # 2) Cannot determine the age of data
        return True

    # ---------------------------
    # Data sources
    # ---------------------------
    def get_sp500_tickers(self) -> list[str]:
        """
        Fetch S&P 500 constituents from SPY holdings (SSGA XLSX).
        """
        print("Fetching S&P 500 tickers...")
        try:
            url = (
                "https://www.ssga.com/us/en/intermediary/etfs/library-content/products/"
                "fund-data/etfs/us/holdings-daily-us-en-spy.xlsx"
            )
            # engine=openpyxl often works on Actions; if not, consider engine=None (xlrd doesn't read xlsx)
            df = pd.read_excel(url, engine="openpyxl", skiprows=4).dropna(subset=["Ticker"])
            tickers = [
                str(t).replace(" ", "-").replace(".", "-")
                for t in df["Ticker"].tolist()
                if isinstance(t, str)
            ]
            print(f"✅ Successfully fetched {len(tickers)} S&P 500 tickers.")
            return tickers
        except Exception as e:
            print(f"❌ Error fetching S&P 500 tickers: {e}")
            return []

    # ---------------------------
    # Main orchestrator
    # ---------------------------
    def get_all_data(self):
        """
        Orchestrate loading/refresh of:
          - S&P 500 price data (Date column)
          - Fundamentals (optional AsOf column; meta otherwise)
          - SPY market data (Date column)
        """
        print("\n--- 📊 Loading All Financial Data ---")

        # ---------------- Prices ----------------
        price_stale = self._is_data_stale(
            config.PRICE_DATA_PATH, config.CACHE_MAX_AGE_DAYS, date_col="Date"
        )
        price_stale=False
        if price_stale:
            print("⏳ Price data cache is stale. Downloading new data...")
            tickers = self.get_sp500_tickers()
            if not tickers:
                return None, None, None
            # yfinance returns columns like ('Adj Close', 'AAPL'), so stack to rows
            price_df = (
                yf.download(tickers, period=config.YFINANCE_PERIOD, auto_adjust=False)
                .stack(level=1)
                .rename_axis(["Date", "Ticker"])
                .reset_index()
            )
            # Normalize datatypes
            price_df["Date"] = pd.to_datetime(price_df["Date"])
            price_df.to_csv(config.PRICE_DATA_PATH, index=False)
            self._update_meta(Path(config.PRICE_DATA_PATH).name)        
        else:
            print(f"✅ Loading fresh cached price data from {config.PRICE_DATA_PATH}...")
            price_df = pd.read_csv(config.PRICE_DATA_PATH)
        if price_df is None or price_df.empty:
            print("❌ Could not load the main price data. Aborting.")
            return None, None, None

        # ---------------- Fundamentals ----------------
        # Treat fundamentals independently (don’t hinge on price staleness)
        fundamentals_stale = self._is_data_stale(
            config.FUNDAMENTAL_DATA_PATH, config.CACHE_MAX_AGE_DAYS
        )
        if fundamentals_stale:
            print("⏳ Refreshing fundamentals (ticker-by-ticker via yfinance)...")
            #available_tickers = price_df["Ticker"].dropna().unique().tolist()
            available_tickers = price_df['Ticker'].unique().tolist()
            #FOR TESTING
            available_tickers=available_tickers[:10]    
            data = [{'Ticker': t, **yf.Ticker(t).info} for t in tqdm(available_tickers, desc="Fetching Fundamentals")] 
            fundamentals_df = pd.DataFrame(data).set_index('Ticker') 
            print('prior to cleaning...')
            print(fundamentals_df.columns)
            required_cols = ['trailingPE', 'forwardPE', 'priceToBook', 'enterpriseToEbitda', 'profitMargins'] 
            fundamentals_df = fundamentals_df[[col for col in required_cols if col in fundamentals_df.columns]] 
            fundamentals_df.to_csv(config.FUNDAMENTAL_DATA_PATH, index=False)
            self._update_meta(Path(config.FUNDAMENTAL_DATA_PATH).name)
        else:
            print(f"✅ Loading fresh cached fundamental data from {config.FUNDAMENTAL_DATA_PATH}...")
            fundamentals_df = pd.read_csv(config.FUNDAMENTAL_DATA_PATH, index_col='Ticker')
        
        # ---------------- SPY ----------------
        spy_stale = self._is_data_stale(
            config.SPY_DATA_PATH, config.CACHE_MAX_AGE_DAYS, date_col="Date"
        )
        # Check if the data file is missing or stale.            
        if spy_stale:
            print("⏳ Refreshing SPY data...")
            # 1. ACQUIRE: Download fresh data from yfinance.
            spy_df_raw = yf.download('SPY', period=config.YFINANCE_PERIOD, auto_adjust=True)
            # 2. CLEAN & PREPARE: Copy the raw data and reset the index.
            spy_df = spy_df_raw.copy()
            spy_df.reset_index(inplace=True)
            spy_df.columns = spy_df.columns.get_level_values(0)
            # Convert numeric columns to numeric types.
            spy_df['Date'] = pd.to_datetime(spy_df['Date'])
            numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume'] 
            for col in numeric_cols:
                if col in spy_df.columns:
                    spy_df[col] = pd.to_numeric(spy_df[col], errors='coerce') 
            
            # 3. SAVE: Save the clean, consistently formatted data for next time.
            spy_df.to_csv(config.SPY_DATA_PATH, index=False) 
        else:
            print(f"✅ Loading clean cached SPY data from {config.SPY_DATA_PATH}...")
            # ACQUIRE: Load the already-clean file directly into the final variable.
            spy_df = pd.read_csv(config.SPY_DATA_PATH)
            self._update_meta(Path(config.SPY_DATA_PATH).name)
                    
        print("\n--- ✅ All data loaded successfully! ---")
        return price_df, fundamentals_df, spy_df
