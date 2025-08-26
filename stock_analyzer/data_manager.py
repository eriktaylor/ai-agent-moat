# stock_analyzer/data_manager.py

import os
import json
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf
from pandas.tseries.offsets import BDay
from tqdm import tqdm

import config


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
    def _write_with_meta(self, df: pd.DataFrame, path: str) -> None:
        """Write CSV + sidecar meta with UTC fetched_at."""
        df.to_csv(path, index=False)
        meta = {"fetched_at": pd.Timestamp.utcnow().isoformat()}
        with open(f"{path}.meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    def _read_meta(self, path: str) -> dict | None:
        meta_path = f"{path}.meta.json"
        if not os.path.exists(meta_path):
            return None
        try:
            with open(meta_path, "r") as f:
                return json.load(f)
        except Exception:
            return None

    # ---------------------------
    # Freshness logic
    # ---------------------------
    def _is_data_stale(self, path: str, max_age_days: int, *, date_col: str | None) -> bool:
        """
        Determine staleness using:
          - meta fetched_at (business days)
          - latest date in CSV (business days)
          - fallback to file mtime
        Returns True if file is missing, empty, unreadable, or older than threshold.
        """
        # Missing file
        if not os.path.exists(path):
            print(f"🔎 Freshness: {os.path.basename(path)} → missing → STALE")
            return True

        now = pd.Timestamp.utcnow().normalize()

        # 1) Sidecar meta (preferred)
        meta = self._read_meta(path)
        if meta and "fetched_at" in meta:
            try:
                fetched_at = pd.to_datetime(meta["fetched_at"], utc=True).normalize()
                age = now - fetched_at
                stale = age > BDay(max_age_days)
                print(
                    f"🔎 Freshness: {os.path.basename(path)} via meta fetched_at={fetched_at.date()} "
                    f"(age={age}) → stale={stale}"
                )
                return bool(stale)
            except Exception:
                # fall through to content-based
                pass

        # 2) Content-based (latest date in CSV)
        try:
            df = pd.read_csv(path)
            if df.empty:
                print(f"🔎 Freshness: {os.path.basename(path)} → empty → STALE")
                return True

            dc = date_col
            if not dc or dc not in df.columns:
                # Try to discover a date column if not provided
                for cand in ("Date", "date", "DATE", "AsOf", "asof"):
                    if cand in df.columns:
                        dc = cand
                        break

            if dc and dc in df.columns:
                # parse dates
                df[dc] = pd.to_datetime(df[dc], errors="coerce", utc=True)
                last = df[dc].max()
                if pd.isna(last):
                    print(
                        f"🔎 Freshness: {os.path.basename(path)} → could not parse '{dc}' → STALE"
                    )
                    return True
                age = now - last.normalize()
                stale = age > BDay(max_age_days)
                print(
                    f"🔎 Freshness: {os.path.basename(path)} via last {dc}={last.date()} "
                    f"(age={age}) → stale={stale}"
                )
                return bool(stale)
        except Exception:
            print(f"🔎 Freshness: {os.path.basename(path)} → read/parse error → STALE")
            return True

        # 3) Fallback to file mtime (least reliable in CI)
        try:
            mtime = pd.Timestamp(os.path.getmtime(path), unit="s", tz="UTC").normalize()
            age = now - mtime
            stale = age > BDay(max_age_days)
            print(
                f"🔎 Freshness: {os.path.basename(path)} via mtime={mtime.date()} "
                f"(age={age}) → stale={stale}"
            )
            return bool(stale)
        except Exception:
            print(f"🔎 Freshness: {os.path.basename(path)} → mtime error → STALE")
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
            price_df["Date"] = pd.to_datetime(price_df["Date"], utc=True)
            self._write_with_meta(price_df, config.PRICE_DATA_PATH)
        else:
            print(f"✅ Loading fresh cached price data from {config.PRICE_DATA_PATH}...")
            price_df = pd.read_csv(config.PRICE_DATA_PATH, parse_dates=["Date"])
        if price_df is None or price_df.empty:
            print("❌ Could not load the main price data. Aborting.")
            return None, None, None

        # ---------------- Fundamentals ----------------
        # Treat fundamentals independently (don’t hinge on price staleness)
        fundamentals_stale = self._is_data_stale(
            config.FUNDAMENTAL_DATA_PATH, config.CACHE_MAX_AGE_DAYS, date_col="AsOf"
        )
        fundamentals_stale=False
        if fundamentals_stale:
            print("⏳ Refreshing fundamentals (ticker-by-ticker via yfinance)...")
            available_tickers = price_df["Ticker"].dropna().unique().tolist()
            # Pull a small subset of fields; yfinance.info can be slow
            rows = []
            for t in tqdm(available_tickers, desc="Fetching Fundamentals"):
                try:
                    info = yf.Ticker(t).info
                except Exception:
                    info = {}
                rows.append(
                    {
                        "Ticker": t,
                        "trailingPE": info.get("trailingPE"),
                        "forwardPE": info.get("forwardPE"),
                        "priceToBook": info.get("priceToBook"),
                        "enterpriseToEbitda": info.get("enterpriseToEbitda"),
                        "profitMargins": info.get("profitMargins"),
                        # Optional: stamp an AsOf date for content freshness in future
                        "AsOf": pd.Timestamp.utcnow().normalize(),
                    }
                )
            fundamentals_df = pd.DataFrame(rows)
            self._write_with_meta(fundamentals_df, config.FUNDAMENTAL_DATA_PATH)
        else:
            print(f"✅ Loading fresh cached fundamental data from {config.FUNDAMENTAL_DATA_PATH}...")
            fundamentals_df = pd.read_csv(
                config.FUNDAMENTAL_DATA_PATH,
                parse_dates=["AsOf"],
                infer_datetime_format=True,
            )
        
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
            #spy_df.reset_index(inplace=True)
            # 3. SAVE: Save the clean, consistently formatted data for next time.
            self._write_with_meta(spy_df, config.SPY_DATA_PATH)
        else:
            print(f"✅ Loading clean cached SPY data from {config.SPY_DATA_PATH}...")
            # ACQUIRE: Load the already-clean file directly into the final variable.
            spy_df = pd.read_csv(config.SPY_DATA_PATH)

        # Convert numeric columns to numeric types.
        """
        spy_df['Date'] = pd.to_datetime(spy_df['Date'])
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume'] 
        for col in numeric_cols:
            if col in spy_df.columns:
                spy_df[col] = pd.to_numeric(spy_df[col], errors='coerce') 
            
        # Optional: drop first row (avoid partial first day)
        if len(spy_df) > 0:
            spy_df = spy_df.iloc[1:].copy()
        """
        
        print("\n--- ✅ All data loaded successfully! ---")
        return price_df, fundamentals_df, spy_df
