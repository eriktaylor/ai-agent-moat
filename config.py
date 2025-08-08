# config.py

"""
Centralized configuration file for the stock analysis and portfolio management system.
"""

import os

# --- Path Configurations ---
# Using os.path.join for cross-platform compatibility.
DRIVE_PATH = '/content/drive/My Drive/'
DATA_DIR = os.path.join(DRIVE_PATH, 'stock_data_cache')
PRICE_DATA_PATH = os.path.join(DATA_DIR, 'sp500_price_data.csv')
FUNDAMENTAL_DATA_PATH = os.path.join(DATA_DIR, 'sp500_fundamental_data.csv')
SPY_DATA_PATH = os.path.join(DATA_DIR, 'spy_market_data.csv')
PORTFOLIO_PATH = os.path.join(DRIVE_PATH, 'live_portfolio.csv')

# --- Data Fetching Parameters ---
# How old a cached file can be before we refresh it.
CACHE_MAX_AGE_DAYS = 5
# The historical data period to download.
YFINANCE_PERIOD = "5y"

# --- Candidate Generation (LightGBM) Parameters ---
# The target is to find stocks that will be in the top 20% of performers.
TARGET_QUANTILE = 0.8
# We look 21 trading days (approx. 1 month) into the future to define the target.
TARGET_FORWARD_PERIOD = 21
# The date to split for training vs. testing the evaluation model (6 months ago).
EVAL_SPLIT_MONTHS = 6
# Number of top candidates to generate from the LightGBM model.
TOP_N_CANDIDATES = 20

# --- Agentic Layer Parameters ---
# Number of top candidates to pass from the scout to the deep-dive analysis.
AGENT_DEEP_DIVE_CANDIDATES = 3

# --- Portfolio Management Parameters ---
# The maximum number of stocks to hold in the portfolio.
MAX_PORTFOLIO_SIZE = 10
# The minimum confidence score from the portfolio manager to initiate a "buy" action.
MIN_BUY_CONFIDENCE = 0.7
# The confidence score below which the manager will consider selling a holding.
SELL_CONFIDENCE_THRESHOLD = 0.4
