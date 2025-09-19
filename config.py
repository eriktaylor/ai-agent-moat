# config.py

"""
Centralized configuration file for the stock analysis and portfolio management system.
"""

# config.py

import os

# --- Path Configurations ---
DATA_DIR = 'data'
PRICE_DATA_PATH = os.path.join(DATA_DIR, 'sp500_price_data.csv')
FUNDAMENTAL_DATA_PATH = os.path.join(DATA_DIR, 'sp500_fundamental_data.csv')
SPY_DATA_PATH = os.path.join(DATA_DIR, 'spy_market_data.csv')
PORTFOLIO_PATH = os.path.join(DATA_DIR, 'live_portfolio.csv')
CANDIDATE_RESULTS_PATH = os.path.join(DATA_DIR, 'quantitative_candidates.csv')
AGENTIC_RESULTS_PATH = os.path.join(DATA_DIR, 'agentic_recommendations.csv')

# --- Data Fetching Parameters ---
# How old a cached file can be before we refresh it.
CACHE_MAX_AGE_DAYS = 3
# The historical data period to download.
YFINANCE_PERIOD = "3y"

# --- Candidate Generation (LightGBM) Parameters ---
TARGET_QUANTILE = 0.8
TARGET_FORWARD_PERIOD = 21
EVAL_SPLIT_MONTHS = 6
TOP_N_CANDIDATES = 250

# --- Agentic Layer Parameters ---
# The number of top-ranked candidates from the quantitative model to pass to the agentic layer.
QUANT_DEEP_DIVE_CANDIDATES = 3
#Use the scout agent to find new tickers outside the S&P500?
ENABLE_SCOUT = False
# The maximum number of new tickers for the scout agent to propose.
MAX_SCOUT_RESULTS = 3
#ALso use Yahoo finance news to bolster reports.
ENABLE_YF_NEWS = True
YF_NEWS_MAX = 12

# --- Portfolio Management Parameters ---
MAX_PORTFOLIO_SIZE = 10
MIN_BUY_CONFIDENCE = 0.7
SELL_CONFIDENCE_THRESHOLD = 0.4

