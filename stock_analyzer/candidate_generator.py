# stock_analyzer/candidate_generator.py

import pandas as pd
import numpy as np
import lightgbm as lgb
import config

class CandidateGenerator:
    """
    Uses a machine learning model to screen and rank stocks based on quantitative factors.
    """
    def _create_features(self, df, spy_df):
        """
        Engineers features for the model from the raw data.
        This version avoids using set_index to prevent state-related errors.
        """
        print("🚀 Starting feature engineering...")
        features_df = df.copy()  # 'date' is a column here

        # --- Momentum and Volatility Features ---
        # These can be calculated before market data is added.
        grouped_by_ticker = features_df.groupby('ticker')
        for lag in [5, 21, 63, 252]:
            features_df[f'return_{lag}d'] = grouped_by_ticker['adj close'].transform(lambda x: x.pct_change(lag))

        log_returns = np.log(grouped_by_ticker['adj close'].transform(lambda x: x / x.shift(1)))
        for lag in [21, 63, 252]:
            features_df[f'volatility_{lag}d'] = log_returns.groupby(features_df['ticker']).transform(lambda x: x.rolling(lag).std())

        # --- Market-Relative Features ---
        spy_df['market_return'] = np.log(spy_df['close'] / spy_df['close'].shift(1))
        features_df = pd.merge(features_df, spy_df[['date', 'market_return']], on='date', how='left')

        # --- Beta Calculation ---
        # **CRITICAL FIX**: Perform the groupby AFTER the 'market_return' column has been merged.
        def rolling_beta(x, window=63):
            log_return = np.log(x['adj close'] / x['adj close'].shift(1))
            cov = log_return.rolling(window=window).cov(x['market_return'])
            market_var = x['market_return'].rolling(window=window).var()
            return cov / market_var # Return the actual beta value

        # Apply the function to each ticker group.
        beta_values = features_df.groupby('ticker').apply(rolling_beta).rename('beta_63d').reset_index()
        
        # Merge the calculated beta values back into the main DataFrame.
        features_df = pd.merge(features_df, beta_values, on=['date', 'ticker'], how='left')
        
        # --- Clean Up ---
        features_df = features_df.drop(columns=['open', 'high', 'low', 'close', 'volume', 'market_return'])
        
        return features_df.dropna()


    def _define_target(self, df):
        """
        Defines the target variable. This function expects 'df' to have a 'date' column.
        """
        df_copy = df.copy()
        df_copy['future_return'] = df_copy.groupby('ticker')['adj close'].shift(-config.TARGET_FORWARD_PERIOD) / df_copy['adj close'] - 1
        df_copy = df_copy.dropna(subset=['future_return'])

        daily_cutoffs = df_copy.groupby('date')['future_return'].quantile(config.TARGET_QUANTILE).rename('cutoff')

        df_copy = df_copy.merge(daily_cutoffs, on='date', how='left')

        df_copy['target'] = (df_copy['future_return'] >= df_copy['cutoff']).astype(int)

        return df_copy.drop(columns=['future_return', 'cutoff'])


    def generate_candidates(self, price_df, fundamentals_df, spy_df):
        """
        Main method to run feature engineering, model training, and candidate prediction.
        """
        price_df.columns = price_df.columns.str.lower()
        fundamentals_df.columns = fundamentals_df.columns.str.lower()
        spy_df.columns = spy_df.columns.str.lower()
        
        spy_df['ticker'] = 'SPY'
        
        price_df['date'] = pd.to_datetime(price_df['date'])
        spy_df['date'] = pd.to_datetime(spy_df['date'])
        
        df = pd.merge(price_df, fundamentals_df, on='ticker', how='left')
    
        features_df = self._create_features(df, spy_df)
        
        final_df = self._define_target(features_df)
        
        if 'date' in final_df.columns:
            final_df.set_index('date', inplace=True)

        X = final_df.drop(columns=['target', 'ticker', 'adj close'])
        y = final_df['target']
    
        print("--- 🏋️ Training Production Model on All Data ---")
        
        counts = y.value_counts()
        if 0 in counts and 1 in counts and counts[1] > 0:
            scale_pos_weight = counts[0] / counts[1]
        else:
            scale_pos_weight = 1
    
        prod_model = lgb.LGBMClassifier(objective='binary', scale_pos_weight=scale_pos_weight, random_state=42)
        prod_model.fit(X, y)
    
        feature_imp = pd.DataFrame(
            sorted(zip(prod_model.feature_importances_, X.columns)),
            columns=['Value','Feature']
        )
        feature_imp_sorted = feature_imp.sort_values(by="Value", ascending=False).head(10)
    
        print("--- Predicting on Most Recent Data ---")
        latest_data = final_df.loc[final_df.index == final_df.index.max()]
        
        if latest_data.empty:
            print("⚠️ No recent data available for prediction.")
            return pd.DataFrame(), pd.DataFrame()

        latest_X = latest_data.drop(columns=['target', 'ticker', 'adj close'])
    
        probabilities = prod_model.predict_proba(latest_X)[:, 1]
        candidates = pd.DataFrame({
            'Ticker': latest_data['ticker'],
            'Quant_Score': probabilities
        }).sort_values(by='Quant_Score', ascending=False).reset_index(drop=True)
    
        top_candidates = candidates.head(config.TOP_N_CANDIDATES)
    
        print(f"💾 Saving top {len(top_candidates)} candidates to {config.CANDIDATE_RESULTS_PATH}...")
        top_candidates.to_csv(config.CANDIDATE_RESULTS_PATH, index=False)
    
        return top_candidates, feature_imp_sorted
