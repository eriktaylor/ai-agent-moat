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
        """
        print("🚀 Starting feature engineering...")
        features_df = df.copy()
        
        # Ensure date is the index for time-series operations
        if 'date' in features_df.columns:
            features_df.set_index('date', inplace=True)
        
        grouped = features_df.groupby('ticker')

        # --- Momentum Features ---
        for lag in [5, 21, 63, 252]:
            features_df[f'return_{lag}d'] = grouped['adj close'].transform(lambda x: x.pct_change(lag))

        # --- Volatility Features ---
        stock_returns = np.log(grouped['adj close'].transform(lambda x: x / x.shift(1)))
        for lag in [21, 63, 252]:
            features_df[f'volatility_{lag}d'] = stock_returns.groupby(features_df['ticker']).transform(lambda x: x.rolling(lag).std())

        # --- Market-Relative Features ---
        if 'date' in spy_df.columns:
            spy_df.set_index('date', inplace=True)
            
        market_returns = np.log(spy_df['close'] / spy_df['close'].shift(1)).rename('market_return')
        
        # Use a robust merge to add market returns
        features_df = features_df.reset_index().merge(market_returns.to_frame(), on='date', how='left')

        def rolling_beta(stock_return, market_return, window=63):
            # The stock_return and market_return are now aligned within the apply function
            cov = stock_return.rolling(window=window).cov(market_return)
            market_var = market_return.rolling(window=window).var()
            return cov / market_var
        
        # --- FIX for Beta Calculation Alignment ---
        # The previous assignment failed due to index misalignment. A robust merge is safer.
        # Calculate beta. The result is a Series with a (ticker, date) MultiIndex.
        beta_series = features_df.groupby('ticker').apply(
            lambda x: rolling_beta(np.log(x['adj close'] / x['adj close'].shift(1)), x['market_return'])
        )

        # Convert the resulting series to a DataFrame for merging.
        beta_df = beta_series.to_frame(name='beta_63d').reset_index()

        # Merge the beta values back using a robust merge on both date and ticker.
        features_df = pd.merge(features_df, beta_df, on=['date', 'ticker'], how='left')
        # --- END FIX ---

        # --- Clean Up ---
        features_df = features_df.drop(columns=['open', 'high', 'low', 'close', 'volume', 'market_return'])
        
        return features_df.dropna()


    def _define_target(self, df):
        """
        Defines the target variable using a robust merge-based approach to avoid index alignment issues.
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
