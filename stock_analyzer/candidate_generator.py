# stock_analyzer/candidate_generator.py

import pandas as pd
import numpy as np
import lightgbm as lgb
import config

class CandidateGenerator:
    """
    Uses a machine learning model to screen and rank stocks based on quantitative factors.
    This version includes fixes for data leakage and improved data handling.
    """
    def _create_features(self, df, spy_df):
        """
        Engineers features for the model from the raw data.
        """
        print("🚀 Starting feature engineering...")
        features_df = df.copy()

        # --- Momentum and Volatility Features ---
        grouped_by_ticker = features_df.groupby('ticker')
        for lag in [5, 21, 63, 252]:
            features_df[f'return_{lag}d'] = grouped_by_ticker['adj close'].transform(lambda x: x.pct_change(lag))

        log_returns = np.log(grouped_by_ticker['adj close'].transform(lambda x: x / x.shift(1)))
        for lag in [21, 63, 252]:
            features_df[f'volatility_{lag}d'] = log_returns.groupby(features_df['ticker']).transform(lambda x: x.rolling(lag).std())

        # --- Market-Relative Features ---
        spy_df['market_return'] = np.log(spy_df['close'] / spy_df['close'].shift(1))
        features_df = pd.merge(features_df, spy_df[['date', 'market_return']], on='date', how='left')

        # --- Beta Calculation (Robust Method) ---
        def calculate_beta(sub_df, window=63):
            log_return = np.log(sub_df['adj close'] / sub_df['adj close'].shift(1))
            market_var = sub_df['market_return'].rolling(window=window).var()
            covariance = log_return.rolling(window=window).cov(sub_df['market_return'])
            return covariance / market_var

        # --- CRITICAL FIX for AssertionError and level_1 bug ---
        # 1. Apply the function. The result is a Series with a (ticker, date) MultiIndex.
        beta_series = features_df.groupby('ticker').apply(calculate_beta)
        
        # 2. Convert to a DataFrame and reset the index. This creates 'ticker', 'date', and the beta column.
        beta_df = beta_series.rename('beta_63d').reset_index()
        
        # 3. Merge the beta values back using a safe, explicit merge. This avoids all index-related errors.
        features_df = pd.merge(features_df, beta_df, on=['date', 'ticker'], how='left')
        # --- END FIX ---
        
        # --- Clean Up ---
        features_df = features_df.drop(columns=['open', 'high', 'low', 'close', 'volume', 'market_return'])
        
        # --- Selective Dropna ---
        critical_features = ['return_252d', 'volatility_252d', 'beta_63d']
        return features_df.dropna(subset=critical_features)


    def _define_target(self, df):
        """
        Defines the target variable.
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
        
        df = pd.merge(
            price_df,
            fundamentals_df.drop(columns=['date'], errors='ignore'),
            on='ticker',
            how='left'
        )
    
        features_df = self._create_features(df, spy_df)
        final_df = self._define_target(features_df)
        
        max_date = final_df['date'].max()
        
        train_df = final_df[final_df['date'] < max_date]
        predict_df = final_df[final_df['date'] == max_date]

        if predict_df.empty:
            print("⚠️ No recent data available for prediction.")
            return pd.DataFrame(), pd.DataFrame()

        train_df = train_df.set_index('date')
        
        X_train = train_df.drop(columns=['target', 'ticker', 'adj close'])
        y_train = train_df['target']
    
        print(f"--- 🏋️ Training Production Model on Data up to {train_df.index.max().date()} ---")
        
        counts = y_train.value_counts()
        if 0 in counts and 1 in counts and counts[1] > 0:
            scale_pos_weight = counts[0] / counts[1]
        else:
            scale_pos_weight = 1
    
        prod_model = lgb.LGBMClassifier(objective='binary', scale_pos_weight=scale_pos_weight, random_state=42)
        prod_model.fit(X_train, y_train)
    
        feature_imp = pd.DataFrame(
            sorted(zip(prod_model.feature_importances_, X_train.columns)),
            columns=['Value','Feature']
        )
        feature_imp_sorted = feature_imp.sort_values(by="Value", ascending=False).head(10)
    
        print(f"--- Predicting on Most Recent Data ({max_date.date()}) ---")
        
        X_predict = predict_df.set_index('date').drop(columns=['target', 'ticker', 'adj close'])
    
        probabilities = prod_model.predict_proba(X_predict)[:, 1]
        candidates = pd.DataFrame({
            'Ticker': predict_df['ticker'],
            'Quant_Score': probabilities
        }).sort_values(by='Quant_Score', ascending=False).reset_index(drop=True)
    
        top_candidates = candidates.head(config.TOP_N_CANDIDATES)
    
        print(f"💾 Saving top {len(top_candidates)} candidates to {config.CANDIDATE_RESULTS_PATH}...")
        top_candidates.to_csv(config.CANDIDATE_RESULTS_PATH, index=False)
    
        return top_candidates, feature_imp_sorted
