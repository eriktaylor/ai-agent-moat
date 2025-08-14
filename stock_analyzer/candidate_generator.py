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
        #features_df = price_df.copy()
        grouped = features_df.groupby('Ticker')

        # --- Momentum Features ---
        for lag in [5, 21, 63, 252]:
            features_df[f'return_{lag}d'] = grouped['Adj Close'].transform(lambda x: x.pct_change(lag))

        # --- Volatility Features ---
        stock_returns = grouped['Adj Close'].transform(lambda x: np.log(x / x.shift(1)))
        for lag in [21, 63, 252]:
            features_df[f'volatility_{lag}d'] = stock_returns.groupby(features_df['Ticker']).transform(lambda x: x.rolling(lag).std())

        # --- Market-Relative Features ---
        market_returns = np.log(spy_df['Close'] / spy_df['Close'].shift(1)).rename('market_return')
        features_df = features_df.join(market_returns)

        def rolling_beta(stock_return, market_return, window=63):
            cov = stock_return.rolling(window=window).cov(market_return)
            market_var = market_return.rolling(window=window).var()
            return cov / market_var
        features_df['beta_63d'] = rolling_beta(stock_returns, features_df['market_return'])

        # --- Combine with Fundamentals and Clean ---
        features_df = features_df.reset_index().merge(fundamentals_df, on='Ticker').set_index('Date')
        features_df = features_df.drop(columns=['Open', 'High', 'Low', 'Close', 'Volume', 'market_return'])
        return features_df.dropna()

    def _define_target(self, df):
        """
        Defines the target variable using a robust merge-based approach to avoid index alignment issues.
        """
        df_copy = df.copy()
        df_copy['future_return'] = df_copy.groupby('Ticker')['Adj Close'].shift(-config.TARGET_FORWARD_PERIOD) / df_copy['Adj Close'] - 1
        df_copy = df_copy.dropna(subset=['future_return'])

        # Calculate daily cutoffs as a separate Series
        daily_cutoffs = df_copy.groupby('Date')['future_return'].quantile(config.TARGET_QUANTILE).rename('cutoff')

        # Merge the cutoffs back into the main DataFrame for stable alignment
        df_copy = df_copy.merge(daily_cutoffs, on='Date', how='left')

        # Perform the comparison on perfectly aligned columns
        df_copy['target'] = (df_copy['future_return'] >= df_copy['cutoff']).astype(int)

        # Clean up and return the result
        return df_copy.drop(columns=['future_return', 'cutoff'])


    # In stock_analyzer/candidate_generator.py
    def generate_candidates(self, price_df, fundamentals_df, spy_df):
        """
        Main method to run feature engineering, model training, and candidate prediction.
        """
        # --- START: FIX ---
        # 1. Standardize all column names to lowercase to prevent mismatches.
        price_df.columns = price_df.columns.str.lower()
        fundamentals_df.columns = fundamentals_df.columns.str.lower()
        spy_df.columns = spy_df.columns.str.lower()
        
        # Ensure 'date' columns are in datetime format
        price_df['date'] = pd.to_datetime(price_df['date'])
        spy_df['date'] = pd.to_datetime(spy_df['date'])
        
        # 2. Perform a LEFT merge on 'ticker' only. This creates our primary DataFrame.
        df = pd.merge(price_df, fundamentals_df, on='ticker', how='left')
        # --- END: FIX ---
    
        # --- START: MODIFIED LOGIC ---
        # 3. Call feature creation using the new 'df' as the main input.
        # We pass 'df' instead of 'price_df' and no longer need 'fundamentals_df' here.
        features_df = self._create_features(df, spy_df)
        
        # 4. Sort index and define the target variable. This part remains the same.
        features_df.sort_index(inplace=True)
        final_df = self._define_target(features_df)
    
        # 5. Create the feature set (X) and target (y) for the model.
        #    Use the lowercased column names ('ticker', 'adj close').
        X = final_df.drop(columns=['target', 'ticker', 'adj close'])
        y = final_df['target']
    
        print("--- 🏋️ Training Production Model on All Data ---")
        
        # Scale_pos_weight logic remains the same.
        counts = y.value_counts()
        if 0 in counts and 1 in counts:
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
    
        print("---  прогнозирование on Most Recent Data ---")
        latest_data = final_df.loc[final_df.index == final_df.index.max()]
        
        # Use lowercased column names here as well.
        latest_X = latest_data.drop(columns=['target', 'ticker', 'adj close'])
    
        if latest_X.empty:
            print("⚠️ No recent data available for prediction.")
            return pd.DataFrame(), pd.DataFrame()
    
        probabilities = prod_model.predict_proba(latest_X)[:, 1]
        candidates = pd.DataFrame({
            'Ticker': latest_data['ticker'], # Use lowercase 'ticker'
            'Quant_Score': probabilities
        }).sort_values(by='Quant_Score', ascending=False).reset_index(drop=True)
    
        top_candidates = candidates.head(config.TOP_N_CANDIDATES)
    
        print(f"💾 Saving top {len(top_candidates)} candidates to {config.CANDIDATE_RESULTS_PATH}...")
        top_candidates.to_csv(config.CANDIDATE_RESULTS_PATH, index=False)
    
        return top_candidates, feature_imp_sorted
