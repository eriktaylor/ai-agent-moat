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
        
        # --- Set Date as Index ---
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
        # Ensure spy_df has a date index for joining
        if 'date' in spy_df.columns:
            spy_df.set_index('date', inplace=True)
            
        market_returns = np.log(spy_df['close'] / spy_df['close'].shift(1)).rename('market_return')
        
        # --- FIX for Duplicate Labels Error ---
        # The .join() method fails because features_df has a non-unique 'date' index.
        # We switch to .merge() and then set the index back to 'date' so the rest of the function works as expected.
        features_df = features_df.reset_index().merge(market_returns.to_frame(), on='date', how='left').set_index('date')
        # --- END FIX ---

        def rolling_beta(stock_return, market_return, window=63):
            cov = stock_return.rolling(window=window).cov(market_return)
            market_var = market_return.rolling(window=window).var()
            return cov / market_var
            
        # Need to group by ticker to calculate beta correctly for each stock
        beta_series = features_df.groupby('ticker').apply(lambda x: rolling_beta(np.log(x['adj close'] / x['adj close'].shift(1)), x['market_return']))
        # The result from apply might have a multi-index, so we need to align it properly
        features_df['beta_63d'] = beta_series.reset_index(level=0, drop=True)


        # --- Clean Up ---
        # The fundamentals are already merged. We just need to drop raw price columns.
        features_df = features_df.drop(columns=['open', 'high', 'low', 'close', 'volume', 'market_return'])
        
        # Reset index to have 'date' and 'ticker' as columns for the next step
        return features_df.dropna().reset_index()


    def _define_target(self, df):
        """
        Defines the target variable using a robust merge-based approach to avoid index alignment issues.
        """
        df_copy = df.copy()
        df_copy['future_return'] = df_copy.groupby('ticker')['adj close'].shift(-config.TARGET_FORWARD_PERIOD) / df_copy['adj close'] - 1
        df_copy = df_copy.dropna(subset=['future_return'])

        # Calculate daily cutoffs as a separate Series
        daily_cutoffs = df_copy.groupby('date')['future_return'].quantile(config.TARGET_QUANTILE).rename('cutoff')

        # Merge the cutoffs back into the main DataFrame for stable alignment
        df_copy = df_copy.merge(daily_cutoffs, on='date', how='left')

        # Perform the comparison on perfectly aligned columns
        df_copy['target'] = (df_copy['future_return'] >= df_copy['cutoff']).astype(int)

        # Clean up and return the result
        return df_copy.drop(columns=['future_return', 'cutoff'])


    def generate_candidates(self, price_df, fundamentals_df, spy_df):
        """
        Main method to run feature engineering, model training, and candidate prediction.
        """
        # 1. Standardize all column names to lowercase to prevent mismatches.
        price_df.columns = price_df.columns.str.lower()
        fundamentals_df.columns = fundamentals_df.columns.str.lower()
        spy_df.columns = spy_df.columns.str.lower()
        
        # Add ticker to spy_df for consistency
        spy_df['ticker'] = 'SPY'
        
        # Ensure 'date' columns are in datetime format
        price_df['date'] = pd.to_datetime(price_df['date'])
        spy_df['date'] = pd.to_datetime(spy_df['date'])
        
        # 2. Perform a LEFT merge on 'ticker' only. This creates our primary DataFrame.
        df = pd.merge(price_df, fundamentals_df, on='ticker', how='left')
    
        # 3. Call feature creation using the new 'df' as the main input.
        features_df = self._create_features(df, spy_df)
        
        # 4. Define the target variable.
        final_df = self._define_target(features_df)
        
        # Set date as index for time-series based operations (like selecting latest data)
        if 'date' in final_df.columns:
            final_df.set_index('date', inplace=True)

        # 5. Create the feature set (X) and target (y) for the model.
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
