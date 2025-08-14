# stock_analyzer/candidate_generator.py

import pandas as pd
import numpy as np
import lightgbm as lgb
import config

class CandidateGenerator:
    """
    Uses a machine learning model to screen and rank stocks based on quantitative factors.
    """
    def _create_features(self, price_df, fundamentals_df, spy_df):
        """
        Engineers features for the model from the raw data.
        """
        print("🚀 Starting feature engineering...")
        
        features_df = price_df.copy()
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
        Defines the target variable based on future relative performance.
        """
        df_copy = df.copy()
        df_copy['future_return'] = df_copy.groupby('Ticker')['Adj Close'].shift(-config.TARGET_FORWARD_PERIOD) / df_copy['Adj Close'] - 1
        df_copy = df_copy.dropna(subset=['future_return'])
        cutoffs = df_copy.groupby(df_copy.index)['future_return'].transform(lambda x: x.quantile(config.TARGET_QUANTILE))
        df_copy['target'] = (df_copy['future_return'] >= cutoffs).astype(int)
        return df_copy.drop(columns=['future_return'])

    def generate_candidates(self, price_df, fundamentals_df, spy_df):
        """
        Main method to run the feature engineering, model training, and candidate prediction.
        """
        features_df = self._create_features(price_df, fundamentals_df, spy_df)
        features_df.sort_index(inplace=True)
        final_df = self._define_target(features_df)

        X = final_df.drop(columns=['target', 'Ticker', 'Adj Close'])
        y = final_df['target']

        print("--- 🏋️ Training Production Model on All Data ---")
        scale_pos_weight = y.value_counts()[0] / y.value_counts()[1]
        prod_model = lgb.LGBMClassifier(objective='binary', scale_pos_weight=scale_pos_weight, random_state=42)
        prod_model.fit(X, y)

        feature_imp = pd.DataFrame(
            sorted(zip(prod_model.feature_importances_, X.columns)),
            columns=['Value','Feature']
        )
        feature_imp_sorted = feature_imp.sort_values(by="Value", ascending=False).head(10)

        print("---  прогнозирование on Most Recent Data ---")
        latest_data = final_df.loc[final_df.index == final_df.index.max()]
        latest_X = latest_data.drop(columns=['target', 'Ticker', 'Adj Close'])

        if latest_X.empty:
            print("⚠️ No recent data available for prediction.")
            return pd.DataFrame(), pd.DataFrame()

        probabilities = prod_model.predict_proba(latest_X)[:, 1]
        candidates = pd.DataFrame({
            'Ticker': latest_data['Ticker'],
            'Quant_Score': probabilities
        }).sort_values(by='Quant_Score', ascending=False).reset_index(drop=True)

        top_candidates = candidates.head(config.TOP_N_CANDIDATES)

        print(f"💾 Saving top {len(top_candidates)} candidates to {config.CANDIDATE_RESULTS_PATH}...")
        top_candidates.to_csv(config.CANDIDATE_RESULTS_PATH, index=False)

        return top_candidates, feature_imp_sorted
