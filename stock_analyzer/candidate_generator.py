# stock_analyzer/candidate_generator.py

import pandas as pd
import numpy as np
import lightgbm as lgb
import config

# at top of file (optional dependency)
try:
    import shap  # pip install shap
    _HAVE_SHAP = True
except Exception:
    _HAVE_SHAP = False

class CandidateGenerator:
    """
    Uses a machine learning model to screen and rank stocks based on quantitative factors.
    This version includes fixes for data leakage and improved data handling.
    """    
    def _importance_and_shap_report(
        self,
        model: lgb.LGBMClassifier,
        X_train: pd.DataFrame,
        shap_sample_size: int = 5000,
        random_state: int = 42,
    ) -> dict:
        """
        Returns a consolidated importance report:
          - gain_importance: DataFrame[Feature, Gain]
          - shap_importance: DataFrame[Feature, MeanAbsSHAP] (None if shap unavailable)
          - combined: DataFrame with both + ranks (where available)
          - suggested_drop: list[str] low-impact features suggested for pruning
        """
        # ---- Gain-based importance (more meaningful than split count) ----
        booster = model.booster_
        gain_vals = booster.feature_importance(importance_type="gain")
        feat_names = booster.feature_name()
        gain_df = (
            pd.DataFrame({"Feature": feat_names, "Gain": gain_vals})
            .sort_values("Gain", ascending=False)
            .reset_index(drop=True)
        )

        # ---- SHAP (optional, sampled to keep it fast) ----
        shap_df = None
        if _HAVE_SHAP:
            # Sample to control cost
            if len(X_train) > shap_sample_size:
                X_for_shap = X_train.sample(shap_sample_size, random_state=random_state)
            else:
                X_for_shap = X_train

            # TreeExplainer is fast/accurate for tree models
            explainer = shap.TreeExplainer(model, feature_names=list(X_train.columns))
            shap_values = explainer.shap_values(X_for_shap)
            # For binary classifier, shap_values is (n_samples, n_features)
            if isinstance(shap_values, list):  # some wrappers return list
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            mean_abs = np.abs(shap_values).mean(axis=0)

            shap_df = (
                pd.DataFrame({"Feature": X_for_shap.columns, "MeanAbsSHAP": mean_abs})
                .sort_values("MeanAbsSHAP", ascending=False)
                .reset_index(drop=True)
            )

        # ---- Combine + ranks ----
        combined = gain_df.copy()
        combined["GainRank"] = combined["Gain"].rank(ascending=False, method="dense")
        if shap_df is not None:
            combined = combined.merge(shap_df, on="Feature", how="left")
            combined["SHAPRank"] = combined["MeanAbsSHAP"].rank(
                ascending=False, method="dense"
            )

        # ---- Suggested prune list (conservative) ----
        # Rule: candidates w/ zero (or near-zero) gain AND tiny mean |SHAP|
        suggested_drop: list[str] = []
        if shap_df is not None:
            # Threshold ~ small fraction of SHAP scale; tweak as needed
            shap_floor = max(shap_df["MeanAbsSHAP"].median() * 0.05,
                             shap_df["MeanAbsSHAP"].max() * 0.01)
            low_shap = set(shap_df.loc[shap_df["MeanAbsSHAP"] <= shap_floor, "Feature"])
            zero_gain = set(gain_df.loc[gain_df["Gain"] <= 0, "Feature"])
            suggested_drop = sorted(low_shap & zero_gain)
        else:
            # If no SHAP, drop only features with zero gain
            suggested_drop = sorted(gain_df.loc[gain_df["Gain"] <= 0, "Feature"])

        return {
            "gain_importance": gain_df,
            "shap_importance": shap_df,    # may be None
            "combined": combined,
            "suggested_drop": suggested_drop,
        }

    def _create_features(self, df, spy_df):
        """
        Engineers features for the model from the raw data.
        """
        print("🚀 Starting feature engineering...")
        features_df = df.copy()

        # --- Momentum and Volatility Features ---
        grouped_by_ticker = features_df.groupby('ticker')
        for lag in [21, 63, 252]:
            features_df[f'return_{lag}d'] = grouped_by_ticker['adj close'].transform(lambda x: x.pct_change(lag))

        log_returns = np.log(grouped_by_ticker['adj close'].transform(lambda x: x / x.shift(1)))
        for lag in [21, 63, 252]:
            features_df[f'volatility_{lag}d'] = log_returns.groupby(features_df['ticker']).transform(lambda x: x.rolling(lag).std())

        # --- Market-Relative Features ---
        spy_df['market_return'] = np.log(spy_df['close'] / spy_df['close'].shift(1))
        features_df = pd.merge(features_df, spy_df[['date', 'market_return']], on='date', how='left')

        # --- Beta Calculation (robust method, no MultiIndex bugs) ---
        def calculate_beta(sub_df, window=63):
            log_return = np.log(sub_df['adj close'] / sub_df['adj close'].shift(1))
            market_var = sub_df['market_return'].rolling(window=window).var()
            covariance = log_return.rolling(window=window).cov(sub_df['market_return'])
            return covariance / market_var

        beta_df = (
            features_df
            .groupby('ticker', group_keys=False)
            .apply(lambda g: pd.DataFrame({
                'date': g['date'],
                'ticker': g['ticker'],
                'beta_63d': calculate_beta(g)
            }))
        )

        features_df = pd.merge(features_df, beta_df[['date', 'ticker', 'beta_63d']], on=['date', 'ticker'], how='left')

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
        df_copy['future_return'] = (
            df_copy.groupby('ticker')['adj close']
            .shift(-config.TARGET_FORWARD_PERIOD) / df_copy['adj close'] - 1
        )

        df_copy = df_copy.dropna(subset=['future_return'])

        daily_cutoffs = df_copy.groupby('date')['future_return'].quantile(config.TARGET_QUANTILE).rename('cutoff')
        df_copy = df_copy.merge(daily_cutoffs, on='date', how='left')
        df_copy['target'] = (df_copy['future_return'] >= df_copy['cutoff']).astype(int)

        return df_copy.drop(columns=['future_return', 'cutoff'])

    def generate_candidates(self, price_df, fundamentals_df, spy_df):
        """
        Main method to run feature engineering, model training, and candidate prediction.
        """
        #label "missing" dividends: a potential feature.
        #for col in ["dividendRate", "dividendYield", "fiveYearAvgDividendYield"]:
        #    fundamentals_df[f"{col}_is_missing"] = fundamentals_df[col].isna().astype(int)

        #convert to lower-case
        price_df.columns = price_df.columns.str.lower()
        fundamentals_df.columns = fundamentals_df.columns.str.lower()
        spy_df.columns = spy_df.columns.str.lower()

        # nuke the timestamp metadata column; it’s not a feature
        fundamentals_df = fundamentals_df.drop(columns=['asof'], errors='ignore')

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
            columns=['Value', 'Feature']
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

        # NEW: consolidated importance + SHAP report
        report = self._importance_and_shap_report(prod_model, X_train)
        
        return top_candidates, feature_imp_sorted, report
        #return top_candidates, feature_imp_sorted
