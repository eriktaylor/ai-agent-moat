# stock_analyzer/portfolio_manager.py

import pandas as pd
from datetime import datetime

class PortfolioManager:
    """
    Analyzes quantitative and agentic results to construct and manage a
    target portfolio based on a dynamic market thesis.
    """
    def _formulate_market_thesis(self, agentic_df: pd.DataFrame) -> tuple[str, str]:
        """
        Analyzes the distribution of agentic ratings to form a market thesis.

        Returns:
            A tuple containing the thesis string and a sentiment ('Aggressive', 'Neutral', 'Defensive').
        """
        if agentic_df.empty or 'Agent_Rating' not in agentic_df.columns:
            return "No agentic recommendations available to form a thesis.", "Neutral"

        buy_ratings = agentic_df[agentic_df['Agent_Rating'] > 0.7].shape[0]
        hold_ratings = agentic_df[(agentic_df['Agent_Rating'] >= 0.5) & (agentic_df['Agent_Rating'] <= 0.7)].shape[0]
        total_analyzed = len(agentic_df)
        
        buy_percentage = (buy_ratings / total_analyzed) * 100 if total_analyzed > 0 else 0

        if buy_percentage > 30:
            sentiment = "Aggressive"
            thesis = f"Market Thesis ({sentiment}): A significant number of buy signals ({buy_ratings}/{total_analyzed}) suggest favorable conditions. The portfolio will be constructed aggressively."
        elif buy_percentage > 10:
            sentiment = "Neutral"
            thesis = f"Market Thesis ({sentiment}): A moderate number of buy signals ({buy_ratings}/{total_analyzed}) suggest a mixed market. The portfolio will be constructed selectively."
        else:
            sentiment = "Defensive"
            thesis = f"Market Thesis ({sentiment}): Very few buy signals ({buy_ratings}/{total_analyzed}) were found, indicating caution is warranted. The portfolio will be highly selective and defensive."
            
        return thesis, sentiment

    def generate_portfolio(self, quant_df: pd.DataFrame, agentic_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates a new target portfolio based on the latest analysis.
        """
        print("--- 🤖 Portfolio Manager Activated ---")

        # Step 1: Form the market thesis
        thesis, sentiment = self._formulate_market_thesis(agentic_df)
        print(thesis)

        # Merge the two dataframes to have all info in one place
        merged_df = pd.merge(agentic_df, quant_df[['Ticker', 'Quant_Score']], on='Ticker', how='left')

        # Step 2: Apply rules based on the thesis to construct the portfolio
        portfolio_stocks = []
        if sentiment == "Aggressive":
            # Select all stocks with a 'Buy' or 'Strong Buy' rating
            portfolio_stocks = merged_df[merged_df['Agent_Rating'] > 0.7]
        elif sentiment == "Neutral":
            # Select only stocks with a stronger 'Buy' rating
            portfolio_stocks = merged_df[merged_df['Agent_Rating'] > 0.75]
        elif sentiment == "Defensive":
            # Be very selective, only 'Strong Buy'
            portfolio_stocks = merged_df[merged_df['Agent_Rating'] > 0.8]

        if portfolio_stocks.empty:
            print("No stocks met the criteria for the current thesis. The portfolio will be empty.")
            return pd.DataFrame()

        # Create the final portfolio DataFrame
        final_portfolio = portfolio_stocks[['Ticker', 'Quant_Score', 'Agent_Rating', 'Justification']].copy()
        final_portfolio.rename(columns={
            'Quant_Score': 'Quant_Score_at_Entry',
            'Agent_Rating': 'Agent_Rating_at_Entry',
            'Justification': 'Rationale'
        }, inplace=True)
        
        final_portfolio['Entry_Date'] = datetime.now().strftime('%Y-%m-%d')
        final_portfolio['Thesis_at_Entry'] = sentiment

        # Reorder columns for clarity
        final_portfolio = final_portfolio[[
            'Ticker', 'Entry_Date', 'Quant_Score_at_Entry', 
            'Agent_Rating_at_Entry', 'Thesis_at_Entry', 'Rationale'
        ]]
        
        print(f"✅ Portfolio constructed with {len(final_portfolio)} positions based on the '{sentiment}' thesis.")
        
        return final_portfolio
