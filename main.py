# main.py

"""
The main entry point for the automated stock analysis and portfolio management pipeline.
This script orchestrates the entire process from data loading to portfolio adjustment.
"""

# The imports now reference the stock_analyzer package
from stock_analyzer.data_manager import DataManager
from stock_analyzer.candidate_generator import CandidateGenerator
# from stock_analyzer.agentic_layer import AgenticLayer
# from stock_analyzer.portfolio_manager import PortfolioManager

def run_pipeline():
    """
    Executes the full pipeline:
    1. Load Data
    2. Generate Candidates
    3. Perform Agentic Analysis
    4. Update Portfolio
    """
    print("🚀 Starting the main investment pipeline...")

    # --- Step 1: Data Loading ---
    data_manager = DataManager()
    price_df, fundamentals_df, spy_df = data_manager.get_all_data()

    if price_df is None:
        print("❌ Pipeline stopped due to data loading failure.")
        return

    # --- Step 2: Candidate Generation (LightGBM) ---
    candidate_generator = CandidateGenerator()
    top_candidates_df = candidate_generator.generate_candidates(price_df, fundamentals_df, spy_df)

    if top_candidates_df.empty:
        print("❌ Pipeline stopped: No candidates were generated.")
        return

    print("\n🏆 Top Quantitative Candidates:")
    print(top_candidates_df)

    # --- Step 3: Agentic Analysis (The "Heavy" part) ---
    # This section will be enabled once we build the agentic_layer.py
    #
    # agentic_layer = AgenticLayer()
    # ranked_candidates = agentic_layer.run_analysis(top_candidates_df)
    #
    # if ranked_candidates.empty:
    #     print("❌ Pipeline stopped: Agentic analysis yielded no ranked candidates.")
    #     return
    #
    # print("\n🤖 Top Agent-Ranked Candidates:")
    # print(ranked_candidates)

    # --- Step 4: Portfolio Management ---
    # This section will be enabled once we build the portfolio_manager.py
    #
    # portfolio_manager = PortfolioManager()
    # portfolio_manager.update_portfolio(ranked_candidates)

    print("\n✅ Pipeline finished successfully!")


if __name__ == "__main__":
    # This allows the script to be run directly from the command line.
    run_pipeline()
