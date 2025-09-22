# stock_analyzer/portfolio_manager.py

import os
import re
import pandas as pd
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI

try:
    # Prefer the newer, non-deprecated package if available
    from langchain_google_community import GoogleSearchAPIWrapper
except ImportError:
    from langchain_community.utilities import GoogleSearchAPIWrapper

class PortfolioManager:
    """
    Acts as a reasoning agent to synthesize market data and internal analysis,
    form a strategic thesis, and construct a weighted target portfolio.
    """
    def __init__(self):
        """Initializes the agent with an LLM and a search tool."""
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
        
        google_api_key = os.environ.get('GOOGLE_API_KEY')
        google_cse_id = os.environ.get('GOOGLE_CSE_ID')
        
        if not google_api_key or not google_cse_id:
            raise ValueError("GOOGLE_API_KEY and GOOGLE_CSE_ID must be set in the environment.")
            
        self.search_wrapper = GoogleSearchAPIWrapper(
            google_api_key=google_api_key,
            google_cse_id=google_cse_id
        )

    def _get_macro_context(self) -> str:
        """Performs targeted web searches to understand the current market environment."""
        print("--- 🔍 Gathering macroeconomic context...")
        queries = [
            "current US stock market sentiment analysis",
            "US federal reserve interest rate outlook 2025",
            "top performing S&P 500 sectors last 3 months"
        ]
        all_results = []
        for query in queries:
            try:
                results = self.search_wrapper.results(query, num_results=2)
                if results:
                    formatted_results = "\n".join([f"- {res.get('title', '')}: {res.get('snippet', '')}" for res in results])
                    all_results.append(f"Search results for '{query}':\n{formatted_results}")
            except Exception as e:
                print(f"Warning: Search for '{query}' failed. {e}")

        context = "\n\n".join(all_results)
        print("--- ✅ Macro context gathered.")
        return context

    def _generate_llm_thesis(self, macro_context: str, internal_signal: str) -> str:
        """Uses the LLM to synthesize external and internal data into a strategic thesis."""
        print("--- 🧠 Formulating investment thesis with LLM...")
        prompt = f"""
        As a Chief Investment Strategist for a quantitative hedge fund, your task is to synthesize external market data with our internal analyst signals to formulate a clear, concise investment thesis for the upcoming week.

        **External Macroeconomic Context:**
        {macro_context}

        **Internal Analyst Signal:**
        {internal_signal}

        Based on a synthesis of all the above information, generate a 2-3 sentence investment thesis. The thesis should state the overall market posture (e.g., Aggressive, Defensive, Cautiously Optimistic) and the primary reasoning.
        """
        response = self.llm.invoke(prompt)
        thesis = response.content
        print(f"--- ✅ Thesis Generated: {thesis}")
        return thesis

    def _construct_llm_portfolio(self, thesis: str, candidates_df: pd.DataFrame) -> dict:
        """Uses the LLM to construct a portfolio based on the thesis and approved candidates."""
        print("--- ⚖️ Constructing portfolio with LLM...")
        
        candidate_list = []
        for _, row in candidates_df.iterrows():
            candidate_list.append(f"- Ticker: {row['Ticker']}, Rationale: {row['Justification']}")
        
        candidates_str = "\n".join(candidate_list)

        prompt = f"""
        You are a Portfolio Manager operating under a specific investment thesis. Your task is to construct a portfolio from a list of analyst-approved candidates.

        **Your Guiding Thesis:**
        "{thesis}"

        **Approved Investment Candidates:**
        {candidates_str}

        **Instructions:**
        1. Based on the thesis, decide which of these stocks, if any, should be included in the portfolio.
        2. Assign a weight (e.g., 4.5%) to each selected stock.
        3. If the thesis is defensive, the total portfolio allocation should be small (e.g., under 15%). If aggressive, it can be larger.
        4. Provide your output ONLY as a comma-separated list of key-value pairs. Example: 'UNH: 5.0%, SMCI: 3.5%'
        """
        response = self.llm.invoke(prompt)
        allocations_str = response.content.strip()

        # Parse the LLM's string output into a dictionary
        allocations = {}
        try:
            pairs = re.findall(r'([A-Z]+):\s*([\d.]+)%', allocations_str)
            for ticker, weight in pairs:
                allocations[ticker] = float(weight)
        except Exception as e:
            print(f"Warning: Could not parse LLM allocation output. Error: {e}")
            return {}
        
        print(f"--- ✅ Portfolio construction complete. Allocations: {allocations}")
        return allocations

    def generate_portfolio(self, quant_df: pd.DataFrame, agentic_df: pd.DataFrame) -> tuple[str, pd.DataFrame]:
        """The main orchestration method for the Portfolio Manager agent."""
        print("\n--- 🤖 Portfolio Manager Activated ---")

        # Create the internal signal summary
        buy_ratings_df = agentic_df[agentic_df['Agent_Rating'] >= 0.7]
        internal_signal = f"My team of analysts has reviewed {len(agentic_df)} high-potential stocks and found {len(buy_ratings_df)} with a 'Buy' or 'Strong Buy' rating."
        print(internal_signal)

        # Step 1: Get macro context
        macro_context = self._get_macro_context()
        
        # Step 2: Generate the thesis
        thesis = self._generate_llm_thesis(macro_context, internal_signal)

        # Step 3: Construct the portfolio
        if buy_ratings_df.empty:
            print("\nNo 'Buy' rated candidates available. The portfolio will be empty.")
            return pd.DataFrame()

        allocations = self._construct_llm_portfolio(thesis, buy_ratings_df)

        if not allocations:
            print("\nThe LLM decided not to allocate to any candidates based on the current thesis.")
            return pd.DataFrame()

        # Step 4: Build the final DataFrame
        portfolio_list = []
        for ticker, weight in allocations.items():
            candidate_info = agentic_df[agentic_df['Ticker'] == ticker].iloc[0]
            quant_info = quant_df[quant_df['Ticker'] == ticker].iloc[0]
            
            portfolio_list.append({
                'Ticker': ticker,
                'Weight_%': weight,
                'Entry_Date': datetime.now().strftime('%Y-%m-%d'),
                'Quant_Score_at_Entry': quant_info['Quant_Score'],
                'Agent_Rating_at_Entry': candidate_info['Agent_Rating'],
                'Thesis_at_Entry': thesis,
                'Rationale': candidate_info['Justification']
            })
            
        final_portfolio = pd.DataFrame(portfolio_list)
        return thesis, final_portfolio
