# stock_analyzer/agentic_layer.py

import os
import json
import pandas as pd
import yfinance as yf
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_core.prompts import ChatPromptTemplate
import config

class AgenticLayer:
    """
    Orchestrates a series of AI agents to perform qualitative analysis on stock candidates.
    1. Scout Agent: Finds new, non-S&P 500 stocks.
    2. Analyst Agent: Performs deep-dive analysis using three personas.
    3. Ranking Judge Agent: Synthesizes analysis into a final rating and recommendation.
    """
    def __init__(self):
        """Initializes the models, tools, and cache."""
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
        self.search_wrapper = GoogleSearchAPIWrapper(
            google_cse_id=os.environ.get('GOOGLE_CSE_ID'),
            google_api_key=os.environ.get('GOOGLE_API_KEY')
        )
        self.analysis_cache = {} # Cache for storing analysis to avoid redundant work

    def _run_scout_agent(self, known_tickers):
        """
        Performs broad web searches to find promising stocks not in the known list.
        """
        print("--- 🕵️ Running Scout Agent to find new tickers ---")
        new_tickers = set()
        # Perform a few generic searches for new ideas
        queries = [
            "top performing small cap stocks 2025",
            "undervalued growth stocks outside S&P 500",
            "companies with recent technological breakthroughs stock"
        ]
        for query in queries:
            # If we've already hit our limit, no need to make more API calls.
            if len(new_tickers) >= config.MAX_SCOUT_RESULTS:
                print("Scout limit reached. Halting search.")
                break
                
            print(f"Scouting with query: '{query}'")
            search_results = self.search_wrapper.results(query, num_results=5)
            context = " ".join([res.get('snippet', '') for res in search_results])

            # Use LLM to extract tickers from search results
            prompt = ChatPromptTemplate.from_template(
                "Extract all valid stock ticker symbols (e.g., AAPL, MSFT) from the following text. "
                "List them separated by commas. If none are found, respond with 'None'.\n\nText: {context}"
            )
            chain = prompt | self.llm
            response = chain.invoke({"context": context})
            found_tickers = [t.strip() for t in response.content.split(',') if t.strip() and t.strip().lower() != 'none']
            for ticker in found_tickers:
                if ticker not in known_tickers:
                    new_tickers.add(ticker)
                    if len(new_tickers) >= config.MAX_SCOUT_RESULTS:
                        break

        # Enforce the exact limit on the final list before returning
        final_scouted_list = list(new_tickers)[:config.MAX_SCOUT_RESULTS]
        print(f"✅ Scout found {len(final_scouted_list)} new potential tickers: {final_scouted_list}")
        return final_scouted_list

    def _run_analyst_agent(self, ticker):
        """
        Performs a deep-dive analysis on a single ticker, caching the results.
        """
        if ticker in self.analysis_cache:
            print(f"--- 🧠 Using cached analysis for {ticker} ---")
            return self.analysis_cache[ticker]

        print(f"--- 👨‍⚖️ Running Analyst Agent on {ticker} ---")
        try:
            # 1. Gather Financial Data
            stock_info = yf.Ticker(ticker).info
            financial_data = {
                "longName": stock_info.get("longName", "N/A"),
                "sector": stock_info.get("sector", "N/A"),
                "trailingPE": stock_info.get("trailingPE", "N/A"),
                "forwardPE": stock_info.get("forwardPE", "N/A"),
                "marketCap": stock_info.get("marketCap", "N/A"),
                "fiftyTwoWeekHigh": stock_info.get("fiftyTwoWeekHigh", "N/A"),
                "fiftyTwoWeekLow": stock_info.get("fiftyTwoWeekLow", "N/A"),
            }

            # 2. Gather Real-Time News Context
            news_query = f'"{financial_data["longName"]}" ({ticker}) stock news analysis OR concerns OR outlook'
            search_results = self.search_wrapper.results(news_query, num_results=7)
            news_context = "\n---\n".join([f"Title: {r.get('title', '')}\nSnippet: {r.get('snippet', '')}" for r in search_results])

            # 3. Generate Multi-Persona Analysis
            system_prompt = (
                "You are an expert financial analyst. Based on the provided financial data and recent news context, "
                "provide a concise analysis from the perspective of a {persona}.\n\n"
                "Current Date: {date}\nFinancial Data: {financial_data}\n\nNews Context:\n{news_context}\n\n"
                "Your {persona} analysis:"
            )
            prompt = ChatPromptTemplate.from_template(system_prompt)
            chain = prompt | self.llm

            personas = ["Market Investor", "Value Investor", "Devil's Advocate"]
            reports = {}
            for persona in personas:
                response = chain.invoke({
                    "persona": persona,
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "financial_data": json.dumps(financial_data),
                    "news_context": news_context
                })
                reports[persona] = response.content
            
            self.analysis_cache[ticker] = reports # Cache the successful result
            return reports
        except Exception as e:
            print(f"❌ Error analyzing {ticker}: {e}")
            return {"error": str(e)}

    def _run_ranking_judge_agent(self, reports, ticker):
        """
        Synthesizes analyst reports into a final rating and recommendation.
        """
        print(f"--- ⚖️ Running Ranking Judge Agent on {ticker} ---")
        if "error" in reports or not all(p in reports for p in ["Market Investor", "Value Investor", "Devil's Advocate"]):
            return {"rating": 0.0, "recommendation": "Neutral", "justification": "Insufficient information due to analysis error."}

        system_prompt = (
            "You are a Ranking Judge, a senior portfolio manager. You have received three reports from your analyst team. "
            "Your task is to synthesize these reports and provide a final, decisive recommendation. "
            "Your response MUST be a single, valid JSON object with three keys:\n"
            "- 'rating': A float score from 0.0 (strong sell) to 1.0 (strong buy).\n"
            "- 'recommendation': A single string, one of 'Buy', 'Sell', 'Hold', or 'Neutral'.\n"
            "- 'justification': A one-sentence summary of your reasoning, weighing the different views.\n\n"
            "Reports:\n"
            "1. Market Investor (focuses on momentum and news):\n{market_report}\n\n"
            "2. Value Investor (focuses on fundamentals and moat):\n{value_report}\n\n"
            "3. Devil's Advocate (focuses on risks and counterarguments):\n{devils_report}\n\n"
            "JSON Response:"
        )
        prompt = ChatPromptTemplate.from_template(system_prompt)
        chain = prompt | self.llm
        
        response = chain.invoke({
            "market_report": reports["Market Investor"],
            "value_report": reports["Value Investor"],
            "devils_report": reports["Devil's Advocate"],
        })

        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            return {"rating": 0.0, "recommendation": "Neutral", "justification": "Failed to parse judge's final decision."}


    def run_analysis(self):
        """
        The main entry point to run the entire agentic analysis pipeline.
        """
        print("\n--- 🎬 Starting Agentic Analysis Layer ---")
        try:
            candidates_df = pd.read_csv(config.CANDIDATE_RESULTS_PATH)
            known_tickers = set(candidates_df['Ticker'])
        except FileNotFoundError:
            print(f"❌ Error: Quantitative candidates file not found at {config.CANDIDATE_RESULTS_PATH}")
            return pd.DataFrame()

        # Step 1: Scout for new tickers
        new_tickers = self._run_scout_agent(known_tickers)

        # Take the top N candidates from the quantitative list
        top_quant_candidates = candidates_df.head(config.QUANT_DEEP_DIVE_CANDIDATES)['Ticker'].tolist()
        print(f"Taking top {len(top_quant_candidates)} quantitative candidates for deep-dive.")

        # Combine with the scouted tickers to form the final analysis list
        tickers_to_analyze = top_quant_candidates + new_tickers
        print(f"Current tickers for analysis: {len(tickers_to_analyze)} ({tickers_to_analyze})")
                
        final_results = []
        for ticker in tickers_to_analyze:
            # Step 2: Run multi-persona analysis
            reports = self._run_analyst_agent(ticker)
            
            # Step 3: Get final judgment
            judgment = self._run_ranking_judge_agent(reports, ticker)
            
            # Combine all information
            quant_score = candidates_df[candidates_df['Ticker'] == ticker]['Quant_Score'].iloc[0] if ticker in known_tickers else 'N/A'
            final_results.append({
                'Ticker': ticker,
                'Quant_Score': quant_score,
                'Agent_Rating': judgment.get('rating'),
                'Agent_Recommendation': judgment.get('recommendation'),
                'Justification': judgment.get('justification')
            })

        # Create and save the final DataFrame
        results_df = pd.DataFrame(final_results).sort_values(by="Agent_Rating", ascending=False).reset_index(drop=True)
        print(f"💾 Saving agentic analysis results to {config.AGENTIC_RESULTS_PATH}...")
        results_df.to_csv(config.AGENTIC_RESULTS_PATH, index=False)
        
        print("✅ Agentic analysis layer completed successfully.")
        return results_df
