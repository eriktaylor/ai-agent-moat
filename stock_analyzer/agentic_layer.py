# stock_analyzer/agentic_layer.py
import os
import re
import json
import logging
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_core.prompts import ChatPromptTemplate

import config

# ------------------------
# Logging Setup
# ------------------------
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO
)

class AgenticLayer:
    """
    Orchestrates multi-agent qualitative analysis on stock candidates.
    1. Scout Agent: Finds new, non-S&P 500 stocks.
    2. Analyst Agent: Performs deep-dive analysis using three personas.
    3. Ranking Judge Agent: Synthesizes analysis into a final rating and recommendation.
    """

    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
        self.search_wrapper = GoogleSearchAPIWrapper(
            google_cse_id=os.environ.get('GOOGLE_CSE_ID'),
            google_api_key=os.environ.get('GOOGLE_API_KEY')
        )
        self.analysis_cache = {}

    # ------------------------
    # Scout Agent
    # ------------------------
    def _run_scout_agent(self, known_tickers):
        logging.info("Running Scout Agent to find new tickers...")
        new_tickers = set()

        queries = [
            "top performing small cap stocks 2025",
            "undervalued growth stocks outside S&P 500",
            "companies with recent technological breakthroughs stock"
        ]

        for query in queries:
            if len(new_tickers) >= config.MAX_SCOUT_RESULTS:
                logging.info("Scout limit reached. Halting search.")
                break

            logging.info(f"Scouting with query: '{query}'")
            try:
                search_results = self.search_wrapper.results(query, num_results=5)
                snippets = " ".join(res.get('snippet', '') for res in search_results if 'snippet' in res)
            except Exception as e:
                logging.error(f"Search API error: {e}")
                continue

            prompt = ChatPromptTemplate.from_template(
                "Extract all valid stock ticker symbols (e.g., AAPL, MSFT) from the following text. "
                "List them separated by commas. If none are found, respond with 'None'.\n\nText: {context}"
            )
            try:
                response = (prompt | self.llm).invoke({"context": snippets})
                found_tickers = [
                    t.strip().upper()
                    for t in response.content.split(',')
                    if t.strip() and t.strip().lower() != 'none'
                ]
            except Exception as e:
                logging.error(f"Ticker extraction LLM error: {e}")
                continue

            for ticker in found_tickers:
                if ticker not in known_tickers:
                    new_tickers.add(ticker)
                    if len(new_tickers) >= config.MAX_SCOUT_RESULTS:
                        break
        
        # Validate each found ticker to ensure it's real
        validated_tickers = []
        for ticker in new_tickers:
            try:
                # A quick check to see if the ticker has a valid market cap or short name
                stock_info = yf.Ticker(ticker).info
                if stock_info.get('marketCap') or stock_info.get('shortName'):
                    validated_tickers.append(ticker)
                    logging.info(f"✅ Validated scouted ticker: {ticker}")
                else:
                    logging.warning(f"❌ Discarding invalid/delisted ticker: {ticker}")
                if len(validated_tickers) >= config.MAX_SCOUT_RESULTS:
                    break
            except Exception:
                logging.warning(f"❌ Discarding invalid ticker after yfinance error: {ticker}")
                continue

        final_list = validated_tickers
        logging.info(f"Scout found {len(final_list)} new tickers: {final_list}")
        return final_list

    # ------------------------
    # Analyst Agent
    # ------------------------
    def _run_analyst_agent(self, ticker):
        if ticker in self.analysis_cache:
            logging.info(f"Using cached analysis for {ticker}")
            return self.analysis_cache[ticker]

        logging.info(f"Running Analyst Agent on {ticker}")
        try:
            stock_info = yf.Ticker(ticker).info
        except Exception as e:
            logging.error(f"Yahoo Finance API error for {ticker}: {e}")
            return {"error": str(e)}

        company_name = stock_info.get("longName", ticker)
        financial_data = {
            "longName": company_name,
            "sector": stock_info.get("sector", "N/A"),
            "trailingPE": stock_info.get("trailingPE", "N/A"),
            "forwardPE": stock_info.get("forwardPE", "N/A"),
            "marketCap": stock_info.get("marketCap", "N/A"),
            "fiftyTwoWeekHigh": stock_info.get("fiftyTwoWeekHigh", "N/A"),
            "fiftyTwoWeekLow": stock_info.get("fiftyTwoWeekLow", "N/A"),
        }

        # News gathering
        news_queries = {
            "Professional & Financial Analysis": f'"{company_name}" ({ticker}) stock analysis site:reuters.com OR site:bloomberg.com OR site:wsj.com',
            "Retail & Social Sentiment": f'"{company_name}" ({ticker}) stock sentiment site:reddit.com OR site:fool.com OR site:seekingalpha.com',
            "Risk Factors & Negative News": f'"{company_name}" ({ticker}) risk OR lawsuit OR investigation OR recall OR safety OR short interest'
        }

        news_context = ""
        for category, query in news_queries.items():
            news_context += f"\n--- {category} ---\n"
            try:
                num_results = 4 if category == "Professional & Financial Analysis" else 2
                results = self.search_wrapper.results(query, num_results=num_results)
                if results:
                    for r in results:
                        news_context += f"**{r.get('title', 'No Title')}**: {r.get('snippet', '')}\n"
                else:
                    news_context += "No recent results found.\n"
            except Exception as e:
                logging.error(f"News search error for {ticker} ({category}): {e}")
                news_context += "Error fetching news.\n"

        # Persona analysis
        personas = ["Market Investor", "Value Investor", "Devil's Advocate"]
        system_prompt = (
            "You are an expert financial analyst writing a report from the perspective of a {persona}. "
            "Reference a specific point from the News Context first, then connect it to the Financial Data, "
            "and end with a clear conclusion.\n\n"
            "--- DATA AS OF {date} ---\n"
            "**Financial Data:**\n{financial_data}\n\n"
            "**News Context:**\n{news_context}\n\n"
            "--- {persona} Analysis ---"
        )
        prompt = ChatPromptTemplate.from_template(system_prompt)
        chain = prompt | self.llm

        reports = {}
        for persona in personas:
            try:
                response = chain.invoke({
                    "persona": persona,
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "financial_data": json.dumps(financial_data),
                    "news_context": news_context
                })
                reports[persona] = response.content
            except Exception as e:
                logging.error(f"Persona analysis error for {ticker} ({persona}): {e}")
                reports[persona] = "Analysis failed."

        self.analysis_cache[ticker] = reports
        return reports

    # ------------------------
    # Ranking Judge Agent
    # ------------------------
    def _run_ranking_judge_agent(self, reports, ticker):
        logging.info(f"Running Ranking Judge Agent on {ticker}")
        if "error" in reports:
            return {"rating": 0.0, "recommendation": "Neutral", "justification": "Analysis error."}

        if not all(p in reports for p in ["Market Investor", "Value Investor", "Devil's Advocate"]):
            return {"rating": 0.0, "recommendation": "Neutral", "justification": "Missing persona analysis."}

        system_prompt = (
            "You are a senior portfolio manager. Based on the three analyst reports, output ONLY a JSON object with double-quoted keys:\n"
            '{"rating": float (0.0-1.0), "recommendation": "Buy"|"Sell"|"Hold"|"Neutral", "justification": "string"}.\n'
            "No extra text or formatting.\n\n"
            "Market Investor:\n{market_report}\n\n"
            "Value Investor:\n{value_report}\n\n"
            "Devil's Advocate:\n{devils_report}"
        )

        try:
            response = (ChatPromptTemplate.from_template(system_prompt) | self.llm).invoke({
                "market_report": reports["Market Investor"],
                "value_report": reports["Value Investor"],
                "devils_report": reports["Devil's Advocate"]
            })
            # --- FIX: Use non-greedy regex for safer JSON extraction ---
            clean_text = re.sub(r"```(?:json)?", "", response.content).strip()
            json_match = re.search(r"\{.*?\}", clean_text, re.DOTALL)
            return json.loads(json_match.group(0)) if json_match else {"rating": 0.0, "recommendation": "Neutral", "justification": "Invalid JSON."}
        except Exception as e:
            logging.error(f"Judge agent error for {ticker}: {e}")
            return {"rating": 0.0, "recommendation": "Neutral", "justification": "Failed to parse decision."}

    # ------------------------
    # Main pipeline
    # ------------------------
    def run_analysis(self):
        logging.info("Starting Agentic Analysis Layer...")

        try:
            quant_df = pd.read_csv(config.CANDIDATE_RESULTS_PATH)
            known_tickers = set(quant_df['Ticker'])
        except FileNotFoundError:
            logging.error(f"Quantitative candidates file missing: {config.CANDIDATE_RESULTS_PATH}")
            return pd.DataFrame()

        try:
            prev_df = pd.read_csv(config.AGENTIC_RESULTS_PATH)
            prev_df['Analysis_Date'] = pd.to_datetime(prev_df['Analysis_Date'])
            known_tickers.update(prev_df['Ticker'])
        except FileNotFoundError:
            logging.info("No previous agentic recommendations found.")
            prev_df = pd.DataFrame()

        new_tickers = self._run_scout_agent(known_tickers)
        top_quant_candidates = quant_df.head(config.QUANT_DEEP_DIVE_CANDIDATES)['Ticker'].tolist()
        
        tickers_to_analyze = list(dict.fromkeys(top_quant_candidates + new_tickers))

        logging.info(f"Analyzing {len(tickers_to_analyze)} unique tickers: {tickers_to_analyze}")
        # --- FIX: Use pandas Timestamp for reliable date operations ---
        today = pd.Timestamp(datetime.now()).normalize()
        results = []

        for ticker in tickers_to_analyze:
            # Use recent analysis if fresh
            if not prev_df.empty and ticker in prev_df['Ticker'].values:
                # Safely get the most recent analysis for a ticker
                last_analysis_row = prev_df[prev_df['Ticker'] == ticker].sort_values(by='Analysis_Date', ascending=False).iloc[0]
                last_date = pd.to_datetime(last_analysis_row['Analysis_Date']).normalize()
                if (today - last_date) < timedelta(days=5):
                    logging.info(f"Using recent analysis for {ticker} from {last_date.date()}")
                    results.append(last_analysis_row.to_dict())
                    continue

            reports = self._run_analyst_agent(ticker)
            judgment = self._run_ranking_judge_agent(reports, ticker)

            quant_score_series = quant_df.loc[quant_df['Ticker'] == ticker, 'Quant_Score']
            quant_score = quant_score_series.iloc[0] if not quant_score_series.empty else 'N/A'

            results.append({
                'Ticker': ticker,
                'Quant_Score': quant_score,
                'Analysis_Date': today.strftime('%Y-%m-%d'),
                'Agent_Rating': judgment.get('rating'),
                'Agent_Recommendation': judgment.get('recommendation'),
                'Justification': judgment.get('justification'),
                'Market_Investor_Analysis': reports.get('Market Investor', 'N/A'),
                'Value_Investor_Analysis': reports.get('Value Investor', 'N/A'),
                'Devils_Advocate_Analysis': reports.get("Devil's Advocate", 'N/A')
            })

        results_df = pd.DataFrame(results)
        # --- FIX: Safely sort by converting rating to numeric ---
        results_df['Agent_Rating'] = pd.to_numeric(results_df['Agent_Rating'], errors='coerce')
        results_df.sort_values(by="Agent_Rating", ascending=False, inplace=True)
        results_df.reset_index(drop=True, inplace=True)
        
        results_df.to_csv(config.AGENTIC_RESULTS_PATH, index=False)
        logging.info(f"Analysis complete. Saved to {config.AGENTIC_RESULTS_PATH}")

        return results_df
