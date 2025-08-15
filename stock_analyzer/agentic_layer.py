# stock_analyzer/agentic_layer.py
import os
import re
import json
import logging
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf
from langchain_google_genai import ChatGoogleGenerativeAI
try:
    # Prefer the community package (old import is deprecated)
    from langchain_google_community import GoogleSearchAPIWrapper
except Exception:
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

# Simple, conservative US-ticker regex (allows e.g., BRK.B)
TICKER_REGEX = re.compile(r"\b[A-Z]{1,5}(?:\.[A-Z])?\b")

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
            if len(new_tickers) >= getattr(config, "MAX_SCOUT_RESULTS", 10):
                logging.info("Scout limit reached. Halting search.")
                break

            logging.info(f"Scouting with query: '{query}'")
            try:
                search_results = self.search_wrapper.results(query, num_results=5)
                snippets = " ".join(res.get('snippet', '') for res in search_results if 'snippet' in res)
            except Exception as e:
                logging.error(f"Search API error: {e}")
                continue

            # Ask LLM to propose tickers (optional signal)...
            try:
                prompt = ChatPromptTemplate.from_template(
                    "Extract stock tickers (e.g., AAPL, MSFT) mentioned in the text. "
                    "Return a comma-separated list or 'None'.\n\nText:\n{context}"
                )
                llm_response = (prompt | self.llm).invoke({"context": snippets}).content
            except Exception as e:
                logging.error(f"Ticker extraction LLM error: {e}")
                llm_response = ""

            # ...but enforce with regex so we don't rely solely on LLM formatting
            candidates = set(TICKER_REGEX.findall(snippets + " " + llm_response))
            for t in sorted(candidates):
                if t not in known_tickers:
                    new_tickers.add(t)
                    if len(new_tickers) >= getattr(config, "MAX_SCOUT_RESULTS", 10):
                        break

        # Validate each found ticker to ensure it's real enough
        validated = []
        for t in new_tickers:
            try:
                info = yf.Ticker(t).info or {}
                if info.get('marketCap') or info.get('shortName') or info.get('longName'):
                    validated.append(t)
                    logging.info(f"✅ Validated scouted ticker: {t}")
                else:
                    logging.warning(f"❌ Discarding invalid/delisted ticker: {t}")
                if len(validated) >= getattr(config, "MAX_SCOUT_RESULTS", 10):
                    break
            except Exception:
                logging.warning(f"❌ Discarding ticker after yfinance error: {t}")
                continue

        logging.info(f"Scout found {len(validated)} new tickers: {validated}")
        return validated

    # ------------------------
    # Analyst Agent
    # ------------------------
    def _run_analyst_agent(self, ticker):
        if ticker in self.analysis_cache:
            logging.info(f"Using cached analysis for {ticker}")
            return self.analysis_cache[ticker]

        logging.info(f"Running Analyst Agent on {ticker}")
        try:
            stock_info = yf.Ticker(ticker).info or {}
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

        # News gathering (keep site: filters plain, no markdown)
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
            "You are an expert financial analyst writing from the perspective of a {persona}. "
            "Start by referencing one specific item from the News Context, connect it to the Financial Data, "
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

        # IMPORTANT: No raw braces in the template (avoids LangChain "missing variables" error)
        system_prompt = (
            "You are a senior portfolio manager. Based on the three analyst reports, output ONLY a valid JSON object "
            "with these keys: \"rating\" (0.0-1.0 float), \"recommendation\" (\"Buy\"|\"Sell\"|\"Hold\"|\"Neutral\"), "
            "and \"justification\" (string). Do not add any extra text or markdown.\n\n"
            "Market Investor report:\n{market_report}\n\n"
            "Value Investor report:\n{value_report}\n\n"
            "Devil's Advocate report:\n{devils_report}"
        )

        try:
            response = (ChatPromptTemplate.from_template(system_prompt) | self.llm).invoke({
                "market_report": reports["Market Investor"],
                "value_report": reports["Value Investor"],
                "devils_report": reports["Devil's Advocate"]
            })
            # Strip any code fences
            clean_text = re.sub(r"```(?:json)?|```", "", response.content).strip()

            # Non-greedy JSON block
            m = re.search(r"\{.*?\}", clean_text, flags=re.DOTALL)
            if not m:
                # Fallback: try converting single quotes to double quotes
                fallback = clean_text.replace("'", '"')
                m = re.search(r"\{.*?\}", fallback, flags=re.DOTALL)
                if not m:
                    return {"rating": 0.0, "recommendation": "Neutral", "justification": "Invalid JSON."}
                clean_text = fallback

            return json.loads(m.group(0))
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
        top_quant_candidates = quant_df.head(getattr(config, "QUANT_DEEP_DIVE_CANDIDATES", 10))['Ticker'].tolist()

        # Deduplicate while preserving order
        tickers_to_analyze = list(dict.fromkeys(top_quant_candidates + new_tickers))

        logging.info(f"Analyzing {len(tickers_to_analyze)} unique tickers: {tickers_to_analyze}")
        today = pd.Timestamp(datetime.now()).normalize()
        results = []

        for ticker in tickers_to_analyze:
            # Use the most recent analysis if it's still fresh
            if not prev_df.empty and ticker in prev_df['Ticker'].values:
                rows = prev_df[prev_df['Ticker'] == ticker].sort_values(by='Analysis_Date', ascending=False)
                if not rows.empty:
                    last_date = pd.to_datetime(rows.iloc[0]['Analysis_Date']).normalize()
                    if (today - last_date) < timedelta(days=5):
                        logging.info(f"Using recent analysis for {ticker} from {last_date.date()}")
                        results.append(rows.iloc[0].to_dict())
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
        # Safe sort (NaN if parsing fails)
        results_df['Agent_Rating'] = pd.to_numeric(results_df['Agent_Rating'], errors='coerce')
        results_df.sort_values(by="Agent_Rating", ascending=False, inplace=True)
        results_df.reset_index(drop=True, inplace=True)

        results_df.to_csv(config.AGENTIC_RESULTS_PATH, index=False)
        logging.info(f"Analysis complete. Saved to {config.AGENTIC_RESULTS_PATH}")

        return results_df
