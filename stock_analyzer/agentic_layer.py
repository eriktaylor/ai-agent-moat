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
    # Prefer non-deprecated package if available
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

# ------------------------
# Constants / Helpers
# ------------------------
# Conservative US ticker regex (blocks single-letter unless whitelisted; allows BRK.B)
TICKER_REGEX = re.compile(r"\b[A-Z]{2,5}(?:\.[A-Z])?\b")
SINGLE_LETTER_WHITELIST = {"F", "T", "C"}  # add if desired

VALID_US_EXCHANGES = {"NMS", "NYQ", "NCM", "NGM", "BATS", "ASE", "PCX"}  # Nasdaq/NYSE family

SEARCH_CACHE_DIR = getattr(config, "SEARCH_CACHE_DIR", "data/search_cache")
os.makedirs(SEARCH_CACHE_DIR, exist_ok=True)

# --- add near top of agentic_layer.py ---
import hashlib
from pathlib import Path

CACHE_VERSION = "v1"
CACHE_TTL_DAYS = getattr(config, "SEARCH_CACHE_TTL_DAYS", 7)  # override in config if you want
SEARCH_CACHE_DIR = getattr(config, "SEARCH_CACHE_DIR", "data/search_cache")
Path(SEARCH_CACHE_DIR).mkdir(parents=True, exist_ok=True)

def _cache_filename(query: str, bucket: int) -> str:
    # stable readable prefix + short digest to avoid collisions
    key = re.sub(r"[^a-zA-Z0-9]+", "_", query)[:100]
    digest = hashlib.sha1(query.encode("utf-8")).hexdigest()[:8]
    return f"{key}_{bucket}_{digest}.json"

def _load_cache(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        # Back-compat: payload might be raw results array
        if isinstance(payload, dict) and "results" in payload:
            return payload
        return {"_meta": {}, "results": payload}
    except Exception:
        return None

def _save_cache(path: str, query: str, num_results: int, results):
    payload = {
        "_meta": {
            "cache_version": CACHE_VERSION,
            "query": query,
            "num_results": num_results,
            "created_utc": datetime.utcnow().isoformat(timespec="seconds"),
        },
        "results": results,
    }
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
    except Exception:
        pass

def _is_stale(meta: dict, ttl_days: int) -> bool:
    try:
        created = pd.to_datetime(meta.get("created_utc"))
        age = pd.Timestamp.utcnow() - created
        return age > pd.Timedelta(days=ttl_days)
    except Exception:
        # If we can't read metadata, treat as stale
        return True

def _prune_local_cache(ttl_days: int = CACHE_TTL_DAYS):
    pruned = 0
    for p in Path(SEARCH_CACHE_DIR).glob("*.json"):
        payload = _load_cache(str(p))
        if payload is None:
            continue
        if _is_stale(payload.get("_meta", {}), ttl_days):
            try:
                p.unlink()
                pruned += 1
            except Exception:
                pass
    if pruned:
        logging.info(f"🧹 Pruned {pruned} stale cache files (> {ttl_days} days)")

def _safe_json_loads(s: str):
    """Lenient JSON extraction from an LLM response."""
    clean = re.sub(r"```(?:json)?|```", "", s).strip()
    m = re.search(r"\{.*?\}", clean, flags=re.DOTALL)  # non-greedy
    if not m:
        fallback = clean.replace("'", '"')
        m = re.search(r"\{.*?\}", fallback, flags=re.DOTALL)
        if not m:
            return None
        clean = fallback
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def _normalize_judgment(j: dict) -> dict:    
    """Enforce consistency between rating and recommendation."""
    try:
        rating = float(j.get("rating", 0.0))
    except Exception:
        rating = 0.0
    rec = (j.get("recommendation") or "Neutral").title()
    if rating >= 0.66:
        rec = "Buy"
    elif rating <= 0.33:
        rec = "Sell"
    else:
        rec = "Hold" if rec not in {"Hold", "Neutral"} else rec

    """
    #OLD VERSION
    j["rating"] = rating
    j["recommendation"] = rec
    j["justification"] = j.get("justification", "")[:1000]  # keep CSV tidy
    return j
    """

    # NEW: confidence normalization
    try:
        conf = float(j.get("confidence", 0.5))
    except Exception:
        conf = 0.5
    # hard clip to [0,1]
    conf = max(0.0, min(1.0, conf))

    j["rating"] = rating
    j["recommendation"] = rec
    j["confidence"] = conf
    j["justification"] = j.get("justification", "")[:1000]  # keep CSV tidy
    return j

def _canonical_upper(s: str) -> str:
    return (s or "").strip().upper()

class AgenticLayer:
    """
    Scout: find new non-index tickers from news
    Analysts: three personas write constrained reports
    Judge: merges into a unified JSON (rating/recommendation/justification)
    """

    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
        self.search_wrapper = GoogleSearchAPIWrapper(
            google_cse_id=os.environ.get('GOOGLE_CSE_ID'),
            google_api_key=os.environ.get('GOOGLE_API_KEY')
        )
        self.analysis_cache = {}

    # ------------------------
    # Caching wrapper (weekly buckets)
    # ------------------------
    def _cached_search(self, query: str, num_results: int):
        # weekly bucket is still fine for naming; TTL enforces freshness
        bucket = (datetime.utcnow().date().toordinal() // 7)
        fname = _cache_filename(query, bucket)
        path = os.path.join(SEARCH_CACHE_DIR, fname)
    
        # Load if present and fresh
        if os.path.exists(path):
            payload = _load_cache(path)
            if payload and not _is_stale(payload.get("_meta", {}), CACHE_TTL_DAYS):
                return payload["results"]
    
        # Otherwise hit the API and refresh cache
        results = self.search_wrapper.results(query, num_results=num_results)
        _save_cache(path, query, num_results, results)
    return results

    # ------------------------
    # Ticker validation
    # ------------------------
    def _is_valid_equity(self, t: str) -> bool:
        t = _canonical_upper(t)
        if len(t) == 1 and t not in SINGLE_LETTER_WHITELIST:
            return False
        try:
            tk = yf.Ticker(t)
            info = tk.info or {}
            fast = getattr(tk, "fast_info", None) or {}
            has_price = fast.get("last_price") or info.get("regularMarketPrice")
            is_equity = (info.get("quoteType") == "EQUITY")
            on_us_exch = (info.get("exchange") in VALID_US_EXCHANGES)
            has_name_or_cap = info.get("shortName") or info.get("longName") or info.get("marketCap")
            return bool(has_price and is_equity and on_us_exch and has_name_or_cap)
        except Exception:
            return False

    # ------------------------
    # Scout Agent
    # ------------------------
    def _run_scout_agent(self, known_tickers):
        logging.info("Running Scout Agent to find new tickers...")
        new_tickers = set()

        # Bias queries toward recent years to keep results fresh.
        queries = [
            "top performing small cap stocks 2024 OR 2025",
            "undervalued growth stocks outside S&P 500 2024 OR 2025",
            "recent technological breakthrough company stock 2024 OR 2025"
        ]

        for query in queries:
            if len(new_tickers) >= getattr(config, "MAX_SCOUT_RESULTS", 10):
                logging.info("Scout limit reached. Halting search.")
                break

            logging.info(f"Scouting with query: '{query}'")
            try:
                search_results = self._cached_search(query, num_results=5)
                snippets = " ".join(r.get("snippet", "") for r in search_results if isinstance(r, dict))
            except Exception as e:
                logging.error(f"Search API error: {e}")
                continue

            # Optional LLM assist (we still gate with regex + validation)
            try:
                prompt = ChatPromptTemplate.from_template(
                    "Extract US stock tickers (e.g., AAPL, MSFT) mentioned in the text. "
                    "Return a comma-separated list or 'None'.\n\nText:\n{context}"
                )
                llm_resp = (prompt | self.llm).invoke({"context": snippets}).content
            except Exception as e:
                logging.error(f"Ticker extraction LLM error: {e}")
                llm_resp = ""

            # Regex-first extraction to avoid relying on the LLM format
            raw = snippets + " " + llm_resp
            candidates = {m.group(0) for m in TICKER_REGEX.finditer(raw)}
            for t in sorted(candidates):
                t = _canonical_upper(t)
                if t in known_tickers:
                    continue
                if self._is_valid_equity(t):
                    new_tickers.add(t)
                    logging.info(f"✅ Validated scouted ticker: {t}")
                    if len(new_tickers) >= getattr(config, "MAX_SCOUT_RESULTS", 10):
                        break
                else:
                    logging.debug(f"Filtered out non-equity/foreign/defunct ticker: {t}")

        final_list = list(new_tickers)[:getattr(config, "MAX_SCOUT_RESULTS", 10)]
        logging.info(f"Scout found {len(final_list)} new tickers: {final_list}")
        return final_list

    # ------------------------
    # Analyst Agent
    # ------------------------
    def _run_analyst_agent(self, ticker):
        ticker = _canonical_upper(ticker)
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

        # Fresh news: bias toward recent years; keep site: filters plain
        news_queries = {
            "Professional & Financial Analysis":
                f'"{company_name}" ({ticker}) stock analysis 2024 OR 2025 site:reuters.com OR site:bloomberg.com OR site:wsj.com',
            "Retail & Social Sentiment":
                f'"{company_name}" ({ticker}) stock sentiment 2024 OR 2025 site:reddit.com OR site:fool.com OR site:seekingalpha.com',
            "Risk Factors & Negative News":
                f'"{company_name}" ({ticker}) risk OR lawsuit OR investigation OR recall OR safety OR short interest 2024 OR 2025'
        }

        """
        news_context = ""
        for category, query in news_queries.items():
            news_context += f"\n--- {category} ---\n"
            try:
                num_results = 4 if category == "Professional & Financial Analysis" else 2
                results = self._cached_search(query, num_results=num_results)
                if results:
                    for r in results:
                        title = r.get("title", "No Title")
                        snippet = r.get("snippet", "")
                        news_context += f"**{title}**: {snippet}\n"
                else:
                    news_context += "No recent results found.\n"
            except Exception as e:
                logging.error(f"News search error for {ticker} ({category}): {e}")
                news_context += "Error fetching news.\n"
        """
        #NEW VERSION        
        news_context = ""
        evidence_count = 0
        bucket_coverage = 0
        earnings_recent_flag = False
        risk_flag = False

        risk_keywords = ("lawsuit", "investigation", "recall", "probe", "SEC", "short seller", "fraud")
        earnings_keywords = ("earnings", "EPS", "guidance", "quarterly results", "Q1", "Q2", "Q3", "Q4")

        for category, query in news_queries.items():
            news_context += f"\n--- {category} ---\n"
            try:
                num_results = 4 if category == "Professional & Financial Analysis" else 2
                results = self._cached_search(query, num_results=num_results)
                if results:
                    bucket_coverage += 1
                    for r in results:
                        title = r.get("title", "No Title") or "No Title"
                        snippet = r.get("snippet", "") or ""
                        news_context += f"**{title}**: {snippet}\n"
                        evidence_count += 1
                        lower_blob = f"{title} {snippet}".lower()
                        if any(k in lower_blob for k in earnings_keywords):
                            earnings_recent_flag = True
                        if any(k in lower_blob for k in risk_keywords):
                            risk_flag = True
                else:
                    news_context += "No recent results found.\n"
            except Exception as e:
                logging.error(f"News search error for {ticker} ({category}): {e}")
                news_context += "Error fetching news.\n"

        # ... persona generation as before ...
        # Persona analysis with hallucination guardrails
        personas = ["Market Investor", "Value Investor", "Devil's Advocate"]
        system_prompt = (
            "You are an expert financial analyst writing from the perspective of a {persona}.\n"
            "RULES:\n"
            "• DO NOT invent numbers. Use numeric values ONLY if they appear in the Financial Data block.\n"
            "• If a metric is missing or 'N/A', say so explicitly.\n"
            "• Reference at least one concrete item from the News Context (title or outlet), but do not invent dates or prices.\n"
            "• End with a one-sentence conclusion.\n\n"
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

        #Old version
        #self.analysis_cache[ticker] = reports
        #return reports

        #NEW VERSION
        self.analysis_cache[ticker] = reports
        # attach observable metadata the judge/scheduler can use
        reports["_meta"] = {
            "evidence_count": evidence_count,
            "bucket_coverage": bucket_coverage,       # 0..3
            "earnings_recent_flag": earnings_recent_flag,
            "risk_flag": risk_flag,
        }
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

        # No raw braces in the template to avoid LangChain variable parsing issues
        """
        system_prompt = (
            "You are a senior portfolio manager. Based on the three analyst reports, output ONLY a valid JSON object "
            'with keys: "rating" (float 0.0-1.0), "recommendation" ("Buy"|"Sell"|"Hold"|"Neutral"), '
            'and "justification" (string). Do not add any extra text or markdown.\n\n'
            "Market Investor report:\n{market_report}\n\n"
            "Value Investor report:\n{value_report}\n\n"
            "Devil's Advocate report:\n{devils_report}"
        )
        """
        #NEW VERSION
        system_prompt = (
            "You are a senior portfolio manager. Based on the three analyst reports, output ONLY a valid JSON object "
            'with keys: "rating" (float 0.0-1.0), "recommendation" ("Buy"|"Sell"|"Hold"|"Neutral"), '
            '"confidence" (float 0.0-1.0), and "justification" (string). '
            "Rules for confidence:\n"
            "• Start from your internal certainty, but:\n"
            "  - If only one or zero distinct news sources are cited, cap confidence at 0.60.\n"
            "  - If any analyst flags a major risk, cap at 0.50.\n"
            "  - If all three analysts broadly agree, you may raise up to 0.85.\n"
            "Return JSON only, with no extra text.\n\n"
            "Market Investor report:\n{market_report}\n\n"
            "Value Investor report:\n{value_report}\n\n"
            "Devil's Advocate report:\n{devils_report}"
        )
        ##END NEW VERSION

        try:
            response = (ChatPromptTemplate.from_template(system_prompt) | self.llm).invoke({
                "market_report": reports["Market Investor"],
                "value_report": reports["Value Investor"],
                "devils_report": reports["Devil's Advocate"]
            })
            parsed = _safe_json_loads(response.content) or {
                "rating": 0.0, "recommendation": "Neutral",
                "justification": "Invalid JSON."
            }
            return _normalize_judgment(parsed)
        except Exception as e:
            logging.error(f"Judge agent error for {ticker}: {e}")
            return {"rating": 0.0, "recommendation": "Neutral", "justification": "Failed to parse decision."}


    def _min_max_norm(self, s):
        try:
            s = pd.to_numeric(s, errors="coerce")
            lo, hi = s.min(), s.max()
            if pd.isna(lo) or pd.isna(hi) or hi == lo:
                return pd.Series([0.0] * len(s), index=s.index)
            return (s - lo) / (hi - lo)
        except Exception:
            return pd.Series([0.0] * len(s), index=s.index)

    
    # ------------------------
    # Main pipeline
    # ------------------------
    def run_analysis(self):
        logging.info("Starting Agentic Analysis Layer...")

        try:
            quant_df = pd.read_csv(config.CANDIDATE_RESULTS_PATH)
            known_tickers = set(_canonical_upper(t) for t in quant_df['Ticker'].astype(str))
        except FileNotFoundError:
            logging.error(f"Quantitative candidates file missing: {config.CANDIDATE_RESULTS_PATH}")
            return pd.DataFrame()

        try:
            prev_df = pd.read_csv(config.AGENTIC_RESULTS_PATH)
            prev_df['Analysis_Date'] = pd.to_datetime(prev_df['Analysis_Date'])
            known_tickers.update(_canonical_upper(t) for t in prev_df['Ticker'].astype(str))
        except FileNotFoundError:
            logging.info("No previous agentic recommendations found.")
            prev_df = pd.DataFrame()

        # Scout for new names (validated)
        #new_tickers = self._run_scout_agent(known_tickers)
        
        # Take top quant names
        #OLD VERSION
        #top_n = getattr(config, "QUANT_DEEP_DIVE_CANDIDATES", 10)
        #top_quant_candidates = [ _canonical_upper(t) for t in quant_df.head(top_n)['Ticker'].astype(str).tolist() ]
        #NEW VERSION
                # --- Priority selection: quant score + staleness + novelty + quant-delta + earnings boost ---
        top_n = getattr(config, "QUANT_DEEP_DIVE_CANDIDATES", 10)

        # Build last analysis date and last quant score per ticker from prev_df (if available)
        last_dates_dict = {}
        last_quant_dict = {}
        if not prev_df.empty:
            prev_sorted = prev_df.sort_values(by='Analysis_Date', ascending=False)
            last_dates_dict = prev_sorted.drop_duplicates('Ticker').set_index('Ticker')['Analysis_Date'].to_dict()
            if 'Quant_Score' in prev_sorted.columns:
                last_quant_dict = prev_sorted.drop_duplicates('Ticker').set_index('Ticker')['Quant_Score'].to_dict()

        quant_df['Ticker'] = quant_df['Ticker'].astype(str).str.upper()

        # Staleness (days since last agentic)
        quant_df['Last_Agentic'] = quant_df['Ticker'].map(last_dates_dict)
        quant_df['Days_Since_Agentic'] = (today - pd.to_datetime(quant_df['Last_Agentic'])).dt.days
        quant_df['Days_Since_Agentic'] = quant_df['Days_Since_Agentic'].fillna(999)  # never analyzed -> very stale

        # Novelty flag: never analyzed before
        quant_df['Novelty_Flag'] = quant_df['Last_Agentic'].isna().astype(int)

        # Quant-score delta vs last run (absolute change)
        if 'Quant_Score' in quant_df.columns:
            quant_df['Prev_Quant'] = quant_df['Ticker'].map(last_quant_dict)
            quant_df['Quant_Delta'] = (quant_df['Quant_Score'] - quant_df['Prev_Quant']).abs()
            quant_df['Quant_Delta'] = quant_df['Quant_Delta'].fillna(0.0)
        else:
            quant_df['Quant_Score'] = 0.0
            quant_df['Quant_Delta'] = 0.0

        # Normalize the parts
        qn = _min_max_norm(quant_df['Quant_Score'])
        stale_n = _min_max_norm(quant_df['Days_Since_Agentic'])
        qdelta_n = _min_max_norm(quant_df['Quant_Delta'])

        # Earnings boost seed (0 now; we'll add per-ticker after analyst pass if needed)
        quant_df['Earnings_Boost'] = 0.0

        # Priority = 0.6*quant + 0.3*staleness + 0.1*novelty + 0.05*quant_delta + 0.05*earnings_boost
        quant_df['Priority'] = 0.6*qn + 0.3*stale_n + 0.1*quant_df['Novelty_Flag'] + 0.05*qdelta_n + 0.05*quant_df['Earnings_Boost']

        # Pick top-N by priority
        top_quant_candidates = quant_df.sort_values('Priority', ascending=False).head(top_n)['Ticker'].tolist()
        #END NEW VERSION
        
        # Deduplicate while preserving order
        tickers_to_analyze = list(dict.fromkeys(top_quant_candidates + new_tickers))
        logging.info(f"Analyzing {len(tickers_to_analyze)} unique tickers: {tickers_to_analyze}")

        today = pd.Timestamp(datetime.now()).normalize()
        results = []

        for ticker in tickers_to_analyze:
            # Use the most recent analysis if it's still fresh
            if not prev_df.empty and ticker in set(_canonical_upper(t) for t in prev_df['Ticker'].astype(str)):
                rows = prev_df[prev_df['Ticker'].str.upper() == ticker].sort_values(by='Analysis_Date', ascending=False)
                if not rows.empty:
                    last_date = pd.to_datetime(rows.iloc[0]['Analysis_Date']).normalize()
                    if (today - last_date) < timedelta(days=5):
                        logging.info(f"Using recent analysis for {ticker} from {last_date.date()}")
                        results.append(rows.iloc[0].to_dict())
                        continue

            reports = self._run_analyst_agent(ticker)
            judgment = self._run_ranking_judge_agent(reports, ticker)

            # Pull quant score if present
            qser = quant_df.loc[quant_df['Ticker'].str.upper() == ticker, 'Quant_Score']
            quant_score = qser.iloc[0] if not qser.empty else 'N/A'

            results.append({
                'Ticker': ticker,
                'Quant_Score': quant_score,
                'Analysis_Date': today.strftime('%Y-%m-%d'),
                'Agent_Rating': judgment.get('rating'),
                'Agent_Recommendation': judgment.get('recommendation'),
                'Agent_Confidence': judgment.get('confidence'),  # <-- NEW
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

        _prune_local_cache()

        return results_df
