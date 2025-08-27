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

import hashlib
from pathlib import Path
from urllib.parse import urlparse

CACHE_VERSION = "v1"

# Search cache
CACHE_TTL_DAYS = getattr(config, "SEARCH_CACHE_TTL_DAYS", 7)  # override in config if you want
SEARCH_CACHE_DIR = getattr(config, "SEARCH_CACHE_DIR", "data/search_cache")
Path(SEARCH_CACHE_DIR).mkdir(parents=True, exist_ok=True)

# --- Article content cache (for full-text fetching) ---
ARTICLE_CACHE_DIR = getattr(config, "ARTICLE_CACHE_DIR", "data/article_cache")
ARTICLE_CACHE_TTL_DAYS = getattr(config, "ARTICLE_CACHE_TTL_DAYS", 14)
Path(ARTICLE_CACHE_DIR).mkdir(parents=True, exist_ok=True)

# Optional deps for article fetching; fall back gracefully if missing
try:
    import requests
    HAVE_REQUESTS = True
except Exception:
    HAVE_REQUESTS = False

try:
    from bs4 import BeautifulSoup
    HAVE_BS4 = True
except Exception:
    HAVE_BS4 = False


def _get_last(df, rows):
    """Return the most recent value for the first matching row name in `rows`."""
    try:
        if df is None or df.empty:
            return None
        for name in rows:
            if name in df.index:
                # yfinance financials/balance_sheet columns are datestamps; get latest (leftmost)
                val = df.loc[name].dropna()
                if not val.empty:
                    return float(val.iloc[0])
    except Exception:
        pass
    return None


def _collect_financials(ticker: str) -> dict:
    tk = yf.Ticker(ticker)
    info = tk.info or {}
    fast = getattr(tk, "fast_info", None) or {}

    # Statements (robust to field-name variants)
    try:
        bs = tk.balance_sheet  # annual
    except Exception:
        bs = pd.DataFrame()
    try:
        cf = tk.cashflow  # annual
    except Exception:
        cf = pd.DataFrame()
    try:
        fin = tk.financials  # annual income statement
    except Exception:
        fin = pd.DataFrame()

    mcap = info.get("marketCap") or fast.get("market_cap")
    price = fast.get("last_price") or info.get("regularMarketPrice")

    total_debt = _get_last(bs, [
        "Total Debt", "Short Long Term Debt", "Short/Long Term Debt",
        "Long Term Debt And Capital Lease Obligation", "Total Debt Net"
    ])
    equity = _get_last(bs, [
        "Total Stockholder Equity", "Total Equity Gross Minority Interest", "Total Equity"
    ])
    cash = _get_last(bs, [
        "Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments", "Cash"
    ])
    ebitda = _get_last(fin, ["Ebitda", "EBITDA"])

    # Free Cash Flow = Operating Cash Flow + Capital Expenditures (capex is typically negative)
    ocf = _get_last(cf, ["Total Cash From Operating Activities", "Operating Cash Flow"])
    capex = _get_last(cf, ["Capital Expenditures"])
    fcf = None
    if ocf is not None and capex is not None:
        fcf = float(ocf + capex)

    # Dividends / Yield
    dividend_yield = info.get("dividendYield")
    if dividend_yield is None:
        try:
            divs = tk.dividends
            if price and divs is not None and not divs.empty:
                trailing_12m = divs[divs.index >= (divs.index.max() - pd.DateOffset(years=1))].sum()
                if trailing_12m and price:
                    dividend_yield = float(trailing_12m / price)
        except Exception:
            pass

    # Ratios
    price_to_book = info.get("priceToBook")
    if price_to_book is None and mcap and equity and equity != 0:
        price_to_book = float(mcap / equity)

    debt_to_equity = None
    if total_debt is not None and equity not in (None, 0):
        debt_to_equity = float(total_debt / equity)

    ev = None
    if mcap is not None:
        ev = float(mcap + (total_debt or 0) - (cash or 0))
    ev_to_ebitda = None
    if ev is not None and ebitda not in (None, 0):
        ev_to_ebitda = float(ev / ebitda)

    # Light sanity caps (avoid absurd numbers from info)
    def _cap(x, lo, hi):
        try:
            x = float(x)
            if x < lo or x > hi:
                return None
            return x
        except Exception:
            return None

    trailing_pe = _cap(info.get("trailingPE"), 0, 5000)
    forward_pe  = _cap(info.get("forwardPE"), 0, 5000)
    fifty_two_high = _cap(info.get("fiftyTwoWeekHigh"), 0, 1e6)
    fifty_two_low  = _cap(info.get("fiftyTwoWeekLow"), 0, 1e6)

    return {
        "longName": info.get("longName") or ticker,
        "sector": info.get("sector") or "N/A",
        "marketCap": mcap,
        "price": price,
        "trailingPE": trailing_pe,
        "forwardPE": forward_pe,
        "priceToBook": price_to_book,
        "debtToEquity": debt_to_equity,
        "freeCashFlow": fcf,
        "dividendYield": dividend_yield,
        "fiftyTwoWeekHigh": fifty_two_high,
        "fiftyTwoWeekLow": fifty_two_low,
        "EV_EBITDA": ev_to_ebitda,
    }


def _cache_filename(query: str, bucket: int) -> str:
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
        return True


def _article_cache_path(url: str) -> str:
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    return os.path.join(ARTICLE_CACHE_DIR, f"{digest}.json")


def _load_article_cache(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload
    except Exception:
        return None


def _save_article_cache(path: str, url: str, text: str):
    payload = {
        "_meta": {
            "url": url,
            "created_utc": datetime.utcnow().isoformat(timespec="seconds"),
            "cache_version": CACHE_VERSION,
        },
        "text": text,
    }
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
    except Exception:
        pass


def _is_article_stale(meta: dict, ttl_days: int = ARTICLE_CACHE_TTL_DAYS) -> bool:
    try:
        created = pd.to_datetime(meta.get("created_utc"))
        age = pd.Timestamp.utcnow() - created
        return age > pd.Timedelta(days=ttl_days)
    except Exception:
        return True


def _extract_main_text(html: str) -> str:
    if not HAVE_BS4:
        return ""
    try:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        container = soup.find("article") or soup.find("main")
        if not container:
            for cls in ["content", "article", "post", "story", "entry"]:
                container = soup.find("div", class_=lambda c: c and cls in c.lower())
                if container:
                    break
        text = (container or soup).get_text(separator=" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:5000]
    except Exception:
        return ""


def _fetch_article_text(url: str, timeout: int = 10) -> str:
    # cache lookup
    p = _article_cache_path(url)
    cached = _load_article_cache(p)
    if cached and not _is_article_stale(cached.get("_meta", {}), ARTICLE_CACHE_TTL_DAYS):
        return cached.get("text", "")

    if not HAVE_REQUESTS:
        return ""

    headers = {"User-Agent": "Mozilla/5.0 (agentic-layer)"}
    text = ""
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        if resp.status_code == 200 and resp.text:
            text = _extract_main_text(resp.text)
    except Exception:
        text = ""

    if text:
        _save_article_cache(p, url, text)
    return text


def _prune_local_cache(ttl_days: int = CACHE_TTL_DAYS):
    # prune search cache
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
        logging.info(f"🧹 Pruned {pruned} stale search cache files (> {ttl_days} days)")

    # prune article cache
    apruned = 0
    for p in Path(ARTICLE_CACHE_DIR).glob("*.json"):
        payload = _load_article_cache(str(p))
        if payload is None:
            continue
        if _is_article_stale(payload.get("_meta", {}), ARTICLE_CACHE_TTL_DAYS):
            try:
                p.unlink()
                apruned += 1
            except Exception:
                pass
    if apruned:
        logging.info(f"🧹 Pruned {apruned} stale article cache files (> {ARTICLE_CACHE_TTL_DAYS} days)")


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

    # confidence normalization
    try:
        conf = float(j.get("confidence", 0.5))
    except Exception:
        conf = 0.5
    conf = max(0.0, min(1.0, conf))  # hard clip

    j["rating"] = rating
    j["recommendation"] = rec
    j["confidence"] = conf
    j["justification"] = j.get("justification", "")[:1000]  # keep CSV tidy
    return j


def _canonical_upper(s: str) -> str:
    return (s or "").strip().upper()


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc
    except Exception:
        return ""


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

        this_year = datetime.utcnow().year
        last_year = this_year - 1

        # Bias queries toward recent years to keep results fresh.
        queries = [
            f"top performing small cap stocks {last_year} OR {this_year}",
            f"undervalued growth stocks outside S&P 500 {last_year} OR {this_year}",
            f"recent technological breakthrough company stock {last_year} OR {this_year}",
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
            _ = yf.Ticker(ticker).info or {}
        except Exception as e:
            logging.error(f"Yahoo Finance API error for {ticker}: {e}")
            return {"error": str(e)}

        # Collect richer financials
        financial_data = _collect_financials(ticker)
        company_name = financial_data.get("longName", ticker)

        # News buckets (diverse sources)
        this_year = datetime.utcnow().year
        last_year = this_year - 1
        news_queries = {
            # Pro outlets
            "Professional & Financial Analysis":
                f'"{company_name}" ({ticker}) stock analysis {last_year} OR {this_year} '
                f'site:reuters.com OR site:bloomberg.com OR site:wsj.com OR site:ft.com OR site:cnbc.com',
            # Aggregators (Google News + Yahoo News)
            "General Headlines":
                f'"{company_name}" ({ticker}) {last_year} OR {this_year} '
                f'site:news.google.com OR site:news.yahoo.com OR site:finance.yahoo.com',
            # Retail/social
            "Retail & Social Sentiment":
                f'"{company_name}" ({ticker}) stock sentiment {last_year} OR {this_year} '
                f'site:reddit.com OR site:fool.com OR site:seekingalpha.com',
            # Risks
            "Risk Factors & Negative News":
                f'"{company_name}" ({ticker}) risk OR lawsuit OR investigation OR recall OR safety OR short interest '
                f'{last_year} OR {this_year}',
        }

        # Evidence/meta
        evidence_items = []
        distinct_domains = set()
        pro_hits = retail_hits = fulltext_hits = 0
        evidence_count = 0
        bucket_coverage = 0
        earnings_recent_flag = False
        risk_flag = False

        # word-boundary regex to reduce false positives (e.g., "sec" != "seconds")
        import re as _re
        risk_patterns = [_re.compile(p) for p in (
            r"\blawsuit\b", r"\binvestigation\b", r"\brecall\b",
            r"\bprobe\b", r"\bsec\b", r"\bshort seller\b", r"\bfraud\b"
        )]
        earnings_patterns = [_re.compile(p) for p in (
            r"\bearnings\b", r"\beps\b", r"\bguidance\b", r"\bquarterly results\b", r"\bq[1-4]\b"
        )]

        # paywall note: keep titles/snippets if body can't be fetched
        paywall_suffixes = ("wsj.com", "bloomberg.com")

        pro_domains = (
            "reuters.com","ft.com","wsj.com","bloomberg.com","sec.gov",
            "investor.", "ir.", "cnbc.com","finance.yahoo.com"
        )
        retail_domains = ("reddit.com","stocktwits.com","fool.com","seekingalpha.com")

        def _norm_domain(u: str) -> str:
            d = _domain(u).lower() if u else ""
            if d.startswith(("www.", "m.")):
                d = d.split(".", 1)[1]
            return d

        def _push_item(title, url, excerpt, category, body_present):
            nonlocal pro_hits, retail_hits, fulltext_hits, evidence_count
            dom = _norm_domain(url)
            evidence_items.append({
                "dom": dom, "title": title or "No Title", "excerpt": (excerpt or "").strip(),
                "category": category, "has_body": bool(body_present),
            })
            if dom:
                distinct_domains.add(dom)
                if any(d in dom for d in pro_domains): pro_hits += 1
                elif any(d in dom for d in retail_domains): retail_hits += 1
            if body_present: fulltext_hits += 1
            evidence_count += 1

        # 1) Google buckets
        for category, query in news_queries.items():
            try:
                num_results = 5 if category in ("Professional & Financial Analysis", "General Headlines") else 4
                results = self._cached_search(query, num_results=num_results)
                if results:
                    bucket_coverage += 1
                    for r in results:
                        title   = (r.get("title") or "No Title")
                        snippet = (r.get("snippet") or "")
                        url     = r.get("link") or r.get("url") or ""
                        dom     = _norm_domain(url)

                        body = ""
                        if url and HAVE_REQUESTS and HAVE_BS4 and not any(dom.endswith(sfx) for sfx in paywall_suffixes):
                            body = _fetch_article_text(url)

                        excerpt = (body[:1200] if body else snippet[:500])
                        _push_item(title, url, excerpt, category, bool(body))

                        lower_blob = f"{title} {snippet} {excerpt}".lower()
                        if any(p.search(lower_blob) for p in earnings_patterns): earnings_recent_flag = True
                        if any(p.search(lower_blob) for p in risk_patterns):     risk_flag = True
            except Exception as e:
                logging.error(f"News search error for {ticker} ({category}): {e}")

        # 2) Yahoo Finance news (0 Google quota)
        ENABLE_YF_NEWS = getattr(config, "ENABLE_YF_NEWS", True)
        YF_NEWS_MAX = getattr(config, "YF_NEWS_MAX", 12)
        if ENABLE_YF_NEWS:
            try:
                yf_news = yf.Ticker(ticker).news or []
            except Exception:
                yf_news = []
            for item in yf_news[:YF_NEWS_MAX]:
                title = item.get("title") or "No Title"
                url   = item.get("link") or ""
                body  = _fetch_article_text(url) if url else ""
                excerpt = (body[:1200] if body else "") or (item.get("summary") or "")[:500]
                _push_item(title, url, excerpt, "General Headlines", bool(body))

                lower_blob = f"{title} {excerpt}".lower()
                if any(p.search(lower_blob) for p in earnings_patterns): earnings_recent_flag = True
                if any(p.search(lower_blob) for p in risk_patterns):     risk_flag = True

        # Select a diversified subset for prompting
        EVIDENCE_BUDGET = getattr(config, "EVIDENCE_BUDGET", 18)
        PER_DOMAIN_CAP = getattr(config, "PER_DOMAIN_CAP", 3)
        pro_like = ("reuters.com","ft.com","wsj.com","bloomberg.com","sec.gov","cnbc.com","finance.yahoo.com","investor.","ir.")

        def _score(it):
            s = 0
            if it["has_body"]: s += 2
            if any(d in (it["dom"] or "") for d in pro_like): s += 1
            return (s, it["dom"] or "", it["title"] or "")

        selected = []
        per_dom = {}
        for it in sorted(evidence_items, key=_score, reverse=True):
            dom = it["dom"] or ""
            if per_dom.get(dom, 0) >= PER_DOMAIN_CAP:
                continue
            selected.append(it)
            per_dom[dom] = per_dom.get(dom, 0) + 1
            if len(selected) >= EVIDENCE_BUDGET:
                break

        # Build News Context grouped by category
        news_context = ""
        for cat in ["Professional & Financial Analysis", "General Headlines", "Retail & Social Sentiment", "Risk Factors & Negative News"]:
            news_context += f"\n--- {cat} ---\n"
            for it in selected:
                if it["category"] == cat:
                    news_context += f"**{it['title']}** ({it['dom'] or 'n/a'}): {it['excerpt']}\n"

        # Persona analysis (dates allowed; discourage stale financials)
        today_str = datetime.now().strftime("%Y-%m-%d")
        personas = ["Market Investor", "Value Investor", "Devil's Advocate"]
        system_prompt = (
            "You are an expert financial analyst writing from the perspective of a {persona}.\n"
            "Rules:\n"
            f"• Today's date is {today_str}. You may cite dates from articles.\n"
            "• Prefer numeric values from the Financial Data block. If you use article numbers, attribute the outlet "
            "  (domain) and note if the article is older than ~12 months.\n"
            "• If a metric is missing or 'N/A', say so explicitly.\n"
            "• Combine fundamentals and news context; end with a one-sentence conclusion.\n\n"
            "--- FINANCIAL DATA (canonical) ---\n{financial_data}\n\n"
            "--- NEWS CONTEXT (titles/excerpts; may include dates) ---\n{news_context}\n\n"
            "--- {persona} Analysis ---"
        )
        prompt = ChatPromptTemplate.from_template(system_prompt)
        chain = prompt | self.llm

        reports = {}
        for persona in personas:
            try:
                response = chain.invoke({
                    "persona": persona,
                    "financial_data": json.dumps(financial_data),
                    "news_context": news_context
                })
                reports[persona] = response.content
            except Exception as e:
                logging.error(f"Persona analysis error for {ticker} ({persona}): {e}")
                reports[persona] = "Analysis failed."

        # attach observable metadata the judge/scheduler can use
        self.analysis_cache[ticker] = reports
        reports["_meta"] = {
            "evidence_total_collected": len(evidence_items),
            "evidence_used": len(selected),
            "domains_used": sorted({it["dom"] for it in selected if it["dom"]}),
            "evidence_count": len(selected),           # kept for backward-compat use by judge
            "bucket_coverage": bucket_coverage,        # 0..4 now
            "earnings_recent_flag": earnings_recent_flag,
            "risk_flag": risk_flag,
            "distinct_domains": len(distinct_domains),
            "pro_hits": pro_hits,
            "retail_hits": retail_hits,
            "fulltext_hits": fulltext_hits,
            "missing_financials": sum(1 for _, v in financial_data.items() if v in (None, "N/A", "")),
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

        # Pull meta for confidence rubric
        meta = reports.get("_meta", {}) if isinstance(reports, dict) else {}
        bucket_coverage = int(meta.get("bucket_coverage", 0) or 0)
        evidence_count = int(meta.get("evidence_count", 0) or 0)
        risk_flag = bool(meta.get("risk_flag", False))

        distinct_domains = int(meta.get("distinct_domains", 0) or 0)
        pro_hits = int(meta.get("pro_hits", 0) or 0)
        retail_hits = int(meta.get("retail_hits", 0) or 0)
        fulltext_hits = int(meta.get("fulltext_hits", 0) or 0)
        missing_financials = int(meta.get("missing_financials", 0) or 0)

        system_prompt = (
            "You are a senior portfolio manager. Based on the three analyst reports, output ONLY a valid JSON object "
            'with keys: "rating" (float 0.0-1.0), "recommendation" ("Buy"|"Sell"|"Hold"|"Neutral"), '
            '"confidence" (float 0.0-1.0), and "justification" (string).\n\n'
            f"META:\n"
            f"- bucket_coverage: {{bucket_coverage}}\n"
            f"- evidence_count: {{evidence_count}}\n"
            f"- risk_flag: {{risk_flag}}\n"
            f"- distinct_domains: {{distinct_domains}}\n"
            f"- pro_hits: {{pro_hits}}\n"
            f"- retail_hits: {{retail_hits}}\n"
            f"- fulltext_hits: {{fulltext_hits}}\n"
            f"- missing_financials: {{missing_financials}}\n\n"
            "Confidence rubric (apply deterministically, then clip 0..1):\n"
            "• Start at 0.45\n"
            "• +0.10 if pro_hits ≥ 2; +0.05 more if pro_hits ≥ 4\n"
            "• +0.05 if distinct_domains ≥ 4; −0.05 if distinct_domains ≤ 1\n"
            "• +0.05 if fulltext_hits ≥ 3\n"
            "• +0.05 if bucket_coverage ≥ 3 and evidence_count ≥ 8\n"
            "• −0.10 if retail_hits > 2 × pro_hits\n"
            "• −0.10 if missing_financials ≥ 3\n"
            "• −0.15 if risk_flag is true\n"
            "Return JSON only, with no extra text.\n\n"
            "Market Investor report:\n{market_report}\n\n"
            "Value Investor report:\n{value_report}\n\n"
            "Devil's Advocate report:\n{devils_report}"
        )

        try:
            response = (ChatPromptTemplate.from_template(system_prompt) | self.llm).invoke({
                "market_report": reports["Market Investor"],
                "value_report": reports["Value Investor"],
                "devils_report": reports["Devil's Advocate"],
                "bucket_coverage": bucket_coverage,
                "evidence_count": evidence_count,
                "risk_flag": risk_flag,
                "distinct_domains": distinct_domains,
                "pro_hits": pro_hits,
                "retail_hits": retail_hits,
                "fulltext_hits": fulltext_hits,
                "missing_financials": missing_financials,
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
        today = pd.Timestamp(datetime.now()).normalize()

        try:
            quant_df = pd.read_csv(config.CANDIDATE_RESULTS_PATH)
            known_tickers = set(_canonical_upper(t) for t in quant_df['Ticker'].astype(str))
        except FileNotFoundError:
            logging.error(f"Quantitative candidates file missing: {config.CANDIDATE_RESULTS_PATH}")
            return pd.DataFrame()

        try:
            prev_df = pd.read_csv(config.AGENTIC_RESULTS_PATH)
            prev_df['Analysis_Date'] = pd.to_datetime(prev_df['Analysis_Date'])
            # normalize ticker casing in prev snapshot for reliable mapping
            prev_df['Ticker'] = prev_df['Ticker'].astype(str).str.upper()
            known_tickers.update(_canonical_upper(t) for t in prev_df['Ticker'].astype(str))
        except FileNotFoundError:
            logging.info("No previous agentic recommendations found.")
            prev_df = pd.DataFrame()

        # Scout for new names (feature-flagged)
        ENABLE_SCOUT = getattr(config, "ENABLE_SCOUT", False)
        new_tickers = self._run_scout_agent(known_tickers) if ENABLE_SCOUT else []

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
        qn = self._min_max_norm(quant_df['Quant_Score'])
        stale_n = self._min_max_norm(quant_df['Days_Since_Agentic'])
        qdelta_n = self._min_max_norm(quant_df['Quant_Delta'])

        # Earnings boost seed (0 now; reserved for later if you want to toggle)
        quant_df['Earnings_Boost'] = 0.0

        # Priority = 0.6*quant + 0.3*staleness + 0.1*novelty + 0.05*quant_delta + 0.05*earnings_boost
        quant_df['Priority'] = 0.6*qn + 0.3*stale_n + 0.1*quant_df['Novelty_Flag'] + 0.05*qdelta_n + 0.05*quant_df['Earnings_Boost']

        # Pick top-N by priority
        top_quant_candidates = quant_df.sort_values('Priority', ascending=False).head(top_n)['Ticker'].tolist()

        # Deduplicate while preserving order
        tickers_to_analyze = list(dict.fromkeys([*top_quant_candidates, *new_tickers]))
        logging.info(f"Analyzing {len(tickers_to_analyze)} unique tickers: {tickers_to_analyze}")

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

            # --- Confidence post-processing (observable caps/boosts) ---
            meta = reports.get("_meta", {}) if isinstance(reports, dict) else {}
            evidence_count = int(meta.get("evidence_count", 0) or 0)
            bucket_coverage = int(meta.get("bucket_coverage", 0) or 0)  # 0..4
            risk_flag = bool(meta.get("risk_flag", False))

            if "confidence" not in judgment or judgment["confidence"] is None:
                pro_hits = int(meta.get("pro_hits", 0) or 0)
                retail_hits = int(meta.get("retail_hits", 0) or 0)
                distinct_domains = int(meta.get("distinct_domains", 0) or 0)
                fulltext_hits = int(meta.get("fulltext_hits", 0) or 0)
                missing_financials = int(meta.get("missing_financials", 0) or 0)

                base_conf = 0.45
                if pro_hits >= 2: base_conf += 0.10
                if pro_hits >= 4: base_conf += 0.05
                if distinct_domains >= 4: base_conf += 0.05
                if distinct_domains <= 1: base_conf -= 0.05
                if fulltext_hits >= 3: base_conf += 0.05
                if bucket_coverage >= 3 and evidence_count >= 8: base_conf += 0.05
                if retail_hits > 2 * max(pro_hits, 1): base_conf -= 0.10
                if missing_financials >= 3: base_conf -= 0.10
                if risk_flag: base_conf -= 0.15
            else:
                base_conf = float(judgment.get("confidence", 0.5) or 0.5)

            # Caps/boosts
            if bucket_coverage <= 1:
                base_conf = min(base_conf, 0.60)
            if risk_flag:
                base_conf = min(base_conf, 0.50)
            if bucket_coverage >= 3 and evidence_count >= 8:
                base_conf = min(base_conf + 0.05, 1.0)

            judgment["confidence"] = max(0.0, min(1.0, base_conf))

            # Pull quant score if present
            qser = quant_df.loc[quant_df['Ticker'].str.upper() == ticker, 'Quant_Score']
            quant_score = qser.iloc[0] if not qser.empty else 'N/A'

            # Results row (includes evidence summary)
            results.append({
                'Ticker': ticker,
                'Quant_Score': quant_score,
                'Analysis_Date': today.strftime('%Y-%m-%d'),
                'Agent_Rating': judgment.get('rating'),
                'Agent_Recommendation': judgment.get('recommendation'),
                'Agent_Confidence': judgment.get('confidence'),
                'Justification': judgment.get('justification'),
                'Market_Investor_Analysis': reports.get('Market Investor', 'N/A'),
                'Value_Investor_Analysis': reports.get('Value Investor', 'N/A'),
                'Devils_Advocate_Analysis': reports.get("Devil's Advocate", 'N/A'),
                'Evidence_Total': meta.get('evidence_total_collected'),
                'Evidence_Used': meta.get('evidence_used'),
                'Domains_Used': ", ".join(meta.get('domains_used', []))[:1000],
            })

        # === carry-forward snapshot ===
        results_df = pd.DataFrame(results)
        results_df['Agent_Rating'] = pd.to_numeric(results_df['Agent_Rating'], errors='coerce')

        if not prev_df.empty:
            prev_latest = prev_df.sort_values('Analysis_Date').drop_duplicates('Ticker', keep='last')
            all_cols = sorted(set(prev_latest.columns).union(results_df.columns))
            prev_latest = prev_latest.reindex(columns=all_cols)
            results_df  = results_df.reindex(columns=all_cols)

            prev_latest = prev_latest.set_index('Ticker')
            today_idxed = results_df.set_index('Ticker')
            prev_latest.update(today_idxed)
            snapshot_df = prev_latest.combine_first(today_idxed).reset_index()
        else:
            snapshot_df = results_df.copy()

        # Final ordering & date formatting
        snapshot_df['Analysis_Date'] = pd.to_datetime(snapshot_df['Analysis_Date']).dt.date.astype(str)
        snapshot_df['Agent_Rating'] = pd.to_numeric(snapshot_df['Agent_Rating'], errors='coerce')
        snapshot_df['Quant_Score']  = pd.to_numeric(snapshot_df['Quant_Score'], errors='coerce')
        snapshot_df.sort_values(by=["Agent_Rating","Quant_Score"], ascending=[False, False], inplace=True)
        snapshot_df.reset_index(drop=True, inplace=True)

        snapshot_df.to_csv(config.AGENTIC_RESULTS_PATH, index=False)
        logging.info(f"Analysis complete. Saved to {config.AGENTIC_RESULTS_PATH} (carry-forward snapshot)")

        _prune_local_cache()

        return results_df
