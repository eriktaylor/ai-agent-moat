import streamlit as st
import datetime
import re
import locale
import os
import traceback
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from research_agent import ResearchAgent
from tools import get_stock_info
from display_utils import display_analysis

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Agent for Financial Moat Analysis",
    page_icon="🤖",
    layout="wide"
)

# --- App Title and Description ---
st.title("AI Agent for Financial Moat Analysis")
st.markdown("""
Welcome to the interactive demo of the AI Financial Analyst Agent.
This tool automates investment research by analyzing a company's competitive advantages, or "moat."
To begin, please enter your Google Gemini API key in the sidebar.
""")

# --- Session State Initialization ---
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'current_analysis_index' not in st.session_state:
    st.session_state.current_analysis_index = None
if 'show_feedback_box' not in st.session_state:
    st.session_state.show_feedback_box = False


# --- Helper Functions ---
def parse_financial_data(data_string):
    data = {}
    if not isinstance(data_string, str) or data_string.startswith("Error"):
        return data
    pattern = re.compile(r"([^:]+):\s*(.*)")
    for line in data_string.split('\n'):
        match = pattern.match(line)
        if match:
            key, value = match.group(1).strip(), match.group(2).strip()
            data[key] = value
    return data

def format_large_number(value_str):
    if value_str is None or value_str == "N/A": return "N/A"
    try:
        num = float(value_str)
        if num >= 1_000_000_000_000: return f"${num / 1_000_000_000_000:.2f}T"
        if num >= 1_000_000_000: return f"${num / 1_000_000_000:.2f}B"
        if num >= 1_000_000: return f"${num / 1_000_000:.2f}M"
        return f"${num:,.2f}"
    except (ValueError, TypeError): return "N/A"

def get_first_available_value(data_dict, keys_to_try):
    for key in keys_to_try:
        if key in data_dict and data_dict[key] not in [None, "N/A", "None"]:
            return data_dict[key]
    return "N/A"

# --- Sidebar for Inputs and API Keys ---
with st.sidebar:
    st.header("⚙️ Configuration")
    google_api_key = st.text_input("Enter your Google Gemini API Key", type="password", help="Get your free API key from [Google AI Studio](https://aistudio.google.com/app/apikey).")
    st.markdown("[Get your free Gemini API key here](https://aistudio.google.com/app/apikey)")
    st.markdown("---")
    st.header("▶️ Run New Analysis")
    company_name_input = st.text_input("Enter Company Name", "Apple")
    stock_ticker_input = st.text_input("Enter Stock Ticker", "AAPL")
    run_button = st.button("Run New Analysis", use_container_width=True, type="primary")

# --- Helper for safely initializing embeddings ---
def _safe_embeddings():
    try:
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:
        st.warning(f"Could not load primary embedding model. Error: {e}. Falling back to a smaller model.")
        return HuggingFaceEmbeddings(model_name="intfloat/e5-small-v2")

# --- Agent Initialization (Cached) ---
@st.cache_resource
def initialize_agent(api_key):
    """Initializes the ResearchAgent."""
    os.environ["GOOGLE_API_KEY"] = api_key
    # Using gemini-2.5-flash per your suggestion
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2, api_key=api_key)
    embeddings = _safe_embeddings()

    # --- FIX: Make search tool initialization fail-soft ---
    try:
        from langchain_community.tools import DuckDuckGoSearchRun
        search_tool = DuckDuckGoSearchRun()
    except Exception as e:
        st.warning(f"DuckDuckGo search is unavailable: {e}. Proceeding without web search.")
        search_tool = None # Set to None if import fails

    return ResearchAgent(llm=llm, embeddings_model=embeddings, search_tool=search_tool)


# --- Main Application Logic ---
def run_full_analysis(company_name, stock_ticker):
    base_results = {
        "company_name": company_name, "stock_ticker": stock_ticker,
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "error": False, "status": "error", "feedback": None,
        "detailed_feedback": None, "financial_data": None,
        "market_outlook": {"answer": "Analysis could not be run."},
        "value_analysis": {"answer": "Analysis could not be run."},
        "devils_advocate": {"answer": "Analysis could not be run."},
        "final_summary": "Analysis could not be run."
    }
    try:
        with st.spinner(f"Running analysis for {company_name}..."):
            research_agent = initialize_agent(google_api_key)
            financial_data_str = get_stock_info.run(stock_ticker)
            base_results["financial_data"] = financial_data_str
            if financial_data_str.startswith("Error"):
                st.warning(f"Could not retrieve complete financial data for {stock_ticker}. Proceeding with web analysis only.")

            base_results["market_outlook"] = research_agent.generate_market_outlook(company_name, stock_ticker)
            base_results["value_analysis"] = research_agent.generate_value_analysis(company_name, stock_ticker)
            base_results["devils_advocate"] = research_agent.generate_devils_advocate_view(company_name, stock_ticker)
            base_results["final_summary"] = research_agent.generate_final_summary(
                company_name,
                base_results["market_outlook"].get('answer', ''),
                base_results["value_analysis"].get('answer', ''),
                base_results["devils_advocate"].get('answer', '')
            )
            base_results["status"] = "complete"
        return base_results
    except Exception as e:
        st.error(f"A critical error occurred during the analysis: {e}")
        st.caption("Traceback:")
        st.code(traceback.format_exc())
        base_results["error"] = True
        return base_results

# --- UI Display Logic ---
if run_button:
    if not google_api_key: st.error("Please enter your Google Gemini API Key.")
    elif not company_name_input or not stock_ticker_input: st.error("Please enter a company name and stock ticker.")
    else:
        st.session_state.analysis_history.insert(0, {
            "company_name": company_name_input, "stock_ticker": stock_ticker_input,
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "status": "pending"
        })
        st.session_state.current_analysis_index = 0
        st.rerun()

pending_analysis_index = next((i for i, an in enumerate(st.session_state.analysis_history) if an.get("status") == "pending"), None)
if pending_analysis_index is not None:
    pending_info = st.session_state.analysis_history[pending_analysis_index]
    full_results = run_full_analysis(pending_info["company_name"], pending_info["stock_ticker"])
    st.session_state.analysis_history[pending_analysis_index] = full_results
    st.session_state.current_analysis_index = pending_analysis_index
    if full_results.get("error"):
        st.error("The analysis failed. Please check the error message above.")
        st.stop()
    else:
        st.rerun()

if st.session_state.current_analysis_index is not None:
    res = st.session_state.analysis_history[st.session_state.current_analysis_index]
    if res.get("status") == "pending": st.info("⏳ Analysis is running..."); st.stop()
    st.header(f"Analysis for {res['company_name']} ({res['stock_ticker'].upper()})", divider="rainbow")
    st.subheader("Key Financial Data")
    financial_data_raw = res.get("financial_data", "")
    financials = parse_financial_data(financial_data_raw)
    if financials and financials.get("Current Price") != "N/A":
        cols = st.columns(4)
        price_val = get_first_available_value(financials, ["Current Price", "Previous Close"])
        cols[0].metric(label="Current Price", value=f"${float(price_val):.2f}" if price_val != "N/A" else "N/A")
        cols[1].metric(label="Market Cap", value=format_large_number(financials.get("Market Cap")))
        cols[2].metric(label="P/E Ratio", value=f"{float(financials.get('Trailing P/E')):.2f}" if financials.get('Trailing P/E') not in [None, "N/A"] else "N/A")
        cols[3].metric(label="EPS", value=f"{float(financials.get('Trailing EPS')):.2f}" if financials.get('Trailing EPS') not in [None, "N/A"] else "N/A")
    else:
        st.warning("Financial data is currently unavailable for this stock.")
        if financial_data_raw and financial_data_raw.startswith("Error:"): st.error(f"Details: {financial_data_raw}")

    tabs = st.tabs(["Final Summary", "Market Outlook", "Value Analysis", "Devil's Advocate", "Feedback"])
    with tabs[0]: display_analysis("Final Consensus Summary", res['company_name'], res.get("final_summary", ""), is_summary=True)
    with tabs[1]: display_analysis("AI-Generated Market Investor Outlook", res['company_name'], res.get("market_outlook", {}))
    with tabs[2]: display_analysis("AI-Generated Value Investor Analysis", res['company_name'], res.get("value_analysis", {}))
    with tabs[3]: display_analysis("AI-Generated Devil's Advocate View", res['company_name'], res.get("devils_advocate", {}))
    with tabs[4]:
        st.subheader("Was this analysis helpful?")
        # Feedback logic remains the same
        if res.get("feedback") is None and not res.get("error"):
            col1, col2, col3, col_spacer = st.columns([1,1,1,3])
            if col1.button("👍 Upvote", key=f"up_{res['date']}", use_container_width=True): res["feedback"] = "upvoted"; st.rerun()
            if col2.button("👎 Downvote", key=f"down_{res['date']}", use_container_width=True): res["feedback"] = "downvoted"; st.session_state.show_feedback_box = True; st.rerun()
            if col3.button("⚠️ Report Error", key=f"err_{res['date']}", use_container_width=True): res["feedback"] = "error"; st.rerun()
            if st.session_state.get('show_feedback_box', False) and res.get("feedback") == "downvoted":
                feedback_text = st.text_area("What could be improved in this analysis?", key=f"text_{res['date']}")
                if st.button("Submit Detailed Feedback", key=f"submit_{res['date']}"):
                    res["detailed_feedback"] = feedback_text; st.session_state.show_feedback_box = False; st.rerun()
        elif not res.get("error"): st.info("Thank you for your feedback on this report!")
        else: st.warning("Feedback is disabled for analyses that resulted in an error.")
else:
    st.info("Run a new analysis using the sidebar.")

st.header("Analysis History", divider="gray")
# History display logic remains the same
if not st.session_state.analysis_history: st.info("No previous analyses to display.")
else:
    for i, analysis in enumerate(st.session_state.analysis_history):
        with st.container(border=True):
            c1, c2, c3 = st.columns([4, 2, 2])
            c1.write(f"**{analysis['company_name']} ({analysis.get('stock_ticker', 'N/A')})**")
            c1.caption(f"_{analysis['date']}_")
            status = analysis.get("status")
            if status == "pending": c2.info("⏳ Pending...")
            elif status == "error": c2.error("❌ Failed")
            else:
                feedback = analysis.get("feedback")
                if feedback == "upvoted": c2.success("👍 Liked")
                elif feedback == "downvoted": c2.warning("👎 Disliked")
            if status in ["complete", "error"]:
                if c3.button("👁️ View", key=f"view_{i}", use_container_width=True): st.session_state.current_analysis_index = i; st.rerun()
            if c3.button("🗑️ Delete", key=f"del_{i}", use_container_width=True, type="secondary"):
                if st.session_state.current_analysis_index == i: st.session_state.current_analysis_index = None
                st.session_state.analysis_history.pop(i); st.rerun()

st.markdown("---")
st.info("**Disclaimer:** This tool is for informational and educational purposes only and does not constitute financial advice.")