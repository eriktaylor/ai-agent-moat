import streamlit as st
import datetime
import re
import locale
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
st.title("🤖 AI Agent for Financial Moat Analysis")
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
    """Parses the financial data string into a dictionary."""
    data = {}
    if not isinstance(data_string, str) or data_string.startswith("Error"):
        return data
    pattern = re.compile(r"([^:]+):\s*(.*)")
    for line in data_string.split('\n'):
        match = pattern.match(line)
        if match:
            key = match.group(1).strip()
            value = match.group(2).strip()
            data[key] = value
    return data

def format_large_number(value_str):
    """Formats a number string into a human-readable currency format (e.g., $2.5T, $100B)."""
    if value_str is None or value_str == "N/A":
        return "N/A"
    try:
        num = float(value_str)
        if num >= 1_000_000_000_000:
            return f"${num / 1_000_000_000_000:.2f}T"
        elif num >= 1_000_000_000:
            return f"${num / 1_000_000_000:.2f}B"
        elif num >= 1_000_000:
            return f"${num / 1_000_000:.2f}M"
        else:
            return f"${num:,.2f}"
    except (ValueError, TypeError):
        return "N/A"

def get_first_available_value(data_dict, keys_to_try):
    """Iterates through a list of keys and returns the first found value."""
    for key in keys_to_try:
        if key in data_dict and data_dict[key] not in [None, "N/A", "None"]:
            return data_dict[key]
    return "N/A"


# --- Sidebar for Inputs and API Keys ---
with st.sidebar:
    st.header("⚙️ Configuration")

    google_api_key = st.text_input(
        "Enter your Google Gemini API Key",
        type="password",
        help="Get your free API key from [Google AI Studio](https://aistudio.google.com/app/apikey)."
    )
    st.markdown("[Get your free Gemini API key here](https://aistudio.google.com/app/apikey)")

    st.markdown("---")
    st.header("▶️ Run New Analysis")
    company_name_input = st.text_input("Enter Company Name", "Apple")
    stock_ticker_input = st.text_input("Enter Stock Ticker", "AAPL")

    run_button = st.button("Run New Analysis", use_container_width=True, type="primary")

# --- Agent Initialization (Cached) ---
@st.cache_resource
def initialize_agent(api_key):
    """Initializes the ResearchAgent."""
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2, google_api_key=api_key)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    from langchain_community.tools import DuckDuckGoSearchRun
    search_tool = DuckDuckGoSearchRun()

    return ResearchAgent(llm=llm, embeddings_model=embeddings, search_tool=search_tool)

# --- Main Application Logic ---
def run_full_analysis(company_name, stock_ticker):
    """Runs the full analysis pipeline and returns the results dictionary."""
    base_results = {
        "company_name": company_name,
        "stock_ticker": stock_ticker,
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "error": False,
        "feedback": None,
        "detailed_feedback": None,
        "financial_data": None,
        "market_outlook": {"answer": "Analysis could not be run due to data retrieval errors."},
        "value_analysis": {"answer": "Analysis could not be run due to data retrieval errors."},
        "devils_advocate": {"answer": "Analysis could not be run due to data retrieval errors."},
        "final_summary": "Analysis could not be run due to data retrieval errors."
    }

    try:
        with st.spinner(f"Running analysis for {company_name}... This may take a moment."):
            research_agent = initialize_agent(google_api_key)
            
            financial_data_str = get_stock_info.run(stock_ticker)
            base_results["financial_data"] = financial_data_str

            if financial_data_str.startswith("Error"):
                st.error(f"Failed to retrieve financial data for {stock_ticker}. The analysis cannot proceed. Details: {financial_data_str}")
                base_results["status"] = "error"
                base_results["error"] = True
                return base_results

            base_results["market_outlook"] = research_agent.generate_market_outlook(company_name, stock_ticker)
            base_results["value_analysis"] = research_agent.generate_value_analysis(company_name, stock_ticker)
            base_results["devils_advocate"] = research_agent.generate_devils_advocate_view(company_name, stock_ticker)
            
            # <<< BUG FIX: Correctly passing company_name as the first argument >>>
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
        base_results["status"] = "error"
        base_results["error"] = True
        return base_results


# --- UI Display Logic ---

if run_button:
    if not google_api_key:
        st.error("Please enter your Google Gemini API Key in the sidebar to proceed.")
    elif not company_name_input or not stock_ticker_input:
        st.error("Please enter a company name and stock ticker.")
    else:
        placeholder = {
            "company_name": company_name_input,
            "stock_ticker": stock_ticker_input,
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "pending"
        }
        st.session_state.analysis_history.insert(0, placeholder)
        st.session_state.current_analysis_index = 0
        st.rerun()

pending_analysis_index = next((i for i, an in enumerate(st.session_state.analysis_history) if an.get("status") == "pending"), None)

if pending_analysis_index is not None:
    pending_info = st.session_state.analysis_history[pending_analysis_index]
    full_results = run_full_analysis(pending_info["company_name"], pending_info["stock_ticker"])
    st.session_state.analysis_history[pending_analysis_index] = full_results
    st.session_state.current_analysis_index = pending_analysis_index
    st.rerun()


# --- Main Analysis Display Section ---
if st.session_state.current_analysis_index is not None:
    res = st.session_state.analysis_history[st.session_state.current_analysis_index]

    if res.get("status") == "pending":
        st.info("⏳ Analysis is running...")
        st.stop()

    st.header(f"Analysis for {res['company_name']} ({res['stock_ticker'].upper()})", divider="rainbow")

    st.subheader("Key Financial Data")
    financial_data_raw = res.get("financial_data", "")
    financials = parse_financial_data(financial_data_raw)
    
    if financials:
        cols = st.columns(4)
        
        price_keys_to_try = ["Current Price", "Regular Market Price", "Previous Close", "Open"]
        price_val = get_first_available_value(financials, price_keys_to_try)
        try:
            price_str = f"${float(price_val):.2f}"
        except (ValueError, TypeError):
            price_str = "N/A"
        cols[0].metric(label="Current Price", value=price_str)
        
        market_cap_val = financials.get("Market Cap", "N/A")
        cols[1].metric(label="Market Cap", value=format_large_number(market_cap_val))

        pe_ratio_val = financials.get("Trailing P/E", "N/A")
        try:
            pe_ratio_str = f"{float(pe_ratio_val):.2f}"
        except (ValueError, TypeError):
            pe_ratio_str = "N/A"
        cols[2].metric(label="P/E Ratio", value=pe_ratio_str)
        
        eps_val = financials.get("Trailing EPS", "N/A")
        try:
            eps_str = f"{float(eps_val):.2f}"
        except (ValueError, TypeError):
            eps_str = "N/A"
        cols[3].metric(label="EPS", value=eps_str)

    else:
        st.warning("Financial data could not be retrieved for this analysis.")
        if financial_data_raw:
            st.error(f"Details: {financial_data_raw}")


    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Final Summary", "Market Outlook", "Value Analysis", "Devil's Advocate", "Feedback"
    ])

    with tab1:
        display_analysis("Final Consensus Summary", res['company_name'], res.get("final_summary", "Not available"), is_summary=True)
    with tab2:
        display_analysis("AI-Generated Market Investor Outlook", res['company_name'], res.get("market_outlook", {}))
    with tab3:
        display_analysis("AI-Generated Value Investor Analysis", res['company_name'], res.get("value_analysis", {}))
    with tab4:
        display_analysis("AI-Generated Devil's Advocate View", res['company_name'], res.get("devils_advocate", {}))
    with tab5:
        st.subheader("Was this analysis helpful?")
        if res.get("feedback") is None and not res.get("error"):
            col1, col2, col3, col_spacer = st.columns([1,1,1,3])

            if col1.button("👍 Upvote", key=f"up_{res['date']}", use_container_width=True):
                res["feedback"] = "upvoted"
                st.toast("Thanks for your feedback!", icon="👍")
                st.rerun()

            if col2.button("👎 Downvote", key=f"down_{res['date']}", use_container_width=True):
                res["feedback"] = "downvoted"
                st.session_state.show_feedback_box = True
                st.toast("Thanks! Please provide details.", icon="👎")
                st.rerun()

            if col3.button("⚠️ Report Error", key=f"err_{res['date']}", use_container_width=True):
                res["feedback"] = "error"
                st.error("Error reported. Thank you for helping improve the agent.")
                st.rerun()

            if st.session_state.get('show_feedback_box', False) and res.get("feedback") == "downvoted":
                feedback_text = st.text_area("What could be improved in this analysis?", key=f"text_{res['date']}")
                if st.button("Submit Detailed Feedback", key=f"submit_{res['date']}"):
                    res["detailed_feedback"] = feedback_text
                    st.success("Your detailed feedback has been submitted. Thank you!")
                    st.session_state.show_feedback_box = False
                    st.rerun()
        elif not res.get("error"):
            st.info("Thank you for your feedback on this report!")
            if res.get("detailed_feedback"):
                st.write("Your detailed feedback:")
                st.info(res.get("detailed_feedback"))
        else:
            st.warning("Feedback is disabled for analyses that resulted in an error.")

else:
    st.info("Run a new analysis using the sidebar, or view a previous analysis from the history below.")


# --- Previous Analyses Section ---
st.header("Analysis History", divider="gray")
if not st.session_state.analysis_history:
    st.info("No previous analyses to display. Run a new analysis to get started.")
else:
    for i, analysis in enumerate(st.session_state.analysis_history):
        with st.container(border=True):
            col1, col2, col3 = st.columns([4, 2, 2])
            with col1:
                st.write(f"**{analysis['company_name']} ({analysis.get('stock_ticker', 'N/A')})**")
                st.caption(f"_{analysis['date']}_")
            with col2:
                status = analysis.get("status")
                if status == "pending":
                    st.info("⏳ Pending...")
                elif status == "error":
                    st.error("❌ Failed")
                else:
                    feedback = analysis.get("feedback")
                    if feedback == "upvoted":
                        st.success("👍 Liked")
                    elif feedback == "downvoted":
                        st.warning("👎 Disliked")
                    elif feedback == "error":
                        st.error("⚠️ Error Reported")
                    else:
                        st.write("")
            with col3:
                b_col1, b_col2 = st.columns(2)
                if status == "complete" or status == "error":
                    if b_col1.button("👁️ View", key=f"view_{i}", use_container_width=True):
                        st.session_state.current_analysis_index = i
                        st.session_state.show_feedback_box = False
                        st.rerun()
                if b_col2.button("🗑️ Delete", key=f"del_{i}", use_container_width=True):
                    if st.session_state.current_analysis_index == i:
                        st.session_state.current_analysis_index = None
                    st.session_state.analysis_history.pop(i)
                    st.rerun()

# --- Footer ---
st.markdown("---")
st.info("""
**Disclaimer:** This tool is for informational and educational purposes only and does not constitute financial advice. All investment decisions should be made with the guidance of a qualified financial professional.
""")
