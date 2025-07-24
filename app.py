import streamlit as st
import os
import datetime
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
# This is crucial for making the app feel persistent.
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'current_analysis' not in st.session_state:
    st.session_state.current_analysis = None

# --- Sidebar for Inputs and API Keys ---
st.sidebar.header("Configuration")

google_api_key = st.sidebar.text_input(
    "Enter your Google Gemini API Key", 
    type="password",
    help="Get your free API key from [Google AI Studio](https://aistudio.google.com/app/apikey)."
)
st.sidebar.markdown("[Get your free Gemini API key here](https://aistudio.google.com/app/apikey)")

st.sidebar.markdown("---")
st.sidebar.header("Run New Analysis")
company_name_input = st.sidebar.text_input("Enter Company Name", "NVIDIA")
stock_ticker_input = st.sidebar.text_input("Enter Stock Ticker", "NVDA")

run_button = st.sidebar.button("Run New Analysis")

# --- Agent Initialization (Cached) ---
@st.cache_resource
def initialize_agent(api_key):
    """
    Initializes the research agent and its components.
    This is cached to avoid re-loading models on every run.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2, google_api_key=api_key)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    search_tool = st.secrets.get("SEARCH_TOOL", "duckduckgo") # Default to duckduckgo if not specified
    if search_tool == "google":
        # In a real app, you might have a selector for different search tools
        from langchain_community.utilities import GoogleSearchAPIWrapper
        search_tool_instance = GoogleSearchAPIWrapper(google_cse_id=st.secrets["GOOGLE_CSE_ID"], google_api_key=api_key)
    else:
        from langchain_community.tools import DuckDuckGoSearchRun
        search_tool_instance = DuckDuckGoSearchRun()

    return ResearchAgent(llm=llm, embeddings_model=embeddings, search_tool=search_tool_instance)

# --- Main Application Logic ---

# Function to run the full analysis
def run_full_analysis(company_name, stock_ticker):
    with st.spinner("Agent is conducting research... This may take a moment."):
        try:
            research_agent = initialize_agent(google_api_key)
            
            # This dictionary will hold all the results
            analysis_results = {
                "company_name": company_name,
                "stock_ticker": stock_ticker,
                "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "error": False,
                "feedback": None
            }

            analysis_results["financial_data"] = get_stock_info.run(stock_ticker)
            analysis_results["market_outlook"] = research_agent.generate_market_outlook(company_name, stock_ticker)
            analysis_results["value_analysis"] = research_agent.generate_value_analysis(company_name, stock_ticker)
            analysis_results["devils_advocate"] = research_agent.generate_devils_advocate_view(company_name, stock_ticker)
            
            analysis_results["final_summary"] = research_agent.generate_final_summary(
                analysis_results["market_outlook"].get('answer', ''),
                analysis_results["value_analysis"].get('answer', ''),
                analysis_results["devils_advocate"].get('answer', '')
            )
            
            st.session_state.current_analysis = analysis_results
            # Add a summary to the history
            history_summary = {k: analysis_results[k] for k in ('company_name', 'stock_ticker', 'date', 'final_summary')}
            st.session_state.analysis_history.insert(0, history_summary) # Insert at the beginning

        except Exception as e:
            st.error(f"An error occurred during the analysis: {e}")
            st.session_state.current_analysis = {"error": True}


# --- UI Display ---

# 1. Previous Analyses Section
st.header("Previous Analyses")
if not st.session_state.analysis_history:
    st.info("No previous analyses to display. Run a new analysis to get started.")
else:
    for i, analysis in enumerate(st.session_state.analysis_history):
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            st.write(f"**{analysis['company_name']} ({analysis['stock_ticker']})**")
        with col2:
            st.write(f"_{analysis['date']}_")
        with col3:
            if st.button("View", key=f"view_{i}"):
                # Find the full analysis in the agent's cache to display
                full_analysis_key = f"context_{analysis['company_name']}_{analysis['stock_ticker']}"
                # This part is a bit tricky without a proper database, we'll re-run for the demo
                # In a real app, you'd fetch the full saved report from a DB
                st.session_state.current_analysis = None # Clear current to force re-display
                run_full_analysis(analysis['company_name'], analysis['stock_ticker'])
                st.rerun()


# 2. Main Analysis Section
if run_button:
    if not google_api_key:
        st.error("Please enter your Google Gemini API Key in the sidebar to proceed.")
    elif not company_name_input or not stock_ticker_input:
        st.error("Please enter a company name and stock ticker.")
    else:
        st.session_state.current_analysis = None # Clear previous analysis
        run_full_analysis(company_name_input, stock_ticker_input)
        st.rerun()

# This block ensures the analysis stays on screen after it's run
if st.session_state.current_analysis and not st.session_state.current_analysis.get("error"):
    res = st.session_state.current_analysis
    
    st.header(f"Analysis for {res['company_name']} ({res['stock_ticker'].upper()})")
    
    st.subheader("1. Key Financial Data")
    st.text(res["financial_data"])

    display_analysis("2. AI-GENERATED MARKET INVESTOR OUTLOOK", res['company_name'], res["market_outlook"])
    display_analysis("3. AI-GENERATED VALUE INVESTOR ANALYSIS", res['company_name'], res["value_analysis"])
    display_analysis("4. AI-GENERATED DEVIL'S ADVOCATE VIEW", res['company_name'], res["devils_advocate"])
    display_analysis("5. FINAL CONSENSUS SUMMARY", res['company_name'], res["final_summary"], is_summary=True)

    # --- Feedback Mechanism ---
    st.markdown("---")
    st.subheader("Feedback")

    if res.get("feedback") is None:
        st.write("Was this analysis helpful?")
        col1, col2, col3, col4 = st.columns(4)
        
        if col1.button("👍 Upvote"):
            res["feedback"] = "upvoted"
            print(f"Feedback for {res['company_name']}: Upvoted")
            st.success("Thanks for your feedback!")
            st.rerun()

        if col2.button("👎 Downvote"):
            res["feedback"] = "downvoted"
            print(f"Feedback for {res['company_name']}: Downvoted")
            st.warning("Thanks for your feedback! Please provide more details below.")
            st.session_state.show_feedback_box = True
            st.rerun()

        if col3.button("⚠️ Report Error"):
            res["feedback"] = "error"
            print(f"ERROR REPORTED for {res['company_name']}")
            # Remove the errored analysis from history
            st.session_state.analysis_history = [h for h in st.session_state.analysis_history if h['date'] != res['date']]
            st.error("Thank you for reporting an error. This analysis has been removed from the history.")
            st.session_state.current_analysis = None
            st.rerun()

        if st.session_state.get('show_feedback_box', False):
            feedback_text = st.text_area("What could be improved?")
            if st.button("Submit Detailed Feedback"):
                res["detailed_feedback"] = feedback_text
                print(f"Detailed feedback for {res['company_name']}: {feedback_text}")
                st.success("Your detailed feedback has been submitted. Thank you!")
                st.session_state.show_feedback_box = False
                st.rerun()
    else:
        st.info("Thank you for your feedback on this report!")

# --- Footer ---
st.markdown("---")
st.info("""
**Disclaimer:** This tool is for informational and educational purposes only and does not constitute financial advice. All investment decisions should be made with the guidance of a qualified financial professional.
""")
