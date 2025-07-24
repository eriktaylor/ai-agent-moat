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
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'current_analysis_index' not in st.session_state:
    st.session_state.current_analysis_index = None

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
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2, google_api_key=api_key)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    from langchain_community.tools import DuckDuckGoSearchRun
    search_tool = DuckDuckGoSearchRun()

    return ResearchAgent(llm=llm, embeddings_model=embeddings, search_tool=search_tool)

# --- Main Application Logic ---

def run_full_analysis(company_name, stock_ticker):
    with st.spinner("Agent is conducting research... This may take a moment."):
        try:
            research_agent = initialize_agent(google_api_key)
            
            analysis_results = {
                "company_name": company_name,
                "stock_ticker": stock_ticker,
                "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "error": False,
                "feedback": None,
                "detailed_feedback": None
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
            
            # Add the full analysis to the history and set it as the current one
            st.session_state.analysis_history.insert(0, analysis_results)
            st.session_state.current_analysis_index = 0

        except Exception as e:
            st.error(f"An error occurred during the analysis: {e}")
            st.session_state.current_analysis_index = None


# --- UI Display ---

# 1. Previous Analyses Section
st.header("Previous Analyses")
if not st.session_state.analysis_history:
    st.info("No previous analyses to display. Run a new analysis to get started.")
else:
    for i, analysis in enumerate(st.session_state.analysis_history):
        col1, col2, col3, col4 = st.columns([3, 2, 1, 1])
        with col1:
            st.write(f"**{analysis['company_name']} ({analysis['stock_ticker']})**")
        with col2:
            st.write(f"_{analysis['date']}_")
        with col3:
            # <<< CHANGE: Display feedback status >>>
            feedback = analysis.get("feedback")
            if feedback == "upvoted":
                st.success("👍 Liked")
            elif feedback == "downvoted":
                st.warning("👎 Disliked")
            elif feedback == "error":
                st.error("⚠️ Error")

        with col4:
            # <<< CHANGE: View button now sets the index to display cached results >>>
            if st.button("View", key=f"view_{i}"):
                st.session_state.current_analysis_index = i
                st.rerun()


# 2. Main Analysis Section
if run_button:
    if not google_api_key:
        st.error("Please enter your Google Gemini API Key in the sidebar to proceed.")
    elif not company_name_input or not stock_ticker_input:
        st.error("Please enter a company name and stock ticker.")
    else:
        st.session_state.current_analysis_index = None # Clear previous analysis view
        run_full_analysis(company_name_input, stock_ticker_input)
        st.rerun()

# This block ensures the analysis stays on screen after it's run or when "View" is clicked
if st.session_state.current_analysis_index is not None:
    # Ensure the index is valid
    if st.session_state.current_analysis_index < len(st.session_state.analysis_history):
        res = st.session_state.analysis_history[st.session_state.current_analysis_index]
        
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
                st.error("Thank you for reporting an error. This analysis has been marked.")
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
