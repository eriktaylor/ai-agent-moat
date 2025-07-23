import streamlit as st
import os
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
This tool automates the initial phase of investment research by analyzing a company's competitive advantages, or "moat."
Enter a company name and stock ticker in the sidebar to begin.
""")

# --- Sidebar for Inputs and API Keys ---
st.sidebar.header("Configuration")

# Using session state to manage API keys and run counts
if 'runs_today' not in st.session_state:
    st.session_state.runs_today = 0

# It's better practice for users of a public app to provide their own keys
google_api_key = st.sidebar.text_input("Enter your Google AI Studio API Key", type="password")
google_cse_id = st.sidebar.text_input("Enter your Google Custom Search Engine ID", type="password")

st.sidebar.markdown("---")
st.sidebar.header("Run Analysis")
company_name = st.sidebar.text_input("Enter Company Name", "NVIDIA")
stock_ticker = st.sidebar.text_input("Enter Stock Ticker", "NVDA")

run_button = st.sidebar.button("Run Analysis")

# --- Agent Initialization (Cached) ---
@st.cache_resource
def initialize_agent(api_key, cse_id):
    """
    Initializes the research agent and its components.
    This is cached to avoid re-loading models on every run.
    """
    # Set environment variables for the agent's tools to use
    os.environ['GOOGLE_API_KEY'] = api_key
    os.environ['GOOGLE_CSE_ID'] = cse_id
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2, google_api_key=api_key)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    return ResearchAgent(llm=llm, embeddings_model=embeddings)

# --- Main Application Logic ---
if run_button:
    # Validate inputs
    if not google_api_key or not google_cse_id:
        st.error("Please enter your Google API Key and CSE ID in the sidebar to proceed.")
    elif not company_name or not stock_ticker:
        st.error("Please enter a company name and stock ticker.")
    else:
        # --- Frontend Daily Limit Control ---
        MAX_RUNS_PER_DAY = 10 # Set a reasonable limit for the demo
        if st.session_state.runs_today >= MAX_RUNS_PER_DAY:
            st.warning("Daily demo limit reached. Please try again tomorrow.")
        else:
            st.session_state.runs_today += 1
            
            with st.spinner("Agent is initializing and conducting research... This may take a moment."):
                try:
                    # Initialize the agent
                    research_agent = initialize_agent(google_api_key, google_cse_id)

                    # --- 1. KEY FINANCIAL DATA ---
                    st.header(f"Analysis for {company_name} ({stock_ticker.upper()})")
                    financial_data_raw = get_stock_info.run(stock_ticker)
                    st.subheader("1. Key Financial Data")
                    st.text(financial_data_raw)

                    # --- 2. AI-GENERATED MARKET INVESTOR OUTLOOK ---
                    market_outlook_result = research_agent.generate_market_outlook(company_name, stock_ticker)
                    display_analysis("2. AI-GENERATED MARKET INVESTOR OUTLOOK", company_name, market_outlook_result)

                    # --- 3. AI-GENERATED VALUE INVESTOR ANALYSIS ---
                    value_analysis_result = research_agent.generate_value_analysis(company_name, stock_ticker)
                    display_analysis("3. AI-GENERATED VALUE INVESTOR ANALYSIS", company_name, value_analysis_result)

                    # --- 4. AI-GENERATED DEVIL'S ADVOCATE VIEW ---
                    devils_advocate_result = research_agent.generate_devils_advocate_view(company_name, stock_ticker)
                    display_analysis("4. AI-GENERATED DEVIL'S ADVOCATE VIEW", company_name, devils_advocate_result)

                    # --- 5. FINAL CONSENSUS SUMMARY ---
                    final_summary = research_agent.generate_final_summary(
                        market_outlook_result.get('answer', ''),
                        value_analysis_result.get('answer', ''),
                        devils_advocate_result.get('answer', '')
                    )
                    display_analysis("5. FINAL CONSENSUS SUMMARY", company_name, final_summary, is_summary=True)

                except Exception as e:
                    st.error(f"An error occurred during the analysis: {e}")

            # --- Feedback Mechanism ---
            st.markdown("---")
            st.subheader("Feedback")
            st.write("Was this analysis helpful?")

            if 'feedback_submitted' not in st.session_state:
                st.session_state.feedback_submitted = False

            if not st.session_state.feedback_submitted:
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("👍 Thumbs Up"):
                        # In a real app, you would log this to a database or analytics service
                        print(f"Feedback for {company_name}: Thumbs Up") 
                        st.success("Thanks for your feedback!")
                        st.session_state.feedback_submitted = True
                        st.rerun()

                with col2:
                    if st.button("👎 Thumbs Down"):
                        print(f"Feedback for {company_name}: Thumbs Down")
                        st.session_state.feedback_submitted = True
                        st.session_state.show_feedback_box = True
                        st.rerun()

            if st.session_state.get('show_feedback_box', False):
                feedback_text = st.text_area("What could be improved? (e.g., was the analysis inaccurate, unclear, etc.?)")
                if st.button("Submit Detailed Feedback"):
                    print(f"Detailed feedback for {company_name}: {feedback_text}")
                    st.success("Your detailed feedback has been submitted. Thank you!")
                    st.session_state.show_feedback_box = False
                    st.rerun()
            
            elif st.session_state.feedback_submitted:
                st.info("Thank you for your feedback!")

# --- Footer and Disclaimer ---
st.markdown("---")
st.info("""
**Disclaimer:** The content and analysis provided by this AI agent are for informational and educational purposes only and should not be construed as financial advice. The agent's output is generated based on publicly available data and is subject to the limitations of the underlying AI models. All investment decisions should be made with the guidance of a qualified financial professional.
""")
