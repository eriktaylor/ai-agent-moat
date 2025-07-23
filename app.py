import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from research_agent import ResearchAgent
from tools import get_stock_info
from display_utils import display_analysis
# <<< CHANGE: Import the DuckDuckGo search tool >>>
from langchain_community.tools import DuckDuckGoSearchRun

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
st.warning("This is a public demo using a free search tool. Results may be less comprehensive than the full version. The daily usage limit is shared among all users.", icon="⚠️")


# --- Sidebar for Inputs ---
st.sidebar.header("Run Analysis")
company_name = st.sidebar.text_input("Enter Company Name", "NVIDIA")
stock_ticker = st.sidebar.text_input("Enter Stock Ticker", "NVDA")

run_button = st.sidebar.button("Run Analysis")

# --- Agent Initialization (Cached) ---
@st.cache_resource
def initialize_agent():
    """
    Initializes the research agent and its components.
    This is cached to avoid re-loading models on every run.
    """
    # <<< CHANGE: Use Streamlit's secrets for your own keys >>>
    # You will set these in your Streamlit Cloud settings
    google_api_key = st.secrets.get("GOOGLE_API_KEY")
    
    if not google_api_key:
        st.error("Google API Key is not configured. Please contact the app administrator.")
        st.stop()

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2, google_api_key=google_api_key)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # <<< CHANGE: Initialize the agent with the keyless DuckDuckGo search tool >>>
    search_tool = DuckDuckGoSearchRun()
    
    return ResearchAgent(llm=llm, embeddings_model=embeddings, search_tool=search_tool)

# --- Main Application Logic ---
if run_button:
    if not company_name or not stock_ticker:
        st.error("Please enter a company name and stock ticker.")
    else:
        with st.spinner("Agent is initializing and conducting research... This may take a moment."):
            try:
                # Initialize the agent
                research_agent = initialize_agent()

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

# --- Footer and Disclaimer ---
st.markdown("---")
st.info("""
**Disclaimer:** The content and analysis provided by this AI agent are for informational and educational purposes only and should not be construed as financial advice. The agent's output is generated based on publicly available data and is subject to the limitations of the underlying AI models. All investment decisions should be made with the guidance of a qualified financial professional.
""")
