# AI Agent for Financial Moat Analysis

This project is an AI-powered agent that automates the initial phase of investment research by analyzing a company's competitive advantages, or "moat."

## Business Objective

Strategic analysis of a company's competitive advantages, or 'moat,' is a cornerstone of investment research. This process is traditionally manual and time-consuming, requiring analysts to sift through disparate sources like financial reports, news articles, and press releases. This project automates the intelligence-gathering and synthesis phase of this critical task, delivering a comprehensive analysis in minutes.

---

## What is an AI Agent?

Unlike a simple chatbot, the AI agent in this project is a more sophisticated system designed to achieve a complex goal. It operates with a distinct workflow:

* **A Reasoning Engine:** It uses a powerful Large Language Model (Google's Gemini 2.5 Flash) for reasoning and analysis.
* **A Set of Tools:** It has access to real-world information through tools like Google Search, a PDF-parsing web scraper, and the Yahoo Finance API.
* **A Workflow with Personas:** The agent gathers data through a multi-tiered search strategy (official news, critical concerns, and deep-dive reports). It then uses this data to reason from two distinct investor "personas"—a short-term **Market Investor** and a long-term **Value Investor**—to provide a balanced and nuanced analysis.

---

## Technology Stack

* **LLM:** Google Gemini 2.5 Flash
* **Agent Framework:** LangChain
* **Embedding Model:** `all-MiniLM-L6-v2` (from Hugging Face)
* **Vector Store:** FAISS (for in-memory semantic search)
* **Data & Tools:**
    * Google Custom Search API
    * Yahoo Finance API (`yfinance`)
    * PDF Parsing (`PyMuPDF`)
    * Google Colab (for development and demonstration)

---

## How to Run

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/eriktaylor/ai-agent-moat.git](https://github.com/eriktaylor/ai-agent-moat.git)
    ```
2.  **Open in Google Colab:** Upload the `app.ipynb` notebook to Google Colab.
3.  **Set Up API Keys:** In the Colab Secrets Manager (found in the left sidebar), add your `GOOGLE_API_KEY` and `GOOGLE_CSE_ID`.
4.  **Run the Notebook:** Execute the cells in the `app.ipynb` notebook. The first cell will install all necessary dependencies, and the final cell will prompt you for a company name and stock ticker to begin the analysis.

