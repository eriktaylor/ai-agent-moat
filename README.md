# AI Agent for Financial Moat Analysis

This project is an AI-powered agent that automates the initial phase of investment research by analyzing a company's competitive advantages, or "moat."

## Business Objective

Strategic analysis of a company's competitive advantages, or 'moat,' is a cornerstone of investment research. This process is traditionally manual and time-consuming, requiring analysts to sift through disparate sources like financial reports, news articles, and press releases. This project automates the intelligence-gathering and synthesis phase of this critical task, delivering a comprehensive analysis in minutes.

---

## What is an AI Agent?

Unlike a simple chatbot, the AI agent in this project is a more sophisticated system designed to achieve a complex goal. It operates with a distinct workflow:

* **A Reasoning Engine:** It uses a powerful Large Language Model (Google's Gemini 2.5 Flash) for reasoning and analysis.
* **A Set of Tools:** It has access to real-world information through tools like Google Search, a PDF-parsing web scraper, and the Yahoo Finance API.
* **A Workflow with Personas:** The agent gathers data through a multi-tiered search strategy that includes official news, critical reports, **and discussions from retail investor forums (like Reddit and Seeking Alpha)**. It then uses this data to reason from two distinct investor "personas"—a short-term **Market Investor** and a long-term **Value Investor**—to provide a balanced and nuanced analysis that contrasts professional and retail sentiment.

---

## The Role of RAG (Retrieval-Augmented Generation)

A key challenge with Large Language Models is that their knowledge is frozen at the time they were trained. To perform timely financial analysis, an agent needs access to up-to-the-minute information. This is where **Retrieval-Augmented Generation (RAG)** becomes essential.

RAG is the process of connecting an LLM to an external, live source of data. For this agent, the RAG pipeline works as follows:

1.  **Retrieve:** The agent uses its tools to fetch the latest news, financial data, and reports from the internet.
2.  **Augment:** This fresh information is compiled into a dynamic, "on-the-fly" knowledge base for the specific company being analyzed.
3.  **Generate:** The LLM is instructed to base its analysis *only* on this freshly provided context.

This RAG architecture is crucial for the agent's reliability. It grounds the LLM's powerful reasoning capabilities in real-time, verifiable facts, significantly reducing the risk of generating outdated or inaccurate information.

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
