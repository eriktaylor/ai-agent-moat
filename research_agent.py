from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import GoogleSearchAPIWrapper
import re
import os

# Import the tools this agent will use
from tools import get_stock_info, scrape_website

class ResearchAgent:
    def __init__(self, llm, embeddings_model):
        self.llm = llm
        self.embeddings_model = embeddings_model
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        # The agent needs access to the API keys from the environment
        self.search_wrapper = GoogleSearchAPIWrapper(
            google_cse_id=os.environ.get('GOOGLE_CSE_ID'),
            google_api_key=os.environ.get('GOOGLE_API_KEY')
        )
        self.cache = {}

    def clear_cache(self):
        """Clears the agent's cache."""
        print("Cache cleared.")
        self.cache = {}

    def _get_context(self, entity_name, ticker):
        """Gathers context and returns financial data and unstructured text separately."""
        context_cache_key = f"context_{entity_name}_{ticker}"
        if context_cache_key in self.cache:
            print("Returning cached context.")
            return self.cache[context_cache_key]

        # Get financial data
        financial_data = "No financial data available."
        if ticker:
            print(f"--- Getting Financial Data for {ticker} ---")
            financial_data_result = get_stock_info.run(ticker)
            if not financial_data_result.startswith("Error"):
                financial_data = financial_data_result
                print("Successfully collected financial data.")

        # Gather unstructured text
        unstructured_text_list = []
        print("--- Tier 1: Official News & Analysis ---")
        headline_query = f'"{entity_name}" recent news'
        headline_results = self.search_wrapper.results(headline_query, num_results=3)
        if headline_results:
            unstructured_text_list.extend([f"Headline: {r.get('title', '')}\nSnippet: {r.get('snippet', '')}" for r in headline_results])
            print(f"Collected {len(headline_results)} headlines and snippets.")
        
        print("--- Tier 2: Critical News & Sentiment ---")
        critical_query = f'\"{entity_name}\" issues OR concerns OR investigation OR recall OR safety OR "short interest"'
        critical_results = self.search_wrapper.results(critical_query, num_results=3)
        if critical_results:
            unstructured_text_list.extend([f"Critical Headline: {r.get('title', '')}\nSnippet: {r.get('snippet', '')}" for r in critical_results])
            print(f"Collected {len(critical_results)} critical headlines and snippets.")

        print("--- Tier 3: Deep Dive ---")
        deep_dive_query = f'\"{entity_name}\" market analysis OR in-depth report filetype:pdf OR site:globenewswire.com OR site:prnewswire.com'
        deep_dive_results = self.search_wrapper.results(deep_dive_query, num_results=2)
        if deep_dive_results:
            urls = [result['link'] for result in deep_dive_results if 'link' in result]
            for url in urls:
                print(f"Scraping {url}...")
                content = scrape_website.run(url)
                if content and not content.startswith("Error"):
                    unstructured_text_list.append(content)
                    print(f"Successfully scraped content from {url}")
                else:
                    print(content)
        
        unstructured_corpus = "\n\n---\n\n".join(unstructured_text_list)
        self.cache[context_cache_key] = (financial_data, unstructured_corpus)
        return financial_data, unstructured_corpus

    def _create_rag_chain(self, system_prompt, unstructured_corpus):
        """Helper to create a RAG chain with a specific prompt."""
        docs = self.text_splitter.split_text(unstructured_corpus)
        vector_store = FAISS.from_texts(texts=docs, embedding=self.embeddings_model)
        retriever = vector_store.as_retriever()
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        return create_retrieval_chain(retriever, question_answer_chain)

    def generate_market_outlook(self, entity_name, ticker):
        print("\nGenerating Market Investor Outlook...")
        financial_data, unstructured_corpus = self._get_context(entity_name, ticker)
        if not unstructured_corpus:
            return "Could not gather context for outlook generation."

        system_prompt = (
            "You are a 'Market Investor' analyst. Here is the key financial data for the company:\n{financial_data}\n\n" 
            "Now, using the retrieved context below (which includes both positive and critical news), generate a report. "
            "The report MUST be structured with the following sections:\n"
            "1. **Market Sentiment:** Synthesize the official news and the critical news to determine the overall market sentiment. Is it bullish, bearish, or mixed? Why?\n"
            "2. **Valuation Analysis:** Is the stock considered expensive or cheap? You MUST reference the 'Trailing P/E' and 'Trailing EPS' from the financial data. If P/E is not applicable because EPS is negative, state this clearly and explain what a negative EPS implies for valuation.\n"
            "3. **Relative Performance (Implied):** Based on the context, how does this company's performance and outlook seem to compare to its peers or the broader market?"
            "\n\nRetrieved Context:\n{context}\n\n"
            "DO NOT give financial advice. This is an objective summary of the data provided."
        )
        rag_chain = self._create_rag_chain(system_prompt, unstructured_corpus)
        response = rag_chain.invoke({"input": f"Market outlook for {entity_name}", "financial_data": financial_data})
        return response['answer']

    def generate_value_analysis(self, entity_name, ticker):
        print("\nGenerating Value Investor Analysis...")
        financial_data, unstructured_corpus = self._get_context(entity_name, ticker)
        if not unstructured_corpus:
            return "Could not gather context for value analysis."

        system_prompt = (
            "You are a 'Value Investor' analyst. Here is the key financial data for the company:\n{financial_data}\n\n" 
            "Now, using the retrieved context below (which includes both positive and critical news), generate a detailed business brief. "
            "The report MUST be structured with the following sections:\n"
            "1. **Valuation Summary:** Start by stating if the company appears 'Overvalued', 'Undervalued', or 'Fairly Valued'. Justify your conclusion briefly by referencing the P/E or EPS from the financial data.\n"
            "2. **SWOT Analysis:** A detailed, bulleted list of the company's Strengths, Weaknesses, Opportunities, and Threats. You MUST incorporate information from the 'Critical News' headlines in the Weaknesses and Threats sections.\n"
            "3. **Competitive Moat:** Based on the SWOT analysis, describe the company's long-term competitive advantages. Is its moat wide, narrow, or degrading? You MUST consider the threats and weaknesses when assessing the durability of the moat."
            "\n\nRetrieved Context:\n{context}"
        )
        rag_chain = self._create_rag_chain(system_prompt, unstructured_corpus)
        response = rag_chain.invoke({"input": f"Value analysis for {entity_name}", "financial_data": financial_data})
        return response['answer']
