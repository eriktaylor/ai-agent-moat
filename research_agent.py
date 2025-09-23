from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
import re
import os
from operator import itemgetter

# Import the tools this agent will use
from tools import get_stock_info, scrape_website

class ResearchAgent:
    def __init__(self, llm, embeddings_model, search_tool):
        self.llm = llm
        self.embeddings_model = embeddings_model
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        self.search_tool = search_tool
        self.cache = {}

    def clear_cache(self):
        """Clears the agent's cache."""
        print("Cache cleared.")
        self.cache = {}

    def _get_context(self, entity_name, ticker):
        """Gathers context and returns financial data and a list of source documents."""
        context_cache_key = f"context_{entity_name}_{ticker}"
        if context_cache_key in self.cache:
            print("Returning cached context.")
            return self.cache[context_cache_key]

        financial_data_result = get_stock_info.run(ticker)
        if financial_data_result.startswith("Error"):
            financial_data = "Financial data is currently unavailable."
        else:
            financial_data = financial_data_result
            print("Successfully collected financial data.")

        source_documents = []
        
        if not self.search_tool:
            print("--- Web search tool not available, skipping web search. ---")
            self.cache[context_cache_key] = (financial_data, source_documents)
            return financial_data, source_documents

        queries = {
            "Official News & Analysis": f'"{entity_name}" recent news',
            "Critical News & Sentiment": f'\"{entity_name}\" issues OR concerns OR investigation OR recall OR safety OR "short interest"',
            "Retail Investor Sentiment": f'site:reddit.com/r/wallstreetbets OR site:reddit.com/r/stocks OR site:seekingalpha.com OR site:fool.com "{entity_name}" OR "{ticker}"',
        }

        for tier, query in queries.items():
            print(f"--- Searching for: {tier} ---")
            try:
                search_result_text = self.search_tool.run(query)
                if search_result_text and "No good search result found" not in search_result_text:
                    doc = Document(page_content=search_result_text, metadata={"source": "DuckDuckGo Search", "title": f"{tier} for {entity_name}"})
                    source_documents.append(doc)
                    print(f"Collected results for {tier}.")
                else:
                    print(f"No good search results for {tier}.")
            except Exception as e:
                print(f"An error occurred during web search for tier '{tier}': {e}")
                continue
        
        self.cache[context_cache_key] = (financial_data, source_documents)
        return financial_data, source_documents

    def _create_rag_chain(self, system_prompt, source_documents):
        if not source_documents:
            def no_rag_chain(input_data):
                prompt = ChatPromptTemplate.from_template(system_prompt).format(
                    financial_data=input_data.get("financial_data", "N/A"),
                    context_formatted="No web context was available for this analysis."
                )
                response = self.llm.invoke(prompt)
                return {"answer": response, "sources": []}
            return no_rag_chain

        vector_store = FAISS.from_documents(documents=source_documents, embedding=self.embeddings_model)
        retriever = vector_store.as_retriever()

        def format_docs_with_citations(docs):
            return "\n\n".join([f"[Source {i+1}]: Title: {doc.metadata.get('title', 'N/A')}\nContent: {doc.page_content}" for i, doc in enumerate(docs)])

        rag_chain = (
            {
                "context_docs": itemgetter("input") | retriever,
                "input": itemgetter("input"),
                "financial_data": itemgetter("financial_data"),
            }
            | RunnablePassthrough.assign(context_formatted=lambda x: format_docs_with_citations(x["context_docs"]))
            | { "answer": ChatPromptTemplate.from_template(system_prompt) | self.llm, "sources": itemgetter("context_docs") }
        )
        return rag_chain

    def _run_analysis(self, entity_name, ticker, system_prompt, query_input):
        financial_data, source_documents = self._get_context(entity_name, ticker)
        rag_chain = self._create_rag_chain(system_prompt, source_documents)
        response = rag_chain.invoke({"input": query_input, "financial_data": financial_data})
        if isinstance(response, dict):
             return {"answer": response['answer'].content, "sources": response['sources']}
        else:
             return {"answer": response.content, "sources": []}

    def generate_market_outlook(self, entity_name, ticker):
        print("\nGenerating Market Investor Outlook...")
        system_prompt = (
            "You are a 'Market Investor' analyst. Key financial data:\n{financial_data}\n\n" 
            "Using the retrieved context below, generate a report with these sections:\n"
            "1. **Professional Market Sentiment:** Based on official news/reports.\n"
            "2. **Retail Investor Sentiment:** Based on forum snippets.\n"
            "3. **Valuation Analysis:** Is it expensive or cheap? Reference 'Trailing P/E' and 'Trailing EPS'.\n"
            "**Cite sources with [1], [2] at the end of sentences.**\n\nContext:\n{context_formatted}\n\n"
            "This is an objective summary, not financial advice. "
            "Do not insert line breaks inside words or numbers. Keep paragraphs wrapped normally."
        )
        return self._run_analysis(entity_name, ticker, system_prompt, f"Market outlook for {entity_name}")

    def generate_value_analysis(self, entity_name, ticker):
        print("\nGenerating Value Investor Analysis...")
        system_prompt = (
            "You are a 'Value Investor' analyst. Key financial data:\n{financial_data}\n\n" 
            "Using the retrieved context below, generate a business brief with these sections:\n"
            "1. **Valuation Summary:** State if it's 'Overvalued', 'Undervalued', or 'Fairly Valued', justifying with data.\n"
            "2. **SWOT Analysis:** Bulleted Strengths, Weaknesses, Opportunities, Threats.\n"
            "3. **Competitive Moat:** Describe its long-term advantages.\n"
            "**Cite sources with [1], [2] at the end of sentences.**\n\nContext:\n{context_formatted}\n\n"
            "Do not insert line breaks inside words or numbers. Keep paragraphs wrapped normally."
        )
        return self._run_analysis(entity_name, ticker, system_prompt, f"Value analysis for {entity_name}")

    def generate_devils_advocate_view(self, entity_name, ticker):
        print("\nGenerating Devil's Advocate View...")
        system_prompt = (
            "You are a skeptical 'Devil's Advocate' analyst. Your purpose is to challenge the bullish thesis. Key financial data:\n{financial_data}\n\n" 
            "Using the retrieved context below, identify the single strongest counter-argument or hidden risk in a concise paragraph.\n"
            "**Cite the source of your information with [1], [2] at the end of the sentence.**\n\nContext:\n{context_formatted}\n\n"
            "Do not insert line breaks inside words or numbers. Keep paragraphs wrapped normally."
        )
        return self._run_analysis(entity_name, ticker, system_prompt, f"Strongest bearish case against {entity_name}?")

    def generate_final_summary(self, entity_name, market_outlook, value_analysis, devils_advocate):
        print("\nGenerating Final Consensus Summary...")
        combined_analysis = (
            f"--- Market Investor Outlook ---\n{market_outlook}\n\n"
            f"--- Value Investor Analysis ---\n{value_analysis}\n\n"
            f"--- Devil's Advocate View ---\n{devils_advocate}"
        )
        # --- FIX: Replaced prompt with a version that includes a clear rubric ---
        system_prompt = (
            "You are a 'Lead Analyst' synthesizing your team's views for {entity_name}.\n"
            "Return exactly two sections in valid Markdown:\n"
            "1. **Consensus Rating:** Choose exactly one: **Bullish**, **Bearish**, or **Neutral with Caution**.\n"
            "   • Choose **Bullish** if BOTH (A) overall sentiment is net positive AND (B) valuation is not clearly overextended.\n"
            "   • Choose **Bearish** if EITHER (A) major red flags/negative catalysts are present OR (B) valuation is extreme and unsupported.\n"
            "   • Choose **Neutral with Caution** ONLY if evidence is genuinely mixed or insufficient.\n"
            "   • If evidence is mixed but slightly positive, prefer **Bullish**; if mixed but slightly negative, prefer **Bearish**.\n"
            "2. **Summary Justification:** 3–5 sentences, weighing the three views. No line breaks inside words.\n\n"
            "--- ANALYSIS CONTEXT ---\n{analysis_context}"
        )
        prompt = ChatPromptTemplate.from_template(system_prompt)
        chain = prompt | self.llm
        response = chain.invoke({"entity_name": entity_name, "analysis_context": combined_analysis})
        return response.content