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

        financial_data = "No financial data available."
        if ticker:
            print(f"--- Getting Financial Data for {ticker} ---")
            financial_data_result = get_stock_info.run(ticker)
            if not financial_data_result.startswith("Error"):
                financial_data = financial_data_result
                print("Successfully collected financial data.")

        source_documents = []
        
        queries = {
            "Official News & Analysis": f'"{entity_name}" recent news',
            "Critical News & Sentiment": f'\"{entity_name}\" issues OR concerns OR investigation OR recall OR safety OR "short interest"',
            "Retail Investor Sentiment": f'site:reddit.com/r/wallstreetbets OR site:reddit.com/r/stocks OR site:seekingalpha.com OR site:fool.com "{entity_name}" OR "{ticker}"',
        }

        for tier, query in queries.items():
            print(f"--- Searching for: {tier} ---")
            search_result_text = self.search_tool.run(query)
            
            if search_result_text and "No good search result found" not in search_result_text:
                doc = Document(
                    page_content=search_result_text,
                    metadata={
                        "source": "DuckDuckGo Search", 
                        "title": f"{tier} for {entity_name}",
                        "published": "N/A"
                    }
                )
                source_documents.append(doc)
                print(f"Collected results for {tier}.")
        
        self.cache[context_cache_key] = (financial_data, source_documents)
        return financial_data, source_documents

    def _create_rag_chain(self, system_prompt, source_documents):
        vector_store = FAISS.from_documents(documents=source_documents, embedding=self.embeddings_model)
        retriever = vector_store.as_retriever()

        def format_docs_with_citations(docs):
            formatted_docs = []
            for i, doc in enumerate(docs):
                doc_string = f"[Source {i+1}]: Title: {doc.metadata.get('title', 'N/A')}\nContent: {doc.page_content}"
                formatted_docs.append(doc_string)
            return "\n\n".join(formatted_docs)

        rag_chain = (
            {
                "context_docs": itemgetter("input") | retriever,
                "input": itemgetter("input"),
                "financial_data": itemgetter("financial_data"),
            }
            | RunnablePassthrough.assign(
                context_formatted=lambda x: format_docs_with_citations(x["context_docs"])
            )
            | {
                "answer": (
                    ChatPromptTemplate.from_template(system_prompt)
                    | self.llm
                ),
                "sources": itemgetter("context_docs"),
            }
        )
        return rag_chain

    def _run_analysis(self, entity_name, ticker, system_prompt, query_input):
        financial_data, source_documents = self._get_context(entity_name, ticker)
        if not source_documents:
            return {"answer": "Could not gather context for analysis.", "sources": []}

        rag_chain = self._create_rag_chain(system_prompt, source_documents)
        
        response = rag_chain.invoke({
            "input": query_input,
            "financial_data": financial_data
        })
        
        return {"answer": response['answer'].content, "sources": response['sources']}

    def generate_market_outlook(self, entity_name, ticker):
        print("\nGenerating Market Investor Outlook...")
        system_prompt = (
            "You are a 'Market Investor' analyst. Here is the key financial data for the company:\n{financial_data}\n\n" 
            "Now, using the retrieved context below, generate a report. The context is a list of documents, each prefixed with a citation number (e.g., [Source 1]). "
            "The report MUST be structured with the following sections:\n"
            "1. **Professional Market Sentiment:** Based on official news and critical reports, what is the professional sentiment?\n"
            "2. **Retail Investor Sentiment:** Based on 'Retail Forum' snippets, what is the general sentiment from retail investors?\n"
            "3. **Valuation Analysis:** Is the stock expensive or cheap? You MUST reference 'Trailing P/E' and 'Trailing EPS' from the financial data. If P/E is not applicable, state this clearly.\n"
            "**Crucially, you MUST cite your sources for any claims made by adding the citation number (e.g., [1], [2]) at the end of the sentence.**"
            "\n\nRetrieved Context:\n{context_formatted}\n\n"
            "DO NOT give financial advice. This is an objective summary."
        )
        return self._run_analysis(entity_name, ticker, system_prompt, f"Market outlook for {entity_name}")

    def generate_value_analysis(self, entity_name, ticker):
        print("\nGenerating Value Investor Analysis...")
        system_prompt = (
            "You are a 'Value Investor' analyst. Here is the key financial data for the company:\n{financial_data}\n\n" 
            "Now, using the retrieved context below, generate a detailed business brief. The context is a list of documents, each prefixed with a citation number (e.g., [Source 1]). "
            "The report MUST be structured with the following sections:\n"
            "1. **Valuation Summary:** Start by stating if the company appears 'Overvalued', 'Undervalued', or 'Fairly Valued', justifying your conclusion with financial data.\n"
            "2. **SWOT Analysis:** A detailed, bulleted list of the company's Strengths, Weaknesses, Opportunities, and Threats.\n"
            "3. **Competitive Moat:** Based on the SWOT analysis, describe the company's long-term competitive advantages.\n"
            "**Crucially, you MUST cite your sources for any claims made by adding the citation number (e.g., [1], [2]) at the end of the sentence.**"
            "\n\nRetrieved Context:\n{context_formatted}"
        )
        return self._run_analysis(entity_name, ticker, system_prompt, f"Value analysis for {entity_name}")

    def generate_devils_advocate_view(self, entity_name, ticker):
        print("\nGenerating Devil's Advocate View...")
        system_prompt = (
            "You are a skeptical 'Devil's Advocate' financial analyst. Your sole purpose is to challenge the bullish investment thesis. "
            "Here is the key financial data for the company:\n{financial_data}\n\n" 
            "Now, using the retrieved context below, identify the single strongest counter-argument or hidden risk. The context is a list of documents, each prefixed with a citation number (e.g., [Source 1]). "
            "Your response must be a concise, well-reasoned paragraph. "
            "**You MUST cite the source of the information you use by adding the citation number (e.g., [1], [2]) at the end of the sentence.**"
            "\n\nRetrieved Context:\n{context_formatted}"
        )
        return self._run_analysis(entity_name, ticker, system_prompt, f"What is the strongest bearish case against {entity_name}?")

    # <<< CHANGE: This method now accepts entity_name to inject into the prompt >>>
    def generate_final_summary(self, entity_name, market_outlook, value_analysis, devils_advocate):
        print("\nGenerating Final Consensus Summary...")
        
        # <<< CHANGE: The input to the chain now includes the entity_name >>>
        combined_analysis = (
            f"COMPANY BEING ANALYZED: {entity_name}\n\n"
            f"--- Market Investor Outlook ---\n{market_outlook}\n\n"
            f"--- Value Investor Analysis ---\n{value_analysis}\n\n"
            f"--- Devil's Advocate View ---\n{devils_advocate}"
        )
        
        # <<< CHANGE: The prompt is now an f-string that uses the entity_name >>>
        system_prompt = (
            "You are a 'Lead Analyst' responsible for synthesizing the views of your team into a final investment rating for {entity_name}. "
            "You have been provided with three reports: a Market Investor's outlook, a Value Investor's analysis, and a Devil's Advocate's critique. "
            "Your task is to synthesize these three perspectives into a final, balanced summary for {entity_name}. "
            "Your response MUST be structured with the following sections:\n"
            "1. **Consensus Rating:** Provide a single rating: **Bullish**, **Bearish**, or **Neutral with Caution**. \n"
            "2. **Summary Justification:** In a concise paragraph, explain your rating by summarizing how you weighed the different perspectives for {entity_name}."
        )
        
        prompt = ChatPromptTemplate.from_template(system_prompt)
        
        chain = prompt | self.llm
        # Pass the combined analysis and the entity name to the chain
        response = chain.invoke({"input": combined_analysis, "entity_name": entity_name})
        return response.content
