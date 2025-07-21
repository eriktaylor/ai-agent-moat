from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.documents import Document
from langchain_community.utilities import GoogleSearchAPIWrapper
import re
import os
from operator import itemgetter

# Import the tools this agent will use
from tools import get_stock_info, scrape_website

class ResearchAgent:
    def __init__(self, llm, embeddings_model):
        self.llm = llm
        self.embeddings_model = embeddings_model
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
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
        
        print("--- Tier 1: Official News & Analysis ---")
        headline_query = f'"{entity_name}" recent news'
        headline_results = self.search_wrapper.results(headline_query, num_results=3)
        if headline_results:
            for r in headline_results:
                doc = Document(
                    page_content=f"Headline: {r.get('title', '')}\nSnippet: {r.get('snippet', '')}",
                    metadata={"source": r.get('link', ''), "title": r.get('title', 'Source')}
                )
                source_documents.append(doc)
            print(f"Collected {len(headline_results)} headlines and snippets.")
        
        print("--- Tier 2: Critical News & Sentiment ---")
        critical_query = f'\"{entity_name}\" issues OR concerns OR investigation OR recall OR safety OR "short interest"'
        critical_results = self.search_wrapper.results(critical_query, num_results=3)
        if critical_results:
            for r in critical_results:
                doc = Document(
                    page_content=f"Critical Headline: {r.get('title', '')}\nSnippet: {r.get('snippet', '')}",
                    metadata={"source": r.get('link', ''), "title": r.get('title', 'Source')}
                )
                source_documents.append(doc)
            print(f"Collected {len(critical_results)} critical headlines and snippets.")

        print("--- Tier 3: Retail Investor Sentiment ---")
        retail_query = f'site:reddit.com/r/wallstreetbets OR site:reddit.com/r/stocks OR site:seekingalpha.com OR site:fool.com "{entity_name}" OR "{ticker}"'
        retail_results = self.search_wrapper.results(retail_query, num_results=4)
        if retail_results:
            for r in retail_results:
                doc = Document(
                    page_content=f"Retail Forum Headline: {r.get('title', '')}\nSnippet: {r.get('snippet', '')}",
                    metadata={"source": r.get('link', ''), "title": r.get('title', 'Source')}
                )
                source_documents.append(doc)
            print(f"Collected {len(retail_results)} retail forum headlines and snippets.")

        print("--- Tier 4: Deep Dive ---")
        deep_dive_query = f'\\"{entity_name}\\" market analysis OR in-depth report filetype:pdf OR site:globenewswire.com OR site:prnewswire.com'
        deep_dive_results = self.search_wrapper.results(deep_dive_query, num_results=2)
        if deep_dive_results:
            urls = [(result['link'], result['title']) for result in deep_dive_results if 'link' in result]
            for url, title in urls:
                print(f"Scraping {url}...")
                content = scrape_website.run(url)
                if content and not content.startswith("Error"):
                    doc = Document(
                        page_content=content,
                        metadata={"source": url, "title": title}
                    )
                    source_documents.append(doc)
                    print(f"Successfully scraped content from {url}")
                else:
                    print(content)
        
        self.cache[context_cache_key] = (financial_data, source_documents)
        return financial_data, source_documents

    # <<< CHANGE: This method now correctly routes data to the retriever >>>
    def _create_rag_chain(self, system_prompt, source_documents):
        """Helper to create a RAG chain that returns sources."""
        vector_store = FAISS.from_documents(documents=source_documents, embedding=self.embeddings_model)
        retriever = vector_store.as_retriever()

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        
        # This LCEL chain now correctly passes only the 'input' string to the retriever
        rag_chain = (
            RunnableParallel(
                context=itemgetter("input") | retriever,
                financial_data=itemgetter("financial_data"),
                input=itemgetter("input"),
            ).assign(answer=question_answer_chain)
        )
        return rag_chain

    def _run_analysis(self, entity_name, ticker, system_prompt, query_input):
        """Generic method to run an analysis and return answer with sources."""
        financial_data, source_documents = self._get_context(entity_name, ticker)
        if not source_documents:
            return {"answer": "Could not gather context for analysis.", "sources": []}

        rag_chain = self._create_rag_chain(system_prompt, source_documents)
        
        response = rag_chain.invoke({
            "input": query_input,
            "financial_data": financial_data
        })
        
        sources = [doc.metadata for doc in response['context']]
        unique_sources = [dict(t) for t in {tuple(d.items()) for d in sources}]
        
        return {"answer": response['answer'], "sources": unique_sources}

    def generate_market_outlook(self, entity_name, ticker):
        print("\nGenerating Market Investor Outlook...")
        system_prompt = (
            "You are a 'Market Investor' analyst. Here is the key financial data for the company:\n{financial_data}\n\n" 
            "Now, using the retrieved context below, generate a report. "
            "The report MUST be structured with the following sections:\n"
            "1. **Professional Market Sentiment:** Based on official news and critical reports, what is the professional sentiment? Is it bullish, bearish, or mixed? Why?\n"
            "2. **Retail Investor Sentiment:** Based on 'Retail Forum Headline' snippets, what is the general sentiment from retail investors? Is it aligned with or diverging from the professional sentiment?\n"
            "3. **Valuation Analysis:** Is the stock expensive or cheap? You MUST reference 'Trailing P/E' and 'Trailing EPS' from the financial data. If P/E is not applicable because EPS is negative, state this clearly.\n"
            "**Crucially, you MUST cite your sources for any claims made, referencing the source title from the metadata of the provided documents.**"
            "\n\nRetrieved Context:\n{context}\n\n"
            "DO NOT give financial advice. This is an objective summary."
        )
        return self._run_analysis(entity_name, ticker, system_prompt, f"Market outlook for {entity_name}")

    def generate_value_analysis(self, entity_name, ticker):
        print("\nGenerating Value Investor Analysis...")
        system_prompt = (
            "You are a 'Value Investor' analyst. Here is the key financial data for the company:\n{financial_data}\n\n" 
            "Now, using the retrieved context below, generate a detailed business brief. "
            "The report MUST be structured with the following sections:\n"
            "1. **Valuation Summary:** Start by stating if the company appears 'Overvalued', 'Undervalued', or 'Fairly Valued'. Justify your conclusion by referencing the P/E or EPS from the financial data.\n"
            "2. **SWOT Analysis:** A detailed, bulleted list of the company's Strengths, Weaknesses, Opportunities, and Threats. Incorporate information from 'Critical News' and 'Retail Forum' headlines.\n"
            "3. **Competitive Moat:** Based on the SWOT analysis, describe the company's long-term competitive advantages. Is its moat wide, narrow, or degrading?\n"
            "**Crucially, you MUST cite your sources for any claims made, referencing the source title from the metadata of the provided documents.**"
            "\n\nRetrieved Context:\n{context}"
        )
        return self._run_analysis(entity_name, ticker, system_prompt, f"Value analysis for {entity_name}")

    def generate_devils_advocate_view(self, entity_name, ticker):
        print("\nGenerating Devil's Advocate View...")
        system_prompt = (
            "You are a skeptical 'Devil's Advocate' financial analyst. Your sole purpose is to challenge the bullish investment thesis. "
            "Here is the key financial data for the company:\n{financial_data}\n\n" 
            "Now, using the retrieved context below, identify the single strongest counter-argument or hidden risk. "
            "Your response must be a concise, well-reasoned paragraph. "
            "Base your argument ONLY on the provided context. Focus exclusively on the most significant risk you can find in the data. "
            "**You MUST cite the source of the information you use.**"
            "\n\nRetrieved Context:\n{context}"
        )
        return self._run_analysis(entity_name, ticker, system_prompt, f"What is the strongest bearish case against {entity_name}?")
