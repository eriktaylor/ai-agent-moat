from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_community.utilities import GoogleSearchAPIWrapper
import re
import os
from operator import itemgetter
import json

# Import the tools this agent will use
from tools_colab import get_stock_info, scrape_website

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
                    metadata={
                        "source": r.get('link', ''), 
                        "title": r.get('title', 'Source'),
                        "published": r.get('publication_time', 'N/A')
                    }
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
                    metadata={
                        "source": r.get('link', ''), 
                        "title": r.get('title', 'Source'),
                        "published": r.get('publication_time', 'N/A')
                    }
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
                    metadata={
                        "source": r.get('link', ''), 
                        "title": r.get('title', 'Source'),
                        "published": r.get('publication_time', 'N/A')
                    }
                )
                source_documents.append(doc)
            print(f"Collected {len(retail_results)} retail forum headlines and snippets.")

        print("--- Tier 4: Deep Dive ---")
        deep_dive_query = f'\\"{entity_name}\\" market analysis OR in-depth report filetype:pdf OR site:globenewswire.com OR site:prnewswire.com'
        deep_dive_results = self.search_wrapper.results(deep_dive_query, num_results=2)
        if deep_dive_results:
            urls = [(result['link'], result['title'], result.get('publication_time', 'N/A')) for result in deep_dive_results if 'link' in result]
            for url, title, published in urls:
                print(f"Scraping {url}...")
                content = scrape_website.run(url)
                if content and not content.startswith("Error"):
                    doc = Document(
                        page_content=content,
                        metadata={"source": url, "title": title, "published": published}
                    )
                    source_documents.append(doc)
                    print(f"Successfully scraped content from {url}")
                else:
                    print(content)
        
        self.cache[context_cache_key] = (financial_data, source_documents)
        return financial_data, source_documents

    def _create_rag_chain(self, system_prompt, source_documents):
        vector_store = FAISS.from_documents(documents=source_documents, embedding=self.embeddings_model)
        retriever = vector_store.as_retriever()

        def format_docs_with_citations(docs):
            formatted_docs = []
            for i, doc in enumerate(docs):
                doc_string = f"[Source {i+1}]: Title: {doc.metadata.get('title', 'N/A')}\nPublished: {doc.metadata.get('published', 'N/A')}\nContent: {doc.page_content}"
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

     def generate_scout_analysis(self, entity_name, ticker, quant_score):
        """
        Performs a hybrid analysis by synthesizing a quantitative score with qualitative news snippets.
        """
        scout_cache_key = f"hybrid_scout_{entity_name}_{ticker}"
        if scout_cache_key in self.cache:
            print(f"🕵️ Returning cached hybrid analysis for {entity_name}...")
            return self.cache[scout_cache_key]

        print(f"🕵️ Performing hybrid scout for {entity_name}...")
        
        try:
            scout_query = f'"{entity_name}" OR "{ticker}" stock news catalyst OR earnings OR guidance OR "analyst rating"'
            search_results = self.search_wrapper.results(scout_query, num_results=4)
        except Exception as e:
            # Handle API failure
            return {"answer": f'{{"justification": "Error: Google Search API failed. ({e})", "hybrid_score": 0}}', "sources": []}

        context_str = "\n---\n".join([f"Title: {r.get('title', '')}\nSnippet: {r.get('snippet', '')}" for r in search_results])
        if not search_results:
            context_str = "No recent news found."

        # Define the new "Hybrid Analyst" prompt
        hybrid_system_prompt = (
            "You are a 'Hybrid Financial Analyst'. Your job is to synthesize a quantitative model score with qualitative news analysis for '{entity_name}'.\n\n"
            "Here are your inputs:\n"
            "1. Quantitative Score: {quant_score:.2f} (A probability from 0.0 to 1.0 that the stock will outperform. Higher is better.)\n"
            "2. Recent News Snippets:\n---\n{context}\n---\n\n"
            "Your Task: Provide a final 'Hybrid Score' from 1-10. A high score requires BOTH a strong quantitative score (e.g., > 0.65) AND a compelling, positive news catalyst. "
            "A weak quant score OR negative/boring news should result in a low score. "
            "Answer in a single, valid JSON object with the following keys:\n"
            "- \"hybrid_score\": (integer, 1-10)\n"
            "- \"justification\": (string) A one-sentence explanation for the score, referencing both the quant score and the news.\n"
            "- \"catalyst_summary\": (string) A brief summary of the news catalyst. If no news, state that.\n"
            "- \"sentiment\": (string) \"Positive\", \"Negative\", or \"Neutral\" based on the news."
            "\n\nJSON Response:"
        )
        prompt = ChatPromptTemplate.from_template(hybrid_system_prompt)

        # Run the LLM call
        chain = prompt | self.llm
        response = chain.invoke({
            "entity_name": entity_name,
            "quant_score": quant_score,
            "context": context_str
        })
        
        final_result = {"answer": response.content, "sources": []}
        self.cache[scout_cache_key] = final_result
        return final_result
    
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


    def generate_final_summary(self, market_outlook, value_analysis, devils_advocate):
        print("\nGenerating Final Consensus Summary...")
        
        combined_analysis = (
            f"--- Market Investor Outlook ---\n{market_outlook}\n\n"
            f"--- Value Investor Analysis ---\n{value_analysis}\n\n"
            f"--- Devil's Advocate View ---\n{devils_advocate}"
        )
        
        # <<< CHANGE: Added a placeholder {analysis_context} to receive the combined reports >>>
        system_prompt_template = (
            "You are a 'Lead Analyst' responsible for synthesizing the views of your team into a final investment rating. "
            "You have been provided with three reports which constitute the analysis context. "
            "Your task is to synthesize these three perspectives into a final, balanced summary. "
            "Your response MUST be structured with the following sections:\n"
            "1. **Consensus Rating:** Provide a single rating: **Bullish**, **Bearish**, or **Neutral with Caution**. \n"
            "2. **Summary Justification:** In a concise paragraph, explain your rating by summarizing how you weighed the different perspectives.\n\n"
            "--- ANALYSIS CONTEXT ---\n"
            "{analysis_context}" # This placeholder will be filled with the combined reports.
        )
        
        prompt = ChatPromptTemplate.from_template(system_prompt_template)
        
        chain = prompt | self.llm
        
        # <<< CHANGE: The key in the invoke dictionary now matches the placeholder >>>
        response = chain.invoke({"analysis_context": combined_analysis})
        
        return response.content
