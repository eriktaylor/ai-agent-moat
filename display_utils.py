import streamlit as st

def display_analysis(title, company_name, result, is_summary=False):
    """
    Formats and displays the agent's analysis in a structured Streamlit format.
    """
    st.subheader(title)
    
    answer = result if is_summary else result.get('answer', 'No analysis generated.')
    sources = result.get('sources', []) if not is_summary else []
    
    # Use st.markdown for the main analysis text. The `unsafe_allow_html=True`
    # allows us to use HTML for styling if needed, but markdown is safer.
    st.markdown(answer)
    
    # Use an expander for the sources to keep the UI clean
    if sources:
        with st.expander("View Sources Used in this Section"):
            for i, source_doc in enumerate(sources):
                source = source_doc.metadata
                title_text = str(source.get('title', 'Untitled Source'))
                published_date_str = str(source.get('published', 'N/A'))
                
                # Only show the date part if it's available
                display_date = f"({published_date_str.split(',')[0]})" if 'N/A' not in published_date_str else ""
                
                st.markdown(f"{i+1}. [{title_text}]({source.get('source', '#')}) {display_date}")
