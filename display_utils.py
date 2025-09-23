import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
import re
import unicodedata

# --- FIX: Added a helper function to sanitize text before rendering ---
def _clean_markdown(text: str) -> str:
    """Normalizes and cleans markdown text to prevent rendering artifacts."""
    if not isinstance(text, str):
        return str(text)
    # Normalize unicode characters
    s = unicodedata.normalize("NFKC", text)
    # Replace single-character line breaks that occur inside words/numbers
    s = re.sub(r'(?<=\w)\n(?=\w)', ' ', s)
    # Collapse 3 or more newlines into a maximum of 2
    s = re.sub(r'\n{3,}', '\n\n', s)
    # Collapse excessive spaces or tabs into a single space
    s = re.sub(r'[ \t]{2,}', ' ', s)
    return s.strip()

def display_analysis(title, company_name, result, is_summary=False):
    """
    Formats and displays the agent's analysis in a structured Streamlit format.
    """
    st.subheader(title)

    answer = result if is_summary else result.get('answer', 'No analysis generated.')
    # --- FIX: Clean the answer text before displaying it ---
    answer = _clean_markdown(answer)
    
    sources = result.get('sources', []) if not is_summary else []

    with st.container(border=True):
        st.markdown(answer)

    # Use an expander for the sources to keep the UI clean
    if sources:
        with st.expander("View Sources Used in this Section"):
            for i, source_doc in enumerate(sources):
                source = source_doc.metadata
                title_text = str(source.get('title', 'Untitled Source'))

                st.markdown(f"**{i+1}. {title_text}**")
                st.caption(f"Source: [{source.get('source', '#')}]({source.get('source', '#')})")