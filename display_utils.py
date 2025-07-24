import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space

def display_analysis(title, company_name, result, is_summary=False):
    """
    Formats and displays the agent's analysis in a structured Streamlit format.
    """
    st.subheader(title)

    answer = result if is_summary else result.get('answer', 'No analysis generated.')
    sources = result.get('sources', []) if not is_summary else []

    # <<< CHANGE: Removed the redundant text_area, now only markdown is shown >>>
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
