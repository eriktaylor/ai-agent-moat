import streamlit as st
from streamlit_extras.keyboard_text import key
from streamlit_extras.add_vertical_space import add_vertical_space

def display_analysis(title, company_name, result, is_summary=False):
    """
    Formats and displays the agent's analysis in a structured Streamlit format
    with a copy-to-clipboard button.
    """
    st.subheader(title)

    answer = result if is_summary else result.get('answer', 'No analysis generated.')
    sources = result.get('sources', []) if not is_summary else []

    # Use a container for the analysis text and the copy button
    with st.container(border=True):
        st.markdown(answer)
        add_vertical_space(1)
        # Simple text area to allow easy copying. A true clipboard button is complex in Streamlit.
        st.text_area("Copyable text:", answer, height=150, label_visibility="collapsed")


    # Use an expander for the sources to keep the UI clean
    if sources:
        with st.expander("View Sources Used in this Section"):
            for i, source_doc in enumerate(sources):
                source = source_doc.metadata
                title_text = str(source.get('title', 'Untitled Source'))

                # Create a more readable source link
                st.markdown(f"**{i+1}. {title_text}**")
                st.caption(f"Source: [{source.get('source', '#')}]({source.get('source', '#')})")
