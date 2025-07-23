from IPython.display import display, HTML

def display_analysis(title, company_name, result, is_summary=False):
    """
    Formats and displays the agent's analysis in a structured HTML format for Colab.
    """
    print(f"\n--- {title} for {company_name} ---")
    
    answer = result if is_summary else result.get('answer', 'No analysis generated.')
    sources = result.get('sources', []) if not is_summary else []
    
    # Determine the border color based on the analysis type
    if "Devil's Advocate" in title:
        border_color = '#c74a4a' # Red for the bearish take
    elif is_summary:
        border_color = '#007bff' # Blue for the final summary
    else:
        border_color = '#555' # Default grey
        
    # Display the main analysis box
    display(HTML(f"""
    <div style="border: 1px solid {border_color}; border-radius: 8px; padding: 20px; max-height: 500px; overflow-y: auto; white-space: pre-wrap; font-family: 'SF Pro Text', 'Inter', sans-serif; line-height: 1.6; background-color: #2c2c2e; color: #f0f0f0;">
        {answer}
    </div>
    """))
    
    # Display the numbered list of sources
    if sources:
        sources_html = "<div style='margin-top: 10px;'><strong>Sources Used in this Section:</strong><ol style='margin-left: 20px; padding-left: 10px; color: #ccc;'>"
        for i, source_doc in enumerate(sources):
            source = source_doc.metadata
            title_text = str(source.get('title', 'Untitled Source'))
            published_date_str = str(source.get('published', 'N/A'))
            
            # Only show the date part if it's available
            display_date = f"({published_date_str.split(',')[0]})" if 'N/A' not in published_date_str else ""
            
            sources_html += f"""
            <li style='margin-bottom: 5px;'>
                <a href='{source.get('source', '#')}' target='_blank' style='color: #8ab4f8;'>{HTML(title_text).data}</a> {display_date}
            </li>
            """
        sources_html += "</ol></div>"
        display(HTML(sources_html))
