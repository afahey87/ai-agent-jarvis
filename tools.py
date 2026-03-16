"""
tools.py
Utility tool wrappers used by the research agent in `main.py`.

This module exposes three tools that the agent may call:
- `search_tool`: quick web search via DuckDuckGo
- `wiki_tool`: fetches short content from Wikipedia
- `save_tool`: appends structured results to a local text file

Each tool is created as a `langchain_core.tools.Tool` so the agent can discover
and invoke it during execution.
"""

from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import tool
from datetime import datetime
from typing import Optional


@tool
def search(query: str) -> str:
    """Search the web for information using DuckDuckGo.
    
    Args:
        query: The search query string.
        
    Returns:
        Search results as a string.
    """
    search_engine = DuckDuckGoSearchRun()
    return search_engine.run(query)


search_tool = search  # Alias for consistency


@tool
def wikipedia_search(query: str) -> str:
    """Fetch information from Wikipedia.
    
    Args:
        query: The Wikipedia search query.
        
    Returns:
        Wikipedia article content as a string.
    """
    api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
    wiki_run = WikipediaQueryRun(api_wrapper=api_wrapper)
    return wiki_run.run(query)


wiki_tool = wikipedia_search  # Alias for consistency


@tool
def save_text_to_file(data: str, filename: str = "research_output.txt") -> str:
    """Saves research data to a text file.
    
    Args:
        data: The textual content to save.
        filename: Destination file path (appends by default).
        
    Returns:
        A success message with the filename.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"

    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)

    return f"Data successfully saved to {filename}"


save_tool = save_text_to_file  # Alias for consistency
