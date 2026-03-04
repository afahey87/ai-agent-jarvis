"""
tools.py
Utility tool wrappers used by the research agent in `main.py`.

This module exposes three tools that the agent may call:
- `search_tool`: quick web search via DuckDuckGo
- `wiki_tool`: fetches short content from Wikipedia
- `save_tool`: appends structured results to a local text file

Each tool is created as a `langchain.Tool` so the agent can discover
and invoke it during execution.
"""

from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime


def save_to_txt(data: str, filename: str = "research_output.txt"):
    """Append research results to a local text file with a timestamp.

    Args:
        data: The textual content to save.
        filename: Destination file path (appends by default).

    Returns:
        A short success message with the filename.
    """

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"

    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)

    return f"Data successfully saved to {filename}"


# Expose a simple save tool the agent can call
save_tool = Tool(
    name="save_text_to_file",
    func=save_to_txt,
    description="Saves structured research data to a text file.",
)


# DuckDuckGo search wrapper (used for general web search queries)
search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="search",
    func=search.run,
    description="Search the web for information",
)


# Wikipedia query wrapper: returns short article content via the API wrapper
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)
