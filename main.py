"""
main.py
Minimal script to run a tool-calling research agent.

This file sets up:
- environment loading from a `.env` file
- a Pydantic model describing the structured response
- a LangChain Chat model and prompt template with output parsing
- tool wrappers (search, wiki, save) imported from `tools.py`

Run the script and enter a research query when prompted. The agent will
attempt to call tools and return a structured `ResearchResponse`.
"""

from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.agents.factory import create_agent
from langchain_core.messages import HumanMessage
from tools import search_tool, wiki_tool, save_tool

# Load environment variables from .env (e.g. API keys)
load_dotenv()


class ResearchResponse(BaseModel):
    """Pydantic schema for the agent's structured research output."""

    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]


# Initialize LLM client (make sure your environment provides needed API keys)
llm = ChatGroq(model="llama-3.1-8b-instant")

# Parser for converting JSON output to Pydantic model
parser = JsonOutputParser(pydantic_object=ResearchResponse)


# Build the prompt template for tool-calling agents
system_prompt = """You are a research assistant that helps generate research papers.
Answer the user query and use necessary tools to gather information.
Format your final response as a JSON object with this structure:
{{
    "topic": "the research topic",
    "summary": "a comprehensive summary of findings",
    "sources": ["list", "of", "sources"],
    "tools_used": ["list", "of", "tools", "used"]
}}
Provide only valid JSON, no additional text."""


# Register tools and create the agent
tools = [search_tool, wiki_tool, save_tool]
agent = create_agent(model=llm, tools=tools, system_prompt=system_prompt)


def main():
    """Prompt the user, invoke the agent, and parse the structured response."""

    query = input("What can I help you research? ")
    
    try:
        # Invoke the agent with the query
        response = agent.invoke({"messages": [HumanMessage(content=query)]})
        print("\n=== Agent Execution Complete ===")
        output = response["messages"][-1].content
        print("Raw agent output:\n", output)
        
        # Parse the structured response
        if output:
            parsed_dict = parser.parse(output)
            structured_response = ResearchResponse(**parsed_dict)
            print("\n=== Parsed Response ===")
            print(f"Topic: {structured_response.topic}")
            print(f"Summary: {structured_response.summary}")
            print(f"Sources: {structured_response.sources}")
            print(f"Tools Used: {structured_response.tools_used}")
    except Exception as e:
        print(f"Error during execution or parsing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()