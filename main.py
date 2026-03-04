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
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
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

# Parser that will convert model output into the `ResearchResponse` model
parser = PydanticOutputParser(pydantic_object=ResearchResponse)


# Build the prompt template; include format instructions for the parser
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use neccessary tools. 
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())


# Register tools and create the agent that can call them
tools = [search_tool, wiki_tool, save_tool]
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools,
)


# Executor runs the agent loop; verbose=True prints tool calls
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


def main():
    """Prompt the user, invoke the agent, and parse the structured response."""

    query = input("What can I help you research? ")
    raw_response = agent_executor.invoke({"query": query})
    print("Raw agent response:\n", raw_response)

    try:
        # Many agent runtimes return a list-of-messages inside `output`; extract safely
        structured_response = parser.parse(raw_response.get("output")[0]["text"])
        print("Parsed response:\n", structured_response)
    except Exception as e:
        # If parsing fails, print the error and raw response for debugging
        print("Error parsing response", e, "Raw Response - ", raw_response)


if __name__ == "__main__":
    main()