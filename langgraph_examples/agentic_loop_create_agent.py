"""Simple agentic loop in LangChain using create_agent."""

from pathlib import Path

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

GRAPH_PNG_PATH = Path(__file__).parent / "latest_graph_run.png"


@tool
def get_current_temperature() -> int:
    """Get the current temperature for a given location."""
    return 22


TOOLS = [get_current_temperature]


def run() -> None:
    """Run the example."""
    llm = ChatOllama(model="llama3.1:8b")
    agent = create_agent(
        model=llm,
        tools=TOOLS,
        system_prompt=SystemMessage(
            "You are a helpful climate assistant. Respond the user query."
        ),
    )

    png_data = agent.get_graph().draw_mermaid_png()
    GRAPH_PNG_PATH.write_bytes(png_data)

    result = agent.invoke(
        {
            "messages": [
                HumanMessage("What is the current temperature in Paris?")
            ]
        }
    )

    messages = result["messages"]

    for message in messages:
        message.pretty_print()


if __name__ == "__main__":
    run()
