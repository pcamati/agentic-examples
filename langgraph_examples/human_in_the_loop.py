"""Exploration of human-in-the-loop interactions in LangGraph."""

from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import MessagesState, add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt

from config.llm_model import LLM_MODEL

GRAPH_PNG_PATH = Path(__file__).parent / "latest_graph_run.png"


class GraphState(TypedDict):
    """The state of the graph, containing the message history."""

    location: str
    temperature: int
    messages: Annotated[list[AnyMessage], add_messages]


@tool
def get_temperature_from_database(
    location: str,
) -> dict[str, str | int | None]:
    """
    Read the current temperature for a given location from a database.

    Args:
        location: The location to get the temperature for.

    Returns:
        A dictionary containing the location and its current temperature.

    """
    is_approved = interrupt("Do you approve the tool call to get temperature?")
    if is_approved:
        return {"location": location, "temperature": 25}
    return {"location": location, "temperature": None}


TOOLS = [get_temperature_from_database]


def llm_node(state: MessagesState) -> dict:
    """Call the LLM with the current message state and return the response."""
    last_message = state["messages"][-1]

    if isinstance(last_message, ToolMessage):
        response = LLM_MODEL.invoke(state["messages"])
    else:
        response = LLM_MODEL.bind_tools(TOOLS).invoke(state["messages"])
    return {"messages": [response]}


def build_graph() -> CompiledStateGraph:
    """Build and compile the graph."""
    builder = StateGraph(MessagesState)

    builder.add_node("llm", llm_node)
    builder.add_node("tools", ToolNode(TOOLS))

    builder.add_edge(START, "llm")
    builder.add_conditional_edges("llm", tools_condition)
    builder.add_edge("tools", "llm")

    builder.add_edge("llm", END)

    checkpointer = InMemorySaver()
    return builder.compile(checkpointer=checkpointer)


def run() -> None:
    """Run the example."""
    graph = build_graph()

    png_data = graph.get_graph().draw_mermaid_png()
    GRAPH_PNG_PATH.write_bytes(png_data)

    system_message = SystemMessage(
        "You are a helpful climate assistant. Respond the user query."
    )
    human_message = HumanMessage("What is the current temperature in Paris?")

    config = {"configurable": {"thread_id": "1"}}
    result = graph.invoke(
        {
            "messages": [
                system_message,
                human_message,
            ]
        },
        config=config,
        version="v2",
    )

    while True:
        user_response = input(
            result.interrupts[0].value + "\nReply yes or no:"
        )
        if user_response.lower() in ["yes", "no"]:
            user_response = user_response.lower() == "yes"
            break

    result = graph.invoke(
        Command(resume=user_response), config=config, version="v2"
    )

    for message in result["messages"]:
        message.pretty_print()


if __name__ == "__main__":
    run()
