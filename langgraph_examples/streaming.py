"""Simple agentic streaming in LangGraph."""

from pathlib import Path

import mlflow.langchain
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from config.llm_model import LLM_MODEL

mlflow.langchain.autolog()

GRAPH_PNG_PATH = Path(__file__).parent / "latest_graph_run.png"


@tool
def get_current_temperature() -> int:
    """Get the current temperature for a given location."""
    writer = get_stream_writer()
    writer("WE ARE INSIDE THE TOOL")
    return 22


TOOLS = [get_current_temperature]


def llm_node(state: MessagesState) -> dict:
    """Call the LLM with the current message state and return the response."""
    llm = LLM_MODEL
    writer = get_stream_writer()

    # Emit a custom message
    writer("WE ARE INSIDE THE LLM NODE")

    last_message = state["messages"][-1]
    # After tool results are returned, invoke without tools so the model is
    # forced to produce a natural-language reply instead of another tool call.
    if isinstance(last_message, ToolMessage):
        response = llm.invoke(state["messages"])
    else:
        response = llm.bind_tools(TOOLS).invoke(state["messages"])
    return {"messages": [response]}


def build_graph() -> CompiledStateGraph:
    """Build and compile the graph."""
    builder = StateGraph(MessagesState)

    builder.add_node("llm", llm_node)
    builder.add_node("tools", ToolNode(TOOLS))  # Already executes the tool

    builder.add_edge(START, "llm")
    builder.add_conditional_edges("llm", tools_condition)
    builder.add_edge("tools", "llm")

    builder.add_edge("llm", END)

    return builder.compile()


def run() -> None:
    """Run the example."""
    graph = build_graph()

    png_data = graph.get_graph().draw_mermaid_png()
    GRAPH_PNG_PATH.write_bytes(png_data)

    system_message = SystemMessage(
        "You are a helpful climate assistant. Respond the user query."
    )
    human_message = HumanMessage("What is the current temperature in Paris?")
    initial_state = {
        "messages": [
            system_message,
            human_message,
        ]
    }

    for chunk in graph.stream(
        initial_state,
        stream_mode=[
            "values",
            "updates",
            "messages",
            "custom",
            "checkpoints",  # Requires a checkpointer
            "tasks",
            "debug",
        ],
        version="v2",
    ):
        if chunk["type"] == "values":
            print("=" * 50)
            print(f"Current graph values: {chunk['data']}")
        elif chunk["type"] == "updates":
            print("=" * 50)
            for node_name, state in chunk["data"].items():
                print(f"Node {node_name} updated: {state}")
        elif chunk["type"] == "messages":
            print("=" * 50)
            print(f"New messages: {chunk['data']}")
        elif chunk["type"] == "custom":
            print("=" * 50)
            print(f"Custom message: {chunk['data']}")
        elif chunk["type"] == "checkpoints":
            print("=" * 50)
            print(f"Checkpoint reached: {chunk['data']['checkpoint_name']}")
        elif chunk["type"] == "tasks":
            print("=" * 50)
            print(f"Task update: {chunk['data']}")
        elif chunk["type"] == "debug":
            print("=" * 50)
            print(f"Debug info: {chunk['data']}")


if __name__ == "__main__":
    run()
