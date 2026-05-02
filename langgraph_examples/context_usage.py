"""Example usage of contexts in LangGraph."""

from pathlib import Path
from typing import Annotated

from langchain.messages import AnyMessage
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from pydantic import BaseModel

GRAPH_PNG_PATH = Path(__file__).parent / "latest_graph_run.png"


class GraphState(BaseModel):
    """Graph state."""

    previous_node: str | None = None
    messages: Annotated[list[AnyMessage], add_messages]


class StaticMemory(BaseModel):
    """Static memory for the graph."""

    user_name: str | None


def llm_node(state: GraphState, runtime: Runtime[StaticMemory]) -> GraphState:
    """Call the LLM with the current message state and return the response."""
    llm = ChatOllama(model="llama3.1:8b")
    user_name = runtime.context.user_name  # noqa: F841

    response = llm.invoke(state.messages)
    return {"messages": [response], "previous_node": "llm_node"}


def static_node(
    state: GraphState, runtime: Runtime[StaticMemory]
) -> GraphState:
    """Add some context to the graph."""
    user_name = runtime.context.user_name

    if user_name:
        return {"previous_node": "static_node"}

    static_message = AIMessage(
        "The static node was reached without user name provided."
        f"The last message is {state.messages[-1].content}"
    )
    return {"messages": [static_message], "previous_node": "static_node"}


def build_graph() -> CompiledStateGraph:
    """Build and compile the graph."""
    builder = StateGraph(GraphState, context_schema=StaticMemory)

    builder.add_node("llm", llm_node)
    builder.add_node("static", static_node)

    builder.add_edge(START, "llm")

    builder.add_edge("llm", "static")

    builder.add_edge("static", END)

    return builder.compile()


def run() -> None:
    """Run the example."""
    graph = build_graph()

    png_data = graph.get_graph().draw_mermaid_png()
    GRAPH_PNG_PATH.write_bytes(png_data)

    system_message = SystemMessage("You reply as concisely as possible.")
    human_message = HumanMessage("What is Physics?")

    initial_messages = {
        "messages": [
            system_message,
            human_message,
        ]
    }

    result = graph.invoke(
        initial_messages,
        context={"user_name": "Patrice"},
    )

    for message in result["messages"]:
        message.pretty_print()

    result = graph.invoke(
        initial_messages,
        context={"user_name": None},
    )

    for message in result["messages"]:
        message.pretty_print()


if __name__ == "__main__":
    run()
