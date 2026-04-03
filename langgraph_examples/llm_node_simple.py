"""Simple LangGraph graph with a single LLM node using Ollama."""

from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.graph.state import CompiledStateGraph

GRAPH_PNG_PATH = Path(__file__).parent / "latest_graph_run.png"


def llm_node(state: MessagesState) -> dict:
    """Call the LLM with the current message state and return the response."""
    llm = ChatOllama(model="llama3.1:8b")
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


def build_graph() -> CompiledStateGraph[
    MessagesState, None, MessagesState, MessagesState
]:
    """Build and compile the graph."""
    builder = StateGraph(MessagesState)
    builder.add_node("llm", llm_node)
    builder.add_edge(START, "llm")
    builder.add_edge("llm", END)
    return builder.compile()


def run() -> None:
    """Run the example."""
    graph = build_graph()

    png_data = graph.get_graph().draw_mermaid_png()
    GRAPH_PNG_PATH.write_bytes(png_data)

    result = graph.invoke({"messages": [HumanMessage("What is Physics?")]})

    for message in result["messages"]:
        message.pretty_print()


if __name__ == "__main__":
    run()
