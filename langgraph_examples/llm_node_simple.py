"""Simple LangGraph graph with a single LLM node."""

import mlflow.langchain
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.graph.state import CompiledStateGraph

from config.llm_model import LLM_MODEL
from utils.utils import save_mermaid_png

mlflow.langchain.autolog()


def llm_node(state: MessagesState) -> dict:
    """Call the LLM with the current message state and return the response."""
    response = LLM_MODEL.invoke(state["messages"])
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

    save_mermaid_png(graph, __file__)

    result = graph.invoke({"messages": [HumanMessage("What is Physics?")]})

    for message in result["messages"]:
        message.pretty_print()


if __name__ == "__main__":
    run()
