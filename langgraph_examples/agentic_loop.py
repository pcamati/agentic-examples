"""Simple agentic loop in LangGraph."""

from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from config.llm_model import LLM_MODEL

GRAPH_PNG_PATH = Path(__file__).parent / "latest_graph_run.png"


@tool
def get_current_temperature() -> int:
    """Get the current temperature for a given location."""
    return 22


TOOLS = [get_current_temperature]


def llm_node(state: MessagesState) -> dict:
    """Call the LLM with the current message state and return the response."""
    llm = LLM_MODEL

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

    result = graph.invoke(
        {
            "messages": [
                system_message,
                human_message,
            ]
        }
    )

    for message in result["messages"]:
        message.pretty_print()


if __name__ == "__main__":
    run()
