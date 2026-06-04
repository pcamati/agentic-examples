"""How to handle multiple interrupts in LangGraph."""

from typing import Annotated, TypedDict

import mlflow.langchain
from langchain.messages import AnyMessage
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt

from config.llm_model import LLM_MODEL
from utils.utils import save_mermaid_png

mlflow.langchain.autolog()


class GraphState(TypedDict):
    """State of the graph."""

    messages: Annotated[AnyMessage, add_messages]
    interrupt_a: str | None = None
    interrupt_b: str | None = None


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


def llm_node(state: GraphState) -> dict:
    """Call the LLM with the current message state and return the response."""
    last_message = state["messages"][-1]

    if isinstance(last_message, ToolMessage):
        response = LLM_MODEL.invoke(state["messages"])
    else:
        response = LLM_MODEL.bind_tools(TOOLS).invoke(state["messages"])
    return {"messages": [response]}


def interrupt_a(state: GraphState) -> dict:  # noqa: ARG001
    """Interrupt function to ask user a question."""
    answer = interrupt({"math_question": "What is 1+3?"})
    if answer == "4":
        return {"interrupt_a": "Correct!"}
    return {"interrupt_a": "Incorrect."}


def interrupt_b(state: GraphState) -> dict:  # noqa: ARG001
    """Interrupt function to ask user a question."""
    answer = interrupt("What is the capital of France?")
    if answer.lower() == "paris":
        return {"interrupt_b": "Correct!"}
    return {"interrupt_b": "Incorrect."}


def build_graph() -> CompiledStateGraph:
    """Build and compile the graph."""
    builder = StateGraph(GraphState)

    builder.add_node("llm", llm_node)
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_node("interrupt_a", interrupt_a)
    builder.add_node("interrupt_b", interrupt_b)

    builder.add_edge(START, "llm")
    builder.add_conditional_edges("llm", tools_condition)
    builder.add_edge("tools", "llm")
    builder.add_edge("llm", "interrupt_a")
    builder.add_edge("llm", "interrupt_b")

    builder.add_edge("llm", END)
    builder.add_edge("interrupt_a", END)
    builder.add_edge("interrupt_b", END)

    checkpointer = InMemorySaver()
    return builder.compile(checkpointer=checkpointer)


def run() -> None:
    """Run the example."""
    graph = build_graph()

    save_mermaid_png(graph, __file__)

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

    interrupt_map = {
        interrupt.id: interrupt.value for interrupt in result.interrupts
    }

    interrupt_answers_map = {}
    for interrupt_id, interrupt_value in interrupt_map.items():
        if isinstance(interrupt_value, dict):
            answer = input(interrupt_value["math_question"])
        else:
            answer = input(interrupt_value)
        interrupt_answers_map[interrupt_id] = answer

    result = graph.invoke(
        Command(resume=interrupt_answers_map), config=config, version="v2"
    )
    # After resuming, graph is still interrupted because the interrupt
    # nodes are executed again.

    for message in result["messages"]:
        message.pretty_print()

    history = list(graph.get_state_history(config))
    for step, snapshot in enumerate(history[::-1]):
        print(
            f"Superstep {step}: state={snapshot.values}, next={snapshot.next}"
        )
        print("=" * 50)


if __name__ == "__main__":
    run()
