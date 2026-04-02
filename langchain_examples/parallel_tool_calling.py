"""
Show how to perform parallel tool calling.

The invoke in the LLM augmented with tools does not execute the tools.
The LLM decides which tools to call and with what arguments.
The actual execution of the tools is separate, and can be done
sequentially or in parallel.
"""

import time
from concurrent.futures import ThreadPoolExecutor

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

TOOL_LATENCY_SECONDS = 2  # Simulates a real network/API call


@tool
def get_current_temperature(location: str) -> str:
    """Get the current temperature for a given location."""
    print(f"  [{location}] started")
    time.sleep(TOOL_LATENCY_SECONDS)  # Simulate I/O latency
    print(f"  [{location}] finished")
    return f"The current temperature in {location} is 22°C."


def run() -> None:
    """Run the example."""
    llm = ChatOllama(model="llama3.1:8b")
    llm_with_tools = llm.bind_tools([get_current_temperature])

    # Ask about two cities so the model issues two tool calls in one response
    response = llm_with_tools.invoke(
        [
            HumanMessage(
                "What is the current temperature in Paris and New York?"
            )
        ]
    )

    print("--- Sequential (start, finish, start, finish) ---")
    for tool_call in response.tool_calls:
        get_current_temperature.invoke(tool_call)

    print("--- Parallel (start, start, finish, finish) ---")
    with ThreadPoolExecutor() as executor:
        list(executor.map(get_current_temperature.invoke, response.tool_calls))


if __name__ == "__main__":
    run()
