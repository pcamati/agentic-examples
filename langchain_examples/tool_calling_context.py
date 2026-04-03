"""
Contexts for tool call using Langchain.

The function name and the docstring of the tool are given as context
to the LLM model, so that it can decide when to call the tool and with
what arguments.
"""

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama


@tool
def always_mention_the_name_patrice() -> str:
    """Call this tool every time. Always mention the name Gabriela."""
    return "what are the two names that appeared in your context?"


def run() -> None:
    """Run the example."""
    llm = ChatOllama(model="llama3.1:8b")
    llm_with_tools = llm.bind_tools([always_mention_the_name_patrice])

    # JSON schema the LLM receives to understand how to call the tool
    print(
        "Tool schema:",
        always_mention_the_name_patrice.args_schema.model_json_schema(),
    )
    print("=" * 50)

    # Full tool payload sent to the model (as registered on the bound LLM)
    print("Bound tools:", llm_with_tools.kwargs["tools"])
    print("=" * 50)

    human_message = HumanMessage(
        "What are the names provided in your context?"
    )
    response = llm_with_tools.invoke([human_message])

    # Text content from the model (usually empty when tools are called)
    print("Content:", response.content)
    print("=" * 50)

    # Structured tool call requests made by the model
    print("Tool calls:", response.tool_calls)
    print("=" * 50)

    # Execute the tool calls and inspect the resulting ToolMessage
    tool_messages = []
    for tool_call in response.tool_calls:
        # Structured arguments the model extracted from natural language
        print(f"Tool call arguments: {tool_call['args']}")
        print("=" * 50)

        tool_message: ToolMessage = always_mention_the_name_patrice.invoke(
            tool_call
        )
        tool_messages.append(tool_message)

        # ToolMessage attributes that flow back into the conversation
        print(f"ToolMessage content: {tool_message.content}")
        print("=" * 50)

    # Full agentic loop: send AIMessage + ToolMessages back to get the
    # final answer
    final_response = llm_with_tools.invoke(
        [human_message, response, *tool_messages]
    )
    print("Final answer:", final_response.content)
    print("=" * 50)


if __name__ == "__main__":
    run()
