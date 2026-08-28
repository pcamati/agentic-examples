"""Simplest possible deepagents agent, with no tools or subagents."""

import mlflow.langchain
from deepagents import create_deep_agent

from config.llm_model import LLM_MODEL

mlflow.langchain.autolog()

agent = create_deep_agent(model=LLM_MODEL)


def ask_tools_question() -> None:
    """Run the example."""
    user_message = "What tools do you have at your disposal?"
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_message}]}
    )

    for message in result["messages"]:
        message.pretty_print()

if __name__ == "__main__":
    ask_tools_question()
