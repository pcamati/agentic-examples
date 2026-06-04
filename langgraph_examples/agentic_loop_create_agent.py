"""Simple agentic loop in LangChain using create_agent."""

import mlflow.langchain
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from config.llm_model import LLM_MODEL
from utils.utils import save_mermaid_png

mlflow.langchain.autolog()


@tool
def get_current_temperature() -> int:
    """Get the current temperature for a given location."""
    return 22


TOOLS = [get_current_temperature]


def run(llm: BaseChatModel) -> None:
    """Run the example."""
    agent = create_agent(
        model=llm,
        tools=TOOLS,
        system_prompt=SystemMessage(
            "You are a helpful climate assistant. Respond the user query."
        ),
    )

    save_mermaid_png(agent, __file__)

    result = agent.invoke(
        {
            "messages": [
                HumanMessage("What is the current temperature in Paris?")
            ]
        }
    )

    messages = result["messages"]

    for message in messages:
        message.pretty_print()


if __name__ == "__main__":
    run(LLM_MODEL)
