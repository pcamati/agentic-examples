# agentic-examples

A collection of agent examples built with [LangGraph](https://github.com/langchain-ai/langgraph) and [LangChain](https://github.com/langchain-ai/langchain) to explore and test different features of agentic system design, including tool use, memory, multi-agent coordination, and local LLM integration via [Ollama](https://ollama.com).

---

## Prerequisites

Before setting up the project, make sure the following tools are installed on your system:

| Tool | Purpose | Install |
|------|---------|---------|
| [uv](https://docs.astral.sh/uv/) | Python package and project manager | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| [Ollama](https://ollama.com/download) | Run local LLMs | See [ollama.com/download](https://ollama.com/download) |
| Python ≥ 3.13 | Runtime | Managed automatically by `uv` |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-org/agentic-examples.git
cd agentic-examples
```

### 2. Create the virtual environment and install dependencies

`uv` will create a virtual environment and install all dependencies declared in `pyproject.toml`:

```bash
uv sync
```

To add new dependencies, use:

```bash
uv add <package-name>
```

### 3. Install pre-commit hooks

The project uses [pre-commit](https://pre-commit.com) to enforce code quality and consistent commit messages. Install the hooks once into your local clone:

```bash
uv run pre-commit install
```

This registers the hooks defined in [`.pre-commit-config.yaml`](.pre-commit-config.yaml):

| Hook | Stage | Description |
|------|-------|-------------|
| `ruff-check` | `pre-commit` | Lints and auto-fixes Python code |
| `ruff-format` | `pre-commit` | Formats Python code |

To run all hooks manually against every file:

```bash
uv run pre-commit run --all-files
```

---

## Ollama Setup

Examples in this repository use local language models served by Ollama. The default model is **llama3.1:8b**.

### 1. Pull the model

```bash
ollama pull llama3.1:8b
```

You can substitute any other supported model (e.g. `mistral`, `gemma3`, `qwen2.5`):

```bash
ollama pull <model-name>
```

### 2. Start the Ollama server

```bash
ollama serve
```

By default, Ollama listens on `http://localhost:11434`. LangChain's `ChatOllama` integration points to this address automatically.

### 3. Verify the model is available

```bash
ollama list
```

---

## Running Examples

With the environment set up and Ollama running, execute any example with `uv run`:

```bash
uv run langchain_llm_call.py
```

---

## Code Quality

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting, configured in `pyproject.toml` with a line length of 79. Run it directly at any time:

```bash
# Lint and auto-fix
uv run ruff check --fix .

# Format
uv run ruff format .
```


