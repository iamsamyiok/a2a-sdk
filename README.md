# Overview

This script implements a Python client for interacting with multiple Agent-to-Agent (A2A)
protocol-compliant agent servers. It uses an OpenAI-like Large Language Model (LLM)
to intelligently decide which agent to route a user's request to and what message
to send based on the agents' advertised capabilities (Agent Cards).

# Features

- Connects to and manages configurations for multiple A2A agent servers.
- Fetches and understands agent capabilities via public and extended Agent Cards.
- Integrates with an OpenAI LLM (configurable model via environment variable) for:
    - Intelligent routing of user requests to the most appropriate agent.
    - Formulation of message payloads based on user input and agent capabilities.
- Configuration:
    - Agent servers are configured via a `servers_config.json` file.
    - LLM parameters (API key, model name) are configured via environment variables.
- Supports sending both unary (request-response) and streaming messages to agents
  (though the current LLM integration defaults to unary for simplicity).
- Provides an interactive command-line example for demonstrating its capabilities.

# Requirements/Dependencies

- Python 3.10+
- `a2a-sdk`: The official A2A Python SDK for client-server communication.
- `openai`: The official OpenAI Python library for LLM interaction.
- `httpx`: An asynchronous HTTP client, used by both `a2a-sdk` and `openai` library.

# Setup and Configuration

## 1. Installation

Install the necessary Python packages using pip:
```bash
pip install a2a-sdk openai httpx
```

## 2. Environment Variables

Set the following environment variables in your system or shell:
- `OPENAI_API_KEY`: **Required**. Your API key for the OpenAI service.
  Example: `export OPENAI_API_KEY="sk-your_openai_api_key_here"`
- `OPENAI_MODEL_NAME`: Optional. The specific OpenAI model to use
  (e.g., "gpt-4", "gpt-3.5-turbo-0125").
  Defaults to "gpt-3.5-turbo" if not set.
  Example: `export OPENAI_MODEL_NAME="gpt-4"`

## 3. Server Configuration File (`servers_config.json`)

- Create a file named `servers_config.json` in the same directory as the `example_client_usage.py` script (or the directory where you intend to run the example).
- This file defines the A2A agent servers that the client can connect to and discover.
- Format: A JSON array of objects. Each object represents an agent server and must contain:
    - `server_id` (string): A unique identifier you assign to this server
      (e.g., "calculator_agent", "weather_reporter").
    - `base_url` (string): The base URL of the A2A agent server
      (e.g., "http://localhost:8000", "https://api.exampleagent.com").
    - `auth_token` (string, optional): An authentication token (e.g., Bearer token)
      if the agent server requires it for accessing its extended agent card or
      for sending messages.

- Example `servers_config.json`:
  ```json
  [
    {
      "server_id": "helloworld_agent",
      "base_url": "http://localhost:9999"
    },
    {
      "server_id": "calculator_agent_prod",
      "base_url": "https://my.calculatorapi.com/a2a",
      "auth_token": "secret-bearer-token-for-calculator"
    },
    {
      "server_id": "local_dev_agent",
      "base_url": "http://127.0.0.1:8001"
    }
  ]
  ```

# Running the Example

1.  Ensure you have created the `servers_config.json` file as described above.
2.  Ensure the `OPENAI_API_KEY` environment variable is set.
3.  Run the example script from your terminal:
    ```bash
    python example_client_usage.py
    ```
4.  The `example_client_usage.py` script, which imports and uses the `MultiAgentA2AClient` from `client.py`, will attempt to load server configurations, initialize the LLM, discover configured agents, and then enter an interactive loop.
5.  Type your queries at the prompt. The client will use the LLM to (try to) select an appropriate agent and send a message. Type 'quit' to exit.

# Code Structure Overview

- `client.py`: Contains the core library code.
    - `MultiAgentA2AClient`: The main class orchestrating all operations. It manages
      agent server configurations, performs agent discovery (fetching Agent Cards),
      sends messages (unary/streaming), and interacts with the LLM engine.
    - `AbstractLLMEngine`: An abstract base class defining the interface for any
      LLM processing engine. Its core method `process` takes user input and available
      agent cards to decide on an action (target agent and message payload).
    - `OpenAIEngine`: A concrete implementation of `AbstractLLMEngine` that uses the
      OpenAI API (via the `openai` library) to make decisions. It constructs a
      system prompt to guide the LLM.
- `example_client_usage.py`: (To be created in a subsequent step) This script will provide the interactive command-line interface (formerly part of `multi_agent_a2a_client.py`'s `__main__` block) to demonstrate the client's functionality. It will handle setup, discovery, and user input processing by importing and utilizing the `MultiAgentA2AClient` from `client.py`.
