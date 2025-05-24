import asyncio
import logging
import json # For pretty printing responses
from client import MultiAgentA2AClient # Import the main client class

# Setup logger for this example script
logger = logging.getLogger(__name__) # This will be __main__ if script is run directly
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

async def main_example():
    # 1. Ensure Comprehensive Setup
    # Logger for this script is configured above.
    # MultiAgentA2AClient will use its own logger internally (named 'client').
    
    logger.info("Starting Multi-Agent A2A Client Interactive Example...")
    client = MultiAgentA2AClient()

    # 2. Configuration Loading
    servers_config_filepath = "servers_config.json"
    logger.info(f"Attempting to load server configurations from '{servers_config_filepath}'.")
    logger.info("Please create this file in the same directory as the script if it doesn't exist.")
    logger.info("Example servers_config.json content (as shown in README.md):")
    logger.info("""
[
  {
    "server_id": "helloworld_agent", 
    "base_url": "http://localhost:9999" 
  },
  {
    "server_id": "calculator_agent",
    "base_url": "http://localhost:8000",
    "auth_token": "your_auth_token_for_calculator_if_needed"
  }
]
    """)

    if client.load_server_configs_from_file(servers_config_filepath):
        logger.info(f"Successfully loaded server configurations from '{servers_config_filepath}'.")
    else:
        logger.warning(f"Could not load '{servers_config_filepath}', or it's empty/invalid. Client may have no pre-configured servers.")
        # Optionally, add a default mock server for demonstration if no config is found
        # For example:
        # client.add_agent_server("default_mock_agent", "http://localhost:12345")
        # logger.info("Added 'default_mock_agent' as no configurations were loaded from file.")

    # 3. LLM Engine Initialization
    logger.info("\nInitializing LLM engine...")
    # Initialize LLM engine, relying on environment variables for API key and model
    # as configured in OpenAIEngine within client.py
    client.initialize_llm_engine(engine_type="openai") 
    if client.llm_engine:
        logger.info("LLM engine initialization process completed (check client logs for details on API key and model).")
    else:
        logger.warning("LLM engine could not be initialized. Ensure OPENAI_API_KEY is set in your environment (see client logs).")
        logger.warning("Queries requiring LLM routing might not work as expected.")
    logger.info("Reminder: Set OPENAI_API_KEY environment variable for LLM features.")
    logger.info("Optionally, set OPENAI_MODEL_NAME (defaults to gpt-3.5-turbo if not set).\n")


    # 4. Agent Discovery
    logger.info("Attempting to discover all configured agents...")
    if not client.agent_servers_config:
        logger.info("No agent servers configured to discover.")
    else:
        for server_id in client.agent_servers_config.keys():
            logger.info(f"Discovering agent: '{server_id}'...")
            card = await client.discover_agent(server_id)
            if card:
                logger.info(f"Successfully discovered agent '{server_id}': {card.name} (API: {card.api_version})")
            else:
                logger.warning(f"Failed to discover agent '{server_id}'. It might be offline or not A2A compliant.")
    
    if not client.agent_cards:
            logger.warning("No agents were discovered or their cards stored. LLM may not be able to route requests effectively.")
    logger.info("-" * 50)

    # 5. Interactive Loop
    logger.info("Starting interactive query loop...\n")
    try:
        while True:
            user_query = input("Enter your query (or 'quit' to exit): ")
            if user_query.lower() == 'quit':
                logger.info("Exiting interactive loop.")
                break
            
            if not user_query.strip():
                logger.info("Query is empty, please try again.")
                continue

            logger.info(f"Processing query: \"{user_query}\"")
            try:
                response = await client.process_user_request(user_query)
                # The logger in client.py (MultiAgentA2AClient and OpenAIEngine) will log details
                # of LLM decision and agent communication.
                # Here, we just print the final outcome to the user.
                print("\n>>> Client Response <<<")
                if isinstance(response, dict):
                    # Pretty print JSON if it's a dictionary (common for successful agent responses)
                    print(json.dumps(response, indent=2, sort_keys=True))
                elif isinstance(response, str):
                    # Print directly if it's a string (e.g., "No suitable agent..." or error message from client)
                    print(response)
                else:
                    # Fallback for other types
                    print(f"Received response of type {type(response)}: {response}")
                print(">>> End of Response <<<\n")

            except Exception as e:
                logger.error(f"An error occurred in the interactive loop while processing your request: {e}", exc_info=True)
                print(f"Error processing your request: {e}")
            print("-" * 30) # Separator for readability
    
    except KeyboardInterrupt:
        logger.info("\nUser interrupted. Exiting.")
    finally:
        # 6. HTTP Client Closure (ensure this happens)
        logger.info("Closing HTTP client...")
        await client.http_client.aclose()
        logger.info("HTTP client closed. Example finished.")

if __name__ == "__main__":
    asyncio.run(main_example())
