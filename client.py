import typing
import httpx
import logging
import json
import os # Added for environment variable access
import openai # User needs to install this: pip install openai
from uuid import uuid4
from abc import ABC, abstractmethod
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import AgentCard, MessageSendParams, SendMessageRequest, SendStreamingMessageRequest

# Configure basic logging
# logging.basicConfig(level=logging.INFO) # Basic config is fine, but for libraries, it's often better to let the application configure logging.
logger = logging.getLogger(__name__)
if not logger.hasHandlers(): # Avoid adding multiple handlers if this module is reloaded.
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Constants for agent card paths
PUBLIC_AGENT_CARD_PATH = "/.well-known/agent.json" # Standard path for public agent card
DEFAULT_EXTENDED_AGENT_CARD_PATH = "/agent/authenticatedExtendedCard" # Common path for extended card


class AbstractLLMEngine(ABC):
    @abstractmethod
    async def process(self, user_input: str, agent_cards: typing.Dict[str, AgentCard]) -> typing.Tuple[typing.Optional[str], typing.Optional[typing.Dict[str, typing.Any]], typing.Optional[str]]:
        """
        Processes user input and available agent capabilities to decide on an action.

        Args:
            user_input: The input from the user.
            agent_cards: A dictionary mapping server_id to AgentCard.

        Returns:
            A tuple containing:
            - target_server_id: The ID of the server to send the message to, or None.
            - message_payload: The payload for the message, or None.
            - error_message: A string containing an error message if processing failed, or None.
        """
        pass


class OpenAIEngine(AbstractLLMEngine):
    def __init__(self, api_key: typing.Optional[str] = None, model: typing.Optional[str] = None):
        resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_api_key:
            err_msg = "OpenAI API key not provided and not found in environment variable OPENAI_API_KEY."
            logger.error(err_msg)
            raise ValueError(err_msg)
        self.api_key = resolved_api_key

        resolved_model = model or os.environ.get("OPENAI_MODEL_NAME") or "gpt-3.5-turbo"
        self.model = resolved_model
        
        self.openai_client = openai.AsyncOpenAI(api_key=self.api_key)
        logger.info(f"OpenAIEngine initialized with model: {self.model}")

    async def process(self, user_input: str, agent_cards: typing.Dict[str, AgentCard]) -> typing.Tuple[typing.Optional[str], typing.Optional[typing.Dict[str, typing.Any]], typing.Optional[str]]:
        system_prompt = (
            "You are an intelligent routing agent. Your task is to analyze the user's input and the capabilities of available agents. "
            "Choose the most suitable agent to handle the request and formulate a JSON message payload for that agent. "
            "The available agents and their descriptions/skills are as follows:\n\n"
        )

        if not agent_cards:
            system_prompt += "No agents are currently available.\n"
        else:
            for server_id, card in agent_cards.items():
                system_prompt += f"Agent ID: '{server_id}'\n"
                system_prompt += f"  Name: {card.name}\n"
                system_prompt += f"  Description: {card.description}\n"
                system_prompt += f"  API Version: {card.api_version}\n"
                # You might want to add more details from the card, e.g., specific capabilities if structured.
                system_prompt += "\n"

        system_prompt += (
            "\nBased on the user's input, decide which agent (by 'Agent ID') is best suited. "
            "If no agent is suitable, set 'server_id' to null. "
            "The message payload should be a JSON object. For text-based messages, use the format: "
            '`{"message": {"content": {"text": "USER_INPUT_ADAPTED_FOR_AGENT"}}}`, '
            "where USER_INPUT_ADAPTED_FOR_AGENT is the (potentially rephrased) user input. "
            "If including context, add a 'context' key like: "
            '`{"message": {"content": {"text": "..."}}, "context": {"conversationId": "some_id"}}`. '
            "If no agent is suitable, or if the user input is not something an agent should handle (e.g. a greeting), respond with "
            '`{"server_id": null, "payload": null}`.\n'
            "Your response MUST be a single JSON object adhering to this structure: "
            '`{"server_id": "AGENT_ID_OR_NULL", "payload": YOUR_JSON_PAYLOAD_OR_NULL}`.'
        )

        logger.debug(f"OpenAI System Prompt: {system_prompt}")
        logger.debug(f"User Input for OpenAI: {user_input}")

        try:
            completion = await self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input},
                ],
                response_format={"type": "json_object"}, # Enforce JSON output if model supports it
            )
            
            response_content = completion.choices[0].message.content
            logger.debug(f"OpenAI raw response: {response_content}")

            if not response_content:
                error_msg = "OpenAI returned an empty response."
                logger.error(error_msg)
                return None, None, error_msg

            parsed_response = json.loads(response_content)
            server_id = parsed_response.get("server_id")
            payload = parsed_response.get("payload")

            if server_id is None and payload is None: # Explicitly no action
                 logger.info("LLM determined no specific agent action is needed.")
                 return None, None, None


            # Basic validation, can be expanded
            if not isinstance(server_id, str) and server_id is not None:
                 error_msg = f"LLM returned invalid server_id type: {type(server_id)}"
                 logger.error(error_msg)
                 return None, None, error_msg
            
            if not isinstance(payload, dict) and payload is not None:
                 error_msg = f"LLM returned invalid payload type: {type(payload)}"
                 logger.error(error_msg)
                 return None, None, error_msg
            
            if server_id and server_id not in agent_cards:
                error_msg = f"LLM chose server_id '{server_id}' but this agent is not available in agent_cards."
                logger.error(error_msg)
                # Potentially allow sending anyway, or strictly return error
                return None, None, error_msg


            logger.info(f"LLM decision: server_id='{server_id}', payload='{payload}'")
            return server_id, payload, None

        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse JSON response from OpenAI: {e}. Response was: {response_content}"
            logger.error(error_msg)
            return None, None, error_msg
        except openai.APIError as e:
            error_msg = f"OpenAI API error: {e}"
            logger.error(error_msg)
            return None, None, error_msg
        except Exception as e:
            error_msg = f"An unexpected error occurred in OpenAIEngine: {e}"
            logger.error(error_msg, exc_info=True)
            return None, None, error_msg


class MultiAgentA2AClient:
    """
    Client for interacting with multiple agent servers and facilitating agent-to-agent communication.
    """

    def __init__(self):
        """
        Initializes the MultiAgentA2AClient.
        """
        self.agent_servers_config: typing.Dict[str, typing.Dict[str, typing.Any]] = {}
        self.agent_cards: typing.Dict[str, AgentCard] = {}  # Stores fetched agent cards (server_id -> AgentCard)
        self.llm_engine: typing.Optional[AbstractLLMEngine] = None
        self.http_client = httpx.AsyncClient()

    def add_agent_server(self, server_id: str, base_url: str, auth_token: str = None):
        """
        Adds or updates an agent server configuration.

        Args:
            server_id: A unique identifier for the agent server.
            base_url: The base URL of the agent server.
            auth_token: Optional authentication token for the server.
        """
        self.agent_servers_config[server_id] = {
            "base_url": base_url,
            "auth_token": auth_token,
        }
        logger.info(f"Added/Updated server config for server_id '{server_id}': base_url='{base_url}'")

    def load_server_configs_from_file(self, filepath: str) -> bool:
        """
        Loads agent server configurations from a JSON file.

        Args:
            filepath: Path to the JSON file.
                      The file should contain an array of objects, each with
                      "server_id", "base_url", and optionally "auth_token".

        Returns:
            True if configurations were loaded successfully, False otherwise.
        """
        logger.info(f"Attempting to load server configurations from '{filepath}'...")
        try:
            with open(filepath, 'r') as f:
                configs = json.load(f)
            
            if not isinstance(configs, list):
                logger.error(f"Error loading server configs: File '{filepath}' should contain a JSON array.")
                return False

            for config in configs:
                server_id = config.get("server_id")
                base_url = config.get("base_url")
                auth_token = config.get("auth_token") # Will be None if not present

                if not server_id or not base_url:
                    logger.warning(f"Skipping invalid server config entry in '{filepath}': {config}. 'server_id' and 'base_url' are required.")
                    continue
                
                self.add_agent_server(server_id, base_url, auth_token)
            
            logger.info(f"Successfully loaded {len(configs)} server configurations from '{filepath}'.")
            return True
        except FileNotFoundError:
            logger.error(f"Error loading server configs: File '{filepath}' not found.")
            return False
        except json.JSONDecodeError as e:
            logger.error(f"Error loading server configs: Could not parse JSON from '{filepath}'. Error: {e}")
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading server configs from '{filepath}': {e}", exc_info=True)
            return False

    def initialize_llm_engine(self, engine_type: str, api_key: typing.Optional[str] = None, model: typing.Optional[str] = None):
        """
        Initializes the LLM engine.

        Args:
            engine_type: The type of LLM engine to use (e.g., "openai").
            api_key: Optional API key for the LLM service. If None, the engine may try env vars.
            model: Optional model name for the LLM service. If None, the engine may try env vars or use a default.
        """
        if engine_type.lower() == "openai":
            try:
                self.llm_engine = OpenAIEngine(api_key=api_key, model=model) # Pass None values directly
                logger.info(f"OpenAI LLM Engine initialization attempt with provided or environment-based config.")
            except ValueError as ve: # Catch specific error from OpenAIEngine if API key is missing
                logger.error(f"Failed to initialize OpenAIEngine: {ve}")
                self.llm_engine = None
            except Exception as e:
                logger.error(f"Failed to initialize OpenAIEngine with an unexpected error: {e}", exc_info=True)
                self.llm_engine = None
        else:
            logger.error(f"Unsupported LLM engine type: {engine_type}")
            self.llm_engine = None

    async def process_user_request(self, user_input: str) -> typing.Any:
        """
        Processes a user request using the initialized LLM engine and sends a message if indicated.
        """
        if not self.llm_engine:
            error_msg = "LLM engine not initialized. Please call initialize_llm_engine first."
            logger.error(error_msg)
            return error_msg

        logger.info(f"Processing user request with LLM: '{user_input}'")
        target_server_id, message_payload, error = await self.llm_engine.process(user_input, self.agent_cards)

        if error:
            logger.error(f"LLM processing error: {error}")
            return f"LLM processing error: {error}"

        if target_server_id and message_payload:
            logger.info(f"LLM decided to send message to '{target_server_id}' with payload: {message_payload}")
            # For now, defaulting to unary. Streaming decision could be part of LLM output or other logic.
            # Also, ensure the payload structure from LLM matches send_unary_message expectations.
            # The LLM is prompted to create payload like: {"message": {"content": {"text": "..."}}}
            # which is what send_unary_message expects for its `payload` argument.
            response = await self.send_unary_message(target_server_id, message_payload)
            return response
        else:
            no_action_msg = "No suitable agent action determined by LLM."
            logger.info(no_action_msg)
            return no_action_msg


    async def discover_agent(self, server_id: str) -> typing.Optional[AgentCard]:
        """
        Discovers an agent by fetching its agent card.

        Args:
            server_id: The ID of the agent server to discover.

        Returns:
            The AgentCard if discovered, otherwise None.
        """
        server_config = self.agent_servers_config.get(server_id)
        if not server_config:
            logger.error(f"Configuration for server_id '{server_id}' not found.")
            return None

        base_url = server_config["base_url"]
        auth_token = server_config.get("auth_token")
        resolver = A2ACardResolver(client=self.http_client, base_url=base_url)
        final_agent_card_to_use: typing.Optional[AgentCard] = None

        logger.info(f"Attempting to fetch public agent card for server_id '{server_id}' from {base_url}{PUBLIC_AGENT_CARD_PATH}")
        try:
            # Fetch public card. Note: A2ACardResolver uses PUBLIC_AGENT_CARD_PATH by default.
            public_card = await resolver.get_agent_card()
            if public_card:
                logger.info(f"Successfully fetched public agent card for server_id '{server_id}'.")
                final_agent_card_to_use = public_card

                if public_card.supports_authenticated_extended_card and auth_token:
                    extended_card_path = public_card.authenticated_extended_card_path or DEFAULT_EXTENDED_AGENT_CARD_PATH
                    logger.info(f"Agent '{server_id}' supports extended card. Attempting to fetch from {base_url}{extended_card_path}.")
                    try:
                        headers = {"Authorization": f"Bearer {auth_token}"}
                        extended_card = await resolver.get_agent_card(
                            relative_card_path=extended_card_path,
                            http_kwargs={'headers': headers}
                        )
                        if extended_card:
                            logger.info(f"Successfully fetched authenticated extended agent card for server_id '{server_id}'.")
                            final_agent_card_to_use = extended_card
                        else:
                            logger.warning(f"Failed to fetch extended agent card for server_id '{server_id}', but public card is available.")
                    except Exception as e:
                        logger.error(f"Error fetching extended agent card for server_id '{server_id}': {e}. Using public card.")
                elif public_card.supports_authenticated_extended_card and not auth_token:
                    logger.warning(f"Agent '{server_id}' supports extended card, but no auth_token provided. Using public card.")
                else:
                    logger.info(f"Agent '{server_id}' does not support extended card or no auth token. Using public card.")
            else:
                logger.error(f"Failed to fetch public agent card for server_id '{server_id}'. No card data returned.")
        except Exception as e:
            logger.error(f"Exception occurred while fetching public agent card for server_id '{server_id}': {e}")
            return None

        if final_agent_card_to_use:
            self.agent_cards[server_id] = final_agent_card_to_use
            logger.info(f"Stored agent card for server_id '{server_id}'.")
        else:
            logger.warning(f"No agent card could be obtained for server_id '{server_id}'.")

        return final_agent_card_to_use

    def get_agent_card(self, server_id: str) -> typing.Optional[AgentCard]:
        """
        Retrieves a previously discovered agent card.

        Args:
            server_id: The ID of the agent server.

        Returns:
            The AgentCard if found, otherwise None.
        """
        return self.agent_cards.get(server_id)

    async def send_unary_message(
        self, server_id: str, payload: typing.Dict[str, typing.Any]
    ) -> typing.Optional[typing.Dict[str, typing.Any]]:
        """
        Sends a unary (non-streaming) message to a specified agent server.

        Args:
            server_id: The ID of the target agent server.
            payload: The message payload. Must include a 'message' key.
                     If payload['message']['messageId'] is not present, it will be generated.

        Returns:
            The response from the agent server as a dictionary, or None if an error occurs.
        """
        agent_card = self.get_agent_card(server_id)
        if not agent_card:
            logger.error(f"No agent card found for server_id '{server_id}'. Cannot send unary message.")
            return None

        a2a_client = A2AClient(client=self.http_client, agent_card=agent_card)

        # Ensure messageId exists
        if not payload.get("message") or not isinstance(payload["message"], dict):
            logger.error(f"Payload for server_id '{server_id}' must contain a 'message' dictionary.")
            return None
        if "messageId" not in payload["message"]:
            payload["message"]["messageId"] = uuid4().hex
            logger.info(f"Generated messageId {payload['message']['messageId']} for unary message to '{server_id}'.")

        try:
            params = MessageSendParams(**payload)
            request = SendMessageRequest(params=params)
            logger.info(f"Sending unary message to '{server_id}': {request.model_dump(mode='json', exclude_none=True)}")
            response = await a2a_client.send_message(request)
            logger.info(f"Received response from '{server_id}': {response.model_dump(mode='json', exclude_none=True)}")
            return response.model_dump(mode='json', exclude_none=True)
        except Exception as e:
            logger.error(f"Error sending unary message to server_id '{server_id}': {e}")
            return None

    async def send_streaming_message(
        self, server_id: str, payload: typing.Dict[str, typing.Any]
    ) -> typing.Optional[typing.AsyncGenerator[typing.Dict[str, typing.Any], None]]:
        """
        Sends a message to a specified agent server and streams the response.

        Args:
            server_id: The ID of the target agent server.
            payload: The message payload. Must include a 'message' key.
                     If payload['message']['messageId'] is not present, it will be generated.

        Returns:
            An async generator yielding response chunks, or None if the agent card is not found.
        """
        agent_card = self.get_agent_card(server_id)
        if not agent_card:
            logger.error(f"No agent card found for server_id '{server_id}'. Cannot send streaming message.")
            return None

        a2a_client = A2AClient(client=self.http_client, agent_card=agent_card)

        if not payload.get("message") or not isinstance(payload["message"], dict):
            logger.error(f"Payload for server_id '{server_id}' must contain a 'message' dictionary.")
            # This path should ideally not return an async generator, but None,
            # as the contract is Optional[AsyncGenerator].
            # However, changing the return type here makes the function signature complex.
            # For now, we log and the generator created below will simply be empty if this path is hit
            # by a programming error (as agent_card check would have returned None already).
            # A more robust solution might raise an error or ensure this state is impossible.
            return None


        if "messageId" not in payload["message"]:
            payload["message"]["messageId"] = uuid4().hex
            logger.info(f"Generated messageId {payload['message']['messageId']} for streaming message to '{server_id}'.")

        params = MessageSendParams(**payload)
        request = SendStreamingMessageRequest(params=params)
        logger.info(f"Sending streaming message to '{server_id}': {request.model_dump(mode='json', exclude_none=True)}")

        async def generator():
            try:
                async for chunk in a2a_client.send_message_streaming(request):
                    yield chunk.model_dump(mode='json', exclude_none=True)
            except Exception as e:
                logger.error(f"Error during streaming message from server_id '{server_id}': {e}")
                # Stop iteration
        
        return generator()
