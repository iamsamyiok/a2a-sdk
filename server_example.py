# This is an example A2A server based on the helloworld example from the google/a2a-python SDK.
# Origin: https://github.com/google/a2a-python/tree/main/examples/helloworld
#
# This server can be used to test the `client.py` and `example_client_usage.py`.
#
# To run this server, you might need to install dependencies:
# pip install uvicorn a2a-sdk
#
# You may also need to ensure that `agent_executor.py` (which defines HelloWorldAgentExecutor)
# is in the same directory or accessible in your PYTHONPATH if you are running this
# script directly and it's not part of a larger package structure.
# For simplicity, the HelloWorldAgentExecutor class definition is included below
# if it's not found in the fetched content.
#
# The server will run on http://localhost:9999 by default.

from agent_executor import (
    HelloWorldAgentExecutor,  # type: ignore[import-untyped]
)

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    ChatMessage, # Added for appended executor
    ContentPart, # Added for appended executor
    SendMessageResponse, # Added for appended executor
    TaskContext # Added for appended executor
)
from a2a.server.agent_executor import AgentExecutor # Added for appended executor


if __name__ == '__main__':
    skill = AgentSkill(
        id='hello_world',
        name='Returns hello world',
        description='just returns hello world',
        tags=['hello world'],
        examples=['hi', 'hello world'],
    )

    extended_skill = AgentSkill(
        id='super_hello_world',
        name='Returns a SUPER Hello World',
        description='A more enthusiastic greeting, only for authenticated users.\n', # Cleaned up newline
        tags=['hello world', 'super', 'extended'],
        examples=['super hi', 'give me a super hello'],
    )

    # This will be the public-facing agent card
    public_agent_card = AgentCard(
        name='Hello World Agent',
        description='Just a hello world agent',
        url='http://localhost:9999/', # Default port for this example
        version='1.0.0',
        defaultInputModes=['text'],
        defaultOutputModes=['text'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],  # Only the basic skill for the public card
        supportsAuthenticatedExtendedCard=True,
    )

    # This will be the authenticated extended agent card
    # It includes the additional 'extended_skill'
    specific_extended_agent_card = public_agent_card.model_copy(
        update={
            'name': 'Hello World Agent - Extended Edition', # Different name for clarity
            'description': 'The full-featured hello world agent for authenticated users.', # Cleaned up newline
            'version': '1.0.1', # Could even be a different version
            # Capabilities and other fields like url, defaultInputModes, defaultOutputModes,
            # supportsAuthenticatedExtendedCard are inherited from public_agent_card unless specified here.
            'skills': [skill, extended_skill],  # Both skills for the extended card
        }
    )

    # If HelloWorldAgentExecutor is defined below, we use that.
    # Otherwise, this script expects it to be importable from agent_executor.py.
    # For this combined script, we will use the one defined below.
    request_handler = DefaultRequestHandler(
        agent_executor=HelloWorldAgentExecutor(), # This will use the class defined below
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(agent_card=public_agent_card,
                                     http_handler=request_handler,
                                     extended_agent_card=specific_extended_agent_card)
    import uvicorn

    uvicorn.run(server.build(), host='0.0.0.0', port=9999)


# --- Simple HelloWorldAgentExecutor (added for self-containment) ---
# from a2a.types import ChatMessage, ContentPart, SendMessageResponse, TaskContext # Moved to top
# from a2a.server.agent_executor import AgentExecutor # Moved to top

class HelloWorldAgentExecutor(AgentExecutor):
    async def send_message(
        self, messages: list[ChatMessage], task_context: TaskContext | None = None
    ) -> SendMessageResponse:
        # Simple echo for demonstration, or a fixed response
        user_message_text = "No specific message found."
        if messages and messages[-1].parts:
            last_part = messages[-1].parts[-1]
            if hasattr(last_part, 'text'): # Check if it's a TextPart or similar
                 user_message_text = last_part.text #type: ignore

        # Check if it's the "extended" skill being called via specific check
        # This is a simplified way to check; the original example might have more robust routing.
        is_super_hello = "super" in user_message_text.lower() # crude check

        if task_context and task_context.skill_id == "super_hello_world":
            response_text = f"SUPER Hello World! You said: '{user_message_text}' (via skill_id)"
        elif is_super_hello: # Fallback if skill_id not directly available/used in routing
            response_text = f"SUPER Hello World! You said: '{user_message_text}'"
        else:
            response_text = f"Hello World! You said: '{user_message_text}'"
        
        return SendMessageResponse(
            messages=[
                ChatMessage(
                    role="assistant",
                    parts=[ContentPart(text=response_text)],
                )
            ]
        )

    async def send_message_streaming(
        self, messages: list[ChatMessage], task_context: TaskContext | None = None
    ):
        # Simplified streaming response
        user_message_text = "No specific message found."
        if messages and messages[-1].parts:
            last_part = messages[-1].parts[-1]
            if hasattr(last_part, 'text'):
                 user_message_text = last_part.text #type: ignore
        
        is_super_hello = "super" in user_message_text.lower()
        base_response = "SUPER Hello Stream! " if is_super_hello else "Hello Stream! "
        
        yield SendMessageResponse(messages=[ChatMessage(role="assistant", parts=[ContentPart(text=base_response)])])
        yield SendMessageResponse(messages=[ChatMessage(role="assistant", parts=[ContentPart(text=f"You said: '{user_message_text}'. ")])])
        yield SendMessageResponse(messages=[ChatMessage(role="assistant", parts=[ContentPart(text="Streaming complete.")])])
# --- End of HelloWorldAgentExecutor ---
