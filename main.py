# 1. Environment & Logging Setup (Must be first)
import os
import logging
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv is not installed or not needed in this environment (e.g. Lambda with pre-configured env vars)
    pass

# Force DEV mode if not explicitly set to 'false' (prevents production hangs)
# We do this BEFORE importing orca to ensure the factory picks it up.
dev_mode_val = os.environ.get("ORCA_DEV_MODE", "true").lower()
is_dev_mode = dev_mode_val == "true"
# os.environ["ORCA_DEV_MODE"] = "true" if is_dev_mode else "false"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 2. Orca & AI Imports
import asyncio
from openai import AsyncOpenAI
from httpx import Timeout
from orca import (
    OrcaHandler, 
    ChatMessage, 
    Variables,
    create_agent_app,
    SessionContext
)
from function_handler import get_available_functions, process_function_calls

# Simple conversation memory (in-memory)
conversation_memory = {}

# Global components
orca_handler_instance = None

async def process_message(data: ChatMessage) -> None:
    """
    Core logic for processing incoming messages.
    Uses SessionContext to manage the agent's interaction with the Orca platform.
    """
    global orca_handler_instance
    
    # Ensure we use the global handler or fallback to a safe default
    handler = orca_handler_instance or OrcaHandler(dev_mode=True)
    
    # Use SessionContext as a context manager for automatic setup/teardown
    with SessionContext(handler, data) as session:
        try:
            logger.info(f"🚀 Processing message for thread {data.thread_id}")
            
            # Variables & Authentication
            vars = Variables(data.variables)
            # Strategy: Variables from request have precedence over environment
            api_key = vars.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
            
            if not api_key:
                logger.error("❌ OpenAI API key is missing in both variables and environment.")
                return session.error("OpenAI API key missing.")

            # AsyncOpenAI is preferred for non-blocking IO in FastAPI
            # Configure timeout to prevent hanging requests (60s connect, 300s read/write)
            timeout = Timeout(60.0, connect=30.0, read=300.0, write=300.0, pool=30.0)
            async with AsyncOpenAI(api_key=api_key, timeout=timeout) as client:
                # Thread Identity & context
                thread_id = data.thread_id
                if thread_id not in conversation_memory:
                    conversation_memory[thread_id] = []
                
                # Append user message to local history
                conversation_memory[thread_id].append({"role": "user", "content": data.message})
                
                # Prepare message history for LLM (limited to last 10 messages)
                messages = [
                    {
                        "role": "system", 
                        "content": (
                            "You are a helpful AI assistant. Use the available tools to provide a rich UI experience: "
                            "generate_image for images, send_video for videos, send_audio for audio, send_location for maps, "
                            "send_trace for debugging, send_buttons for interaction, send_card_list for lists, "
                            "and track_usage for token tracking."
                        )
                    }
                ]
                messages.extend(conversation_memory[thread_id][-10:])
                session.loading.start("thinking")
                
                try:
                    # Initiate Streaming Chat Completion
                    stream = await client.chat.completions.create(
                        model=data.model or "gpt-4o",
                        messages=messages,
                        tools=get_available_functions(),
                        stream=True
                    )
                    
                    full_response = ""
                    function_calls = []
                    
                    # Process the stream asynchronously
                    async for chunk in stream:
                        if not chunk.choices:
                            continue
                            
                        delta = chunk.choices[0].delta
                        
                        # Handle text content
                        if delta.content:
                            content = delta.content
                            full_response += content
                            session.stream(content)
                        
                        # Handle function calling deltas
                        if delta.tool_calls:
                            for tc in delta.tool_calls:
                                if tc.function:
                                    if len(function_calls) <= tc.index:
                                        # Start of a new function call
                                        function_calls.append({
                                            "id": tc.id, 
                                            "function": {"name": tc.function.name, "arguments": ""}
                                        })
                                        session.stream(f"\n🔧 **Calling:** {tc.function.name}")
                                    # Accumulate JSON arguments
                                    if tc.function.arguments:
                                        function_calls[tc.index]["function"]["arguments"] += tc.function.arguments

                    # Execute planned tool calls
                    if function_calls:
                        # Small "finalizing" state before tools
                        await asyncio.sleep(1) # Visibility delay
                        fn_res, _ = await process_function_calls(function_calls, session)
                        full_response += (fn_res or "")
                    
                    # Save assistant response to history
                    conversation_memory[thread_id].append({"role": "assistant", "content": full_response})
                    
                finally:
                    # Always stop loading, even if error occurs
                    session.loading.end("thinking")
            
        except Exception as e:
            logger.error(f"❌ Error in process_message: {e}", exc_info=True)
            # Report error back to the Orca session
            session.error(f"Execution Error: {str(e)}")

# Initialize the FAST API application with Orca factory
app, orca_handler_instance = create_agent_app(
    process_message_func=process_message,
    title="Dummy Orca Agent",
    description="A simple AI agent with advanced UI capabilities and Orca SDK integration",
    version="1.0.4"
)

# # Overwrite handler dev_mode to sync with our detection
# if orca_handler_instance:
#     orca_handler_instance.dev_mode = is_dev_mode

if __name__ == "__main__":
    import uvicorn
    logger.info(f"🚀 Starting Orca Agent (is_dev_mode={is_dev_mode})...")
    uvicorn.run(app, host="0.0.0.0", port=5001)
