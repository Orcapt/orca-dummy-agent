"""
Simple AI Agent with Chat Completion
===================================

A minimal AI agent that demonstrates basic chat completion functionality
using the Lexia platform with streaming support.

Key Features:
- Simple chat completion using OpenAI
- Real-time response streaming via Lexia
- Basic conversation memory
- Clean, minimal implementation

Usage:
    python main.py

The server will start on http://localhost:8000 with the following endpoints:
- POST /api/v1/send_message - Main chat endpoint
- GET /api/v1/health - Health check
- GET /api/v1/ - Root information
- GET /api/v1/docs - Interactive API documentation

Author: Lexia Team
License: MIT
"""

import logging
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import Lexia components
from lexia import (
    LexiaHandler, 
    ChatMessage, 
    Variables,
    create_lexia_app,
    add_standard_endpoints
)

# Import function handler for image generation
from function_handler import get_available_functions, process_function_calls

# Simple conversation memory (in-memory)
conversation_memory = {}

# Initialize Lexia handler
lexia = LexiaHandler()

# Create the FastAPI app
app = create_lexia_app(
    title="Simple AI Agent",
    version="1.0.0",
    description="A simple AI agent with chat completion functionality"
)

async def process_message(data: ChatMessage) -> None:
    """
    Process incoming chat messages using OpenAI and send responses via Lexia.
    
    This is a simplified version that focuses on basic chat completion
    with streaming support.
    
    Args:
        data: ChatMessage object containing the incoming message and metadata
    """
    try:
        logger.info(f"🚀 Processing message for thread {data.thread_id}")
        logger.info(f"📝 Message: {data.message[:100]}...")
        
        # Get OpenAI API key
        vars = Variables(data.variables)
        openai_api_key = vars.get("OPENAI_API_KEY")
        if not openai_api_key:
            error_msg = "Sorry, the OpenAI API key is missing. Please configure it in the agent settings."
            logger.error("OpenAI API key not found")
            lexia.stream_chunk(data, error_msg)
            lexia.complete_response(data, error_msg)
            return
        
        # Initialize OpenAI client
        client = OpenAI(api_key=openai_api_key)
        
        # Get conversation history for this thread
        thread_id = data.thread_id
        if thread_id not in conversation_memory:
            conversation_memory[thread_id] = []
        
        # Add user message to memory
        conversation_memory[thread_id].append({"role": "user", "content": data.message})
        
        # Prepare messages for OpenAI
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant. You can generate dummy images for testing when users ask for images. Use the generate_image function to return a demo image URL based on user descriptions."}
        ]
        
        # Add conversation history (keep last 10 messages)
        history = conversation_memory[thread_id][-10:]
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        
        logger.info(f"🤖 Sending to OpenAI model: {data.model}")
        
        # Get available functions from function handler
        available_functions = get_available_functions()
        
        # Stream response from OpenAI with function calling support
        stream = client.chat.completions.create(
            model=data.model,
            messages=messages,
            tools=available_functions,
            tool_choice="auto",
            max_tokens=1000,
            temperature=0.7,
            stream=True
        )
        
        # Process streaming response
        full_response = ""
        usage_info = None
        function_calls = []
        generated_image_url = None
        
        logger.info("📡 Streaming response from OpenAI...")
        
        for chunk in stream:
            # Handle content chunks
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                # Stream chunk to Lexia
                lexia.stream_chunk(data, content)
            
            # Handle function call chunks
            if chunk.choices[0].delta.tool_calls:
                logger.info(f"🔧 Tool call chunk detected: {chunk.choices[0].delta.tool_calls}")
                for tool_call in chunk.choices[0].delta.tool_calls:
                    if tool_call.function:
                        # Initialize function call if it's new
                        if len(function_calls) <= tool_call.index:
                            function_calls.append({
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": ""
                                }
                            })
                            logger.info(f"🔧 New function call initialized: {tool_call.function.name}")
                            
                            # Stream function call announcement to Lexia
                            function_msg = f"\n🔧 **Calling function:** {tool_call.function.name}"
                            lexia.stream_chunk(data, function_msg)
                        
                        # Accumulate function arguments
                        if tool_call.function.arguments:
                            function_calls[tool_call.index]["function"]["arguments"] += tool_call.function.arguments
                            logger.info(f"🔧 Accumulated arguments for function {tool_call.index}: {tool_call.function.arguments}")
            
            # Capture usage information
            if chunk.usage:
                usage_info = chunk.usage
                logger.info(f"📊 Usage info: {usage_info}")
        
        logger.info(f"✅ OpenAI response complete. Length: {len(full_response)} characters")
        
        # Process function calls if any were made using the function handler
        function_result, generated_image_url = await process_function_calls(function_calls, lexia, data)
        if function_result:
            full_response += function_result
        
        logger.info(f"🖼️ Final generated_image_url value: {generated_image_url}")
        
        # Add assistant response to memory
        conversation_memory[thread_id].append({"role": "assistant", "content": full_response})
        
        # Send complete response to Lexia with full data structure
        logger.info("📤 Sending complete response to Lexia...")
        
        # Include generated image in the response if one was created
        if generated_image_url:
            logger.info(f"🖼️ Including generated image in API call: {generated_image_url}")
            # Use the complete_response method that includes the file field
            lexia.complete_response(data, full_response, usage_info, file_url=generated_image_url)
        else:
            # Normal response without image
            lexia.complete_response(data, full_response, usage_info)
        
        logger.info(f"🎉 Message processing completed for thread {data.thread_id}")
            
    except Exception as e:
        error_msg = f"Error processing message: {str(e)}"
        logger.error(error_msg, exc_info=True)
        lexia.send_error(data, error_msg)

# Add standard Lexia endpoints
add_standard_endpoints(
    app, 
    conversation_manager=None,  # Using simple in-memory storage
    lexia_handler=lexia,
    process_message_func=process_message
)

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting Simple AI Agent...")
    print("=" * 50)
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/api/v1/health")
    print("💬 Chat Endpoint: http://localhost:8000/api/v1/send_message")
    print("=" * 50)
    print("\n✨ This simple agent demonstrates:")
    print("   - Basic chat completion with OpenAI")
    print("   - Dummy image generation via function calling")
    print("   - Real-time streaming via Lexia")
    print("   - Simple conversation memory")
    print("   - Clean, minimal implementation")
    print("\n🔧 Perfect for testing Lexia functionality!")
    print("=" * 50)
    
    # Start the FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000)
