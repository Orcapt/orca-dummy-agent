"""
Function Handler Module for Simple AI Agent
==========================================

This module handles function calling capabilities for the simple AI agent.
It contains the function definitions and execution logic for image generation.

Key Features:
- DALL-E 3 image generation function
- Function schema definitions
- Function execution and error handling
- Streaming progress updates to Lexia

Author: Lexia Team
License: MIT
"""

import asyncio
import logging
import os
import json
from openai import OpenAI
from lexia import Variables

# Configure logging
logger = logging.getLogger(__name__)

# Available functions schema for OpenAI
AVAILABLE_FUNCTIONS = [
    {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": "Generate a dummy image for testing purposes (returns a fixed demo image URL)",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "A detailed description of the image you want to generate. Be specific about style, colors, composition, and mood."
                    },
                    "size": {
                        "type": "string",
                        "enum": ["1024x1024", "1792x1024", "1024x1792"],
                        "description": "The size of the generated image. 1024x1024 is square, 1792x1024 is landscape, 1024x1792 is portrait."
                    },
                    "quality": {
                        "type": "string",
                        "enum": ["standard", "hd"],
                        "description": "Image quality. HD is higher quality but costs more."
                    },
                    "style": {
                        "type": "string",
                        "enum": ["vivid", "natural"],
                        "description": "Image style. Vivid is more dramatic, natural is more realistic."
                    }
                },
                "required": ["prompt"]
            }
        }
    }
]

async def generate_image_with_dalle(
    prompt: str, 
    variables: list = None,
    size: str = "1024x1024", 
    quality: str = "standard", 
    style: str = "vivid"
) -> str:
    """
    Generate a dummy image URL for testing purposes.
    
    This function simulates image generation by returning a fixed dummy image URL
    instead of actually calling DALL-E 3. Perfect for testing and demonstration.
    
    Args:
        prompt: Detailed text description of the image to generate (used for logging)
        variables: List of variables containing API keys (not used in dummy mode)
        size: Image dimensions (not used in dummy mode)
        quality: Image quality (not used in dummy mode)
        style: Image style (not used in dummy mode)
    
    Returns:
        str: Fixed dummy image URL
        
    Example:
        >>> image_url = await generate_image_with_dalle("A beautiful sunset")
        >>> print(f"Dummy image URL: {image_url}")
    """
    try:
        # Fixed dummy image URL for testing
        dummy_image_url = "https://fsn1.your-objectstorage.com/lexia-production/demo/images/generated_1758945096.png?X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=TQUW1QDE6DNCEW3SSSMX%2F20250927%2Ffsn1%2Fs3%2Faws4_request&X-Amz-Date=20250927T035136Z&X-Amz-SignedHeaders=host&X-Amz-Expires=604800&X-Amz-Signature=b5a8cfa4c98f6efb302d53837468a3106e69a41d0acded2a6bfbb06a66e1f609"
        
        logger.info(f"🎨 [DUMMY MODE] Simulating image generation for prompt: {prompt}")
        logger.info(f"🎨 [DUMMY MODE] Parameters - Size: {size}, Quality: {quality}, Style: {style}")
        logger.info(f"✅ [DUMMY MODE] Returning fixed dummy image URL: {dummy_image_url}")
        
        return dummy_image_url
        
    except Exception as e:
        error_msg = f"Error in dummy image generation: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise Exception(error_msg)

async def execute_function_call(
    function_call: dict, 
    lexia_handler, 
    data
) -> tuple[str, str]:
    """
    Execute a function call and return the result and any generated file URL.
    
    Args:
        function_call: The function call object from OpenAI
        lexia_handler: The Lexia handler instance for streaming updates
        data: The original chat message data
        
    Returns:
        tuple: (result_message, generated_file_url or None)
    """
    try:
        function_name = function_call['function']['name']
        logger.info(f"🔧 Processing function: {function_name}")
        
        # Stream generic function processing start to Lexia
        processing_msg = f"\n⚙️ **Processing function:** {function_name}"
        lexia_handler.stream_chunk(data, processing_msg)
        
        if function_name == "generate_image":
            return await _execute_generate_image(function_call, lexia_handler, data)
        else:
            error_msg = f"Unknown function: {function_name}"
            logger.error(error_msg)
            return f"\n\n❌ **Function Error:** {error_msg}", None
            
    except Exception as e:
        error_msg = f"Error executing function {function_call['function']['name']}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        function_error = f"\n\n❌ **Function Execution Error:** {error_msg}"
        return function_error, None

async def _execute_generate_image(
    function_call: dict, 
    lexia_handler, 
    data
) -> tuple[str, str]:
    """
    Execute the generate_image function specifically.
    
    Args:
        function_call: The function call object from OpenAI
        lexia_handler: The Lexia handler instance for streaming updates
        data: The original chat message data
        
    Returns:
        tuple: (result_message, generated_image_url)
    """
    try:
        args = json.loads(function_call["function"]["arguments"])
        logger.info(f"🎨 Executing DALL-E image generation with args: {args}")
        
        # Stream function execution start to Lexia
        execution_msg = f"\n🚀 **Executing function:** generate_image (Dummy Mode)"
        lexia_handler.stream_chunk(data, execution_msg)
        
        # Stream image generation start markdown
        lexia_handler.stream_chunk(data, "[lexia.loading.image.start]")
        
        # Wait 5 seconds to simulate image generation process
        logger.info("⏳ [DUMMY MODE] Waiting 5 seconds to simulate image generation...")
        await asyncio.sleep(5)
        
        # Generate the dummy image URL
        image_url = await generate_image_with_dalle(
            prompt=args.get("prompt"),
            variables=data.variables,
            size=args.get("size", "1024x1024"),
            quality=args.get("quality", "standard"),
            style=args.get("style", "vivid")
        )
        
        logger.info(f"✅ [DUMMY MODE] Image URL returned: {image_url}")
        
        # Stream image generation end markdown
        lexia_handler.stream_chunk(data, "[lexia.loading.image.end]")
        
        # Stream function completion to Lexia
        completion_msg = f"\n✅ **Function completed successfully:** generate_image (Dummy Mode)"
        lexia_handler.stream_chunk(data, completion_msg)
        
        # Add image generation result to response
        image_result = f"\n\n🎨 **Dummy Image Generated Successfully!**\n\n**Prompt:** {args.get('prompt')}\n**Image URL:** [lexia.image.start]{image_url}[lexia.image.end] \n\n*Demo image for testing purposes*"
        
        # Stream the image result to Lexia
        lexia_handler.stream_chunk(data, image_result)
        
        logger.info(f"✅ Image generation completed: {image_url}")
        
        return image_result, image_url
        
    except Exception as e:
        error_msg = f"Error executing generate_image function: {str(e)}"
        logger.error(error_msg, exc_info=True)
        function_error = f"\n\n❌ **Function Execution Error:** {error_msg}"
        return function_error, None

def get_available_functions() -> list:
    """
    Get the list of available functions for OpenAI.
    
    Returns:
        list: List of function schemas
    """
    return AVAILABLE_FUNCTIONS

async def process_function_calls(
    function_calls: list, 
    lexia_handler, 
    data
) -> tuple[str, str]:
    """
    Process a list of function calls and return the combined result.
    
    Args:
        function_calls: List of function call objects from OpenAI
        lexia_handler: The Lexia handler instance for streaming updates
        data: The original chat message data
        
    Returns:
        tuple: (combined_result_message, generated_file_url or None)
    """
    if not function_calls:
        logger.info("🔧 No function calls to process")
        return "", None
    
    logger.info(f"🔧 Processing {len(function_calls)} function calls...")
    logger.info(f"🔧 Function calls details: {function_calls}")
    
    combined_result = ""
    generated_file_url = None
    
    for function_call in function_calls:
        try:
            result, file_url = await execute_function_call(function_call, lexia_handler, data)
            combined_result += result
            
            if file_url and not generated_file_url:
                generated_file_url = file_url
                
        except Exception as e:
            error_msg = f"Error processing function call: {str(e)}"
            logger.error(error_msg, exc_info=True)
            combined_result += f"\n\n❌ **Function Processing Error:** {error_msg}"
    
    return combined_result, generated_file_url
