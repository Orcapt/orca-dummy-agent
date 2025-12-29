"""
Function Handler Module for Simple AI Agent
==========================================

This module handles function calling capabilities for the simple AI agent.
It contains the function definitions and execution logic for image generation.

Key Features:
- DALL-E 3 image generation function
- Function schema definitions
- Function execution and error handling
- Streaming progress updates via Orca Session

Author: Orca Team
License: MIT
"""
from typing import Tuple, Optional, Callable, Dict
import asyncio
import logging
import json
from orca import Variables

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
    },
    {
        "type": "function",
        "function": {
            "name": "send_video",
            "description": "Send a video or YouTube link to the user",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Video URL"},
                    "is_youtube": {"type": "boolean", "description": "Whether it's a YouTube link"}
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_audio",
            "description": "Send one or more audio tracks to the user",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Audio URL"},
                    "label": {"type": "string", "description": "Track label"},
                    "mime_type": {"type": "string", "description": "MIME type (e.g. audio/mp3)"}
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_location",
            "description": "Send a map location to the user",
            "parameters": {
                "type": "object",
                "properties": {
                    "lat": {"type": "number", "description": "Latitude"},
                    "lng": {"type": "number", "description": "Longitude"},
                    "label": {"type": "string", "description": "Location description"}
                },
                "required": ["lat", "lng"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_trace",
            "description": "Send a debug/internal trace message (not visible to end-users unless explicitly allowed)",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Trace content"},
                    "visibility": {"type": "string", "enum": ["all", "admin"], "default": "all"}
                },
                "required": ["content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_buttons",
            "description": "Send a block of interactive buttons to the user",
            "parameters": {
                "type": "object",
                "properties": {
                    "buttons": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string", "enum": ["link", "action"]},
                                "label": {"type": "string"},
                                "value": {"type": "string", "description": "URL for link, ID for action"},
                                "color": {"type": "string", "description": "primary, destructive, etc."},
                                "row": {"type": "integer"}
                            },
                            "required": ["type", "label", "value"]
                        }
                    }
                },
                "required": ["buttons"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_card_list",
            "description": "Send a list of visual cards to the user",
            "parameters": {
                "type": "object",
                "properties": {
                    "cards": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "photo": {"type": "string", "description": "Image URL"},
                                "header": {"type": "string", "description": "Title"},
                                "subheader": {"type": "string", "description": "Description"},
                                "text": {"type": "string", "description": "Additional text"}
                            }
                        }
                    }
                },
                "required": ["cards"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "track_usage",
            "description": "Track token usage for the current request",
            "parameters": {
                "type": "object",
                "properties": {
                    "tokens": {"type": "integer", "description": "Number of tokens"},
                    "token_type": {"type": "string", "enum": ["prompt", "completion", "total"]},
                    "cost": {"type": "string", "description": "Optional cost (e.g. '$0.002')"},
                    "label": {"type": "string", "description": "Optional label"}
                },
                "required": ["tokens", "token_type"]
            }
        }
    }
]


# =========================
# Helpers
# =========================

def _parse_args(function_call: dict) -> dict:
    return json.loads(function_call["function"]["arguments"])


async def _run_with_loading(session, key: str, coro, min_delay: float = 1.5):
    """
    Run a coroutine with a loading indicator and ensure it's visible for at least min_delay.
    """
    session.loading.start(key)
    start_time = asyncio.get_event_loop().time()
    try:
        result = await coro
        # Calculate how much longer we need to wait to hit the min_delay
        elapsed = asyncio.get_event_loop().time() - start_time
        if elapsed < min_delay:
            await asyncio.sleep(min_delay - elapsed)
        return result
    finally:
        session.loading.end(key)


def _fail(fn_name: str, error: Exception):
    msg = f"Error executing function `{fn_name}`: {error}"
    logger.error(msg, exc_info=True)
    return f"\n\n❌ **Function Execution Error:** {msg}", None


# =========================
# Dummy Image Generator
# =========================

async def generate_image_with_dalle(
    prompt: str,
    size: str = "1024x1024",
    quality: str = "standard",
    style: str = "vivid",
) -> str:
    logger.info("🎨 [DUMMY] Generating image")
    await asyncio.sleep(1)
    return (
        "https://fsn1.your-objectstorage.com/"
        "lexia-production/demo/images/generated_1758945096.png"
    )

# =========================
# Handlers
# =========================

async def handle_generate_image(fc: dict, session):
    args = _parse_args(fc)

    session.stream("\n🚀 **Executing function:** generate_image (Dummy Mode)")

    async def job():
        return await generate_image_with_dalle(**args)

    try:
        image_url = await _run_with_loading(session, "generating", job())

        session.image.send(image_url)

        result = (
            "\n\n🎨 **Dummy Image Generated Successfully!**\n\n"
            f"**Prompt:** {args.get('prompt')}"
        )
        session.stream(result)

        return result, image_url

    except Exception as e:
        return _fail("generate_image", e)


async def handle_send_video(fc: dict, session):
    args = _parse_args(fc)

    async def job():
        await asyncio.sleep(1)
        return args["url"], args.get("is_youtube", False)

    try:
        url, is_youtube = await _run_with_loading(session, "video", job())

        if is_youtube:
            session.video.youtube(url)
        else:
            session.video.send(url)

        return f"\n✅ Video sent: {url}", None

    except Exception as e:
        return _fail("send_video", e)


async def handle_send_audio(fc: dict, session):
    args = _parse_args(fc)

    async def job():
        await asyncio.sleep(1)
        return args

    try:
        data = await _run_with_loading(session, "audio", job())

        session.audio.send_single(
            data["url"],
            data.get("label"),
            data.get("mime_type"),
        )

        return "\n✅ Audio sent", None

    except Exception as e:
        return _fail("send_audio", e)


async def handle_send_location(fc: dict, session):
    args = _parse_args(fc)

    async def job():
        await asyncio.sleep(1)
        return args["lat"], args["lng"]

    try:
        lat, lng = await _run_with_loading(session, "map", job())
        session.location.send_coordinates(lat, lng)
        return "\n✅ Location sent", None

    except Exception as e:
        return _fail("send_location", e)


async def handle_send_trace(fc: dict, session):
    args = _parse_args(fc)
    session.tracing.send(args["content"], args.get("visibility", "all"))
    return "\n✅ Trace sent", None


async def handle_send_buttons(fc: dict, session):
    args = _parse_args(fc)

    session.button.begin()
    for b in args["buttons"]:
        action = (
            session.button.add_link
            if b["type"] == "link"
            else session.button.add_action
        )
        action(
            b["label"],
            b["value"],
            row=b.get("row"),
            color=b.get("color"),
        )
    session.button.end()

    return "\n✅ Buttons sent", None


async def handle_send_card_list(fc: dict, session):
    args = _parse_args(fc)

    async def job():
        await asyncio.sleep(1)
        return args["cards"]

    try:
        cards = await _run_with_loading(session, "card.list", job())
        session.card.send(cards)
        return "\n✅ Card list sent", None

    except Exception as e:
        return _fail("send_card_list", e)


async def handle_track_usage(fc: dict, session):
    args = _parse_args(fc)
    session.usage.track(
        args["tokens"],
        args["token_type"],
        cost=args.get("cost"),
        label=args.get("label"),
    )
    return "\n✅ Usage tracked", None


# =========================
# Dispatcher
# =========================

FUNCTION_HANDLERS: Dict[str, Callable] = {
    "generate_image": handle_generate_image,
    "send_video": handle_send_video,
    "send_audio": handle_send_audio,
    "send_location": handle_send_location,
    "send_trace": handle_send_trace,
    "send_buttons": handle_send_buttons,
    "send_card_list": handle_send_card_list,
    "track_usage": handle_track_usage,
}


async def execute_function_call(function_call: dict, session):
    fn_name = function_call["function"]["name"]
    logger.info(f"🔧 Processing function: {fn_name}")
    session.stream(f"\n⚙️ **Processing function:** {fn_name}\n")

    handler = FUNCTION_HANDLERS.get(fn_name)
    if not handler:
        return f"\n❌ Unknown function: {fn_name}", None

    return await handler(function_call, session)


# =========================
# Public API
# =========================

def get_available_functions() -> list:
    return AVAILABLE_FUNCTIONS


async def process_function_calls(function_calls: list, session):
    if not function_calls:
        return "", None

    output = ""
    file_url: Optional[str] = None

    for fc in function_calls:
        result, url = await execute_function_call(fc, session)
        output += result
        file_url = file_url or url

    return output, file_url