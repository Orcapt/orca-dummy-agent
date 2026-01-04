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
            "description": "Send one or more audio tracks to the user. Use tracks array for multiple tracks to avoid clearing previous content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Audio URL (for single track, deprecated - use tracks array instead)"},
                    "label": {"type": "string", "description": "Track label (for single track)"},
                    "mime_type": {"type": "string", "description": "MIME type (e.g. audio/mp3, for single track)"},
                    "tracks": {
                        "type": "array",
                        "description": "Array of audio tracks to send. Use this to send multiple tracks without clearing previous content.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "url": {"type": "string", "description": "Audio URL"},
                                "label": {"type": "string", "description": "Track label"},
                                "mime_type": {"type": "string", "description": "MIME type (e.g. audio/mp3)"}
                            },
                            "required": ["url"]
                        }
                    }
                }
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
    },
    {
        "type": "function",
        "function": {
            "name": "complete_streaming_example",
            "description": "Demonstrate a complete streaming experience with all loading states and content types. This is useful for testing and showcasing all available UI components.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "test_loading_states",
            "description": "Test individual loading states (thinking, searching, analyzing, coding, generating). Useful for debugging specific loading states.",
            "parameters": {
                "type": "object",
                "properties": {
                    "states": {
                        "type": "array",
                        "description": "List of loading states to test. Options: thinking, searching, analyzing, coding, generating",
                        "items": {"type": "string"}
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "test_image",
            "description": "Test image display with loading state. Useful for debugging image component.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Image URL (optional, defaults to demo image)"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "test_video",
            "description": "Test video display with loading state. Useful for debugging video component.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Video URL (optional, defaults to demo video)"
                    },
                    "is_youtube": {
                        "type": "boolean",
                        "description": "Whether it's a YouTube link"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "test_cards",
            "description": "Test card list display with loading state. Useful for debugging card component.",
            "parameters": {
                "type": "object",
                "properties": {
                    "count": {
                        "type": "integer",
                        "description": "Number of cards to generate (default: 3)"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "test_audio",
            "description": "Test audio display with loading state. Useful for debugging audio component.",
            "parameters": {
                "type": "object",
                "properties": {
                    "count": {
                        "type": "integer",
                        "description": "Number of audio tracks to generate (default: 2)"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "test_map",
            "description": "Test map display with loading state. Useful for debugging map component.",
            "parameters": {
                "type": "object",
                "properties": {
                    "lat": {
                        "type": "number",
                        "description": "Latitude (default: 35.6892 for Tehran)"
                    },
                    "lng": {
                        "type": "number",
                        "description": "Longitude (default: 51.3890 for Tehran)"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "test_buttons",
            "description": "Test buttons display. Useful for debugging button component.",
            "parameters": {
                "type": "object",
                "properties": {
                    "count": {
                        "type": "integer",
                        "description": "Number of buttons to generate (default: 3)"
                    }
                },
                "required": []
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
    return f"  ❌ **Function Execution Error:** {msg}", None


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

    session.stream(" 🚀 **Executing function:** generate_image (Dummy Mode)")

    async def job():
        return await generate_image_with_dalle(**args)

    try:
        image_url = await _run_with_loading(session, "generating", job())

        session.image.send(image_url)

        result = (
            "  🎨 **Dummy Image Generated Successfully!**  "
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

        return f" ✅ Video sent: {url}", None

    except Exception as e:
        return _fail("send_video", e)


async def handle_send_audio(fc: dict, session):
    args = _parse_args(fc)

    async def job():
        await asyncio.sleep(1)
        return args

    try:
        data = await _run_with_loading(session, "audio", job())

        # Support both single track (backward compatibility) and multiple tracks
        if "tracks" in data and data["tracks"]:
            # Multiple tracks: convert to format expected by send()
            tracks = []
            for track in data["tracks"]:
                track_dict = {"url": track["url"]}
                if track.get("label"):
                    track_dict["label"] = track["label"]
                if track.get("mime_type"):
                    track_dict["type"] = track["mime_type"]
                tracks.append(track_dict)
            session.audio.send(tracks)
            return f" ✅ {len(data['tracks'])} audio track(s) sent", None
        elif "url" in data:
            # Single track: use send_single for backward compatibility
            session.audio.send_single(
                data["url"],
                data.get("label"),
                data.get("mime_type"),
            )
            return " ✅ Audio sent", None
        else:
            return " ❌ No audio URL or tracks provided", None

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
        return " ✅ Location sent", None

    except Exception as e:
        return _fail("send_location", e)


async def handle_send_trace(fc: dict, session):
    args = _parse_args(fc)
    session.tracing.send(args["content"], args.get("visibility", "all"))
    return " ✅ Trace sent", None


async def handle_send_buttons(fc: dict, session):
    args = _parse_args(fc)

    session.button.begin()
    for b in args["buttons"]:
        button_type = b.get("type", "action")
        label = b.get("label", "")
        row = b.get("row")
        color = b.get("color")
        
        if button_type == "link":
            # For link buttons: use "url" or "value" as URL
            url = b.get("url") or b.get("value", "")
            if not url:
                logger.warning(f"Link button missing URL, skipping: {b}")
                continue
            session.button.add_link(label, url, row=row, color=color)
        else:
            # For action buttons: use "id" or "value" as action_id
            action_id = b.get("id") or b.get("value", "")
            if not action_id:
                logger.warning(f"Action button missing ID, skipping: {b}")
                continue
            session.button.add_action(label, action_id, row=row, color=color)
    
    session.button.end()

    return " ✅ Buttons sent", None


async def handle_send_card_list(fc: dict, session):
    args = _parse_args(fc)

    async def job():
        await asyncio.sleep(1)
        return args["cards"]

    try:
        cards = await _run_with_loading(session, "card.list", job())
        session.card.send(cards)
        return " ✅ Card list sent", None

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
    return " ✅ Usage tracked", None


# =========================
# Streaming Example Helpers (for debugging)
# =========================

async def _demo_loading_state(session, state_name: str, message: str, delay: float = 0.5):
    """Helper to demonstrate a loading state"""
    logger.info(f"🔄 [DEMO] Loading state: {state_name}")
    session.loading.start(state_name)
    try:
        await asyncio.sleep(delay)
        session.stream(f" {message} ")
    finally:
        session.loading.end(state_name)


async def _demo_image(session):
    """Demo: Image with loading state"""
    logger.info("🖼️ [DEMO] Image")
    await _demo_loading_state(session, "image", "Image loaded successfully!")
    session.image.send("https://picsum.photos/400/300")
    session.stream("\n")


async def _demo_video(session):
    """Demo: Video with loading state"""
    logger.info("🎥 [DEMO] Video")
    await _demo_loading_state(session, "video", "Video loaded!")
    session.video.send("https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4")
    session.stream("\n")


async def _demo_youtube(session):
    """Demo: YouTube video with loading state"""
    logger.info("📺 [DEMO] YouTube")
    await _demo_loading_state(session, "youtube", "YouTube video loaded!")
    session.video.youtube("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    session.stream("\n")


async def _demo_cards(session):
    """Demo: Cards with loading state"""
    logger.info("🃏 [DEMO] Cards")
    await _demo_loading_state(session, "card.list", "Cards loaded!")
    cards = [
        {
            "photo": "https://picsum.photos/300/200?random=1",
            "header": "Card 1",
            "subheader": "First result",
            "text": "Detailed information about card 1"
        },
        {
            "photo": "https://picsum.photos/300/200?random=2",
            "header": "Card 2",
            "subheader": "Second result",
            "text": "Detailed information about card 2"
        },
        {
            "photo": "https://picsum.photos/300/200?random=3",
            "header": "Card 3",
            "subheader": "Third result",
            "text": "Detailed information about card 3"
        }
    ]
    session.card.send(cards)
    session.stream("\n")


async def _demo_map(session):
    """Demo: Map with loading state"""
    logger.info("🗺️ [DEMO] Map")
    await _demo_loading_state(session, "map", "Map loaded! This is Tehran, Iran.")
    session.location.send_coordinates(35.6892, 51.3890)
    session.stream("\n")


async def _demo_buttons(session):
    """Demo: Buttons"""
    logger.info("🔘 [DEMO] Buttons")
    session.stream("## Additional Content\n\nHere are some buttons:\n\n")
    try:
        # Use begin/end to group all buttons in a single block
        session.button.begin()
        session.button.add_action("Get More Info", "1", row=1)
        session.button.add_action("Save Results", "2", row=1)
        session.button.add_link("Visit Website", "https://example.com", row=2)
        session.button.end()
        logger.debug("Buttons streamed successfully")
    except Exception as e:
        logger.error(f"Error in _demo_buttons: {e}", exc_info=True)
    session.stream("\n")


async def _demo_audio(session):
    """Demo: Audio with loading state"""
    logger.info("🎵 [DEMO] Audio")
    session.stream("## Audio Content  ")
    await _demo_loading_state(session, "audio", "")
    
    # Use send() with list of tracks (correct API)
    tracks = [
        {
            "url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3",
            "label": "Sample Audio 1",
            "type": "audio/mpeg"
        },
        {
            "url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-2.mp3",
            "label": "Sample Audio 2",
            "type": "audio/mpeg"
        }
    ]
    session.audio.send(tracks)
    session.stream("\n")


async def _demo_code(session):
    """Demo: Code example"""
    logger.info("💻 [DEMO] Code")
    code_example = """## Code Example

```javascript
const example = {
  message: "Streaming complete!",
  timestamp: new Date().toISOString(),
  status: "success"
};
console.log(example);
```

"""
    session.stream(code_example)


async def _demo_tracing(session):
    """Demo: Tracing information"""
    logger.info("🔍 [DEMO] Tracing")
    session.stream("## Tracing Information  ")
    trace_content = {
        "request_id": "req_complete_stream",
        "timestamp": "2024-12-31T10:00:00Z",
        "duration": "5000ms",
        "status": "success",
        "model": "gpt-4",
        "tokens": 500,
        "message": "Complete streaming example finished successfully"
    }
    session.tracing.send(str(trace_content), "all")
    session.stream("\n")


async def handle_complete_streaming_example(fc: dict, session):
    """
    Demonstrates a complete streaming experience with all loading states and content types.
    This function showcases:
    - All loading states (thinking, searching, analyzing, coding, generating, image, video, youtube, card, map)
    - All content types (text, images, videos, YouTube, cards, maps, buttons, audio, code, tracing)
    """
    try:
        # Stream all content types
        await _demo_loading_state(session, "thinking", "Thinking...")
        await _demo_loading_state(session, "searching", "Searching...")
        await _demo_loading_state(session, "analyzing", "Analyzing...")
        await _demo_loading_state(session, "coding", "Coding...")
        await _demo_loading_state(session, "generating", "Generating...")
        
        await _demo_image(session)
        await _demo_video(session)
        await _demo_youtube(session)
        await _demo_cards(session)
        await _demo_map(session)
        await _demo_audio(session)
        await _demo_code(session)
        await _demo_tracing(session)
        
        # Buttons at the end
        await _demo_buttons(session)
        
        # Don't stream anything after buttons
        return " ✅ Complete streaming example executed successfully!", None
    except Exception as e:
        logger.error(f"❌ [COMPLETE_STREAMING] Error: {e}", exc_info=True)
        return _fail("complete_streaming_example", e)


async def handle_test_loading_states(fc: dict, session):
    """Test individual loading states"""
    args = _parse_args(fc)
    states = args.get("states", ["thinking", "searching", "analyzing", "coding", "generating"])
    
    logger.info(f"🧪 [TEST] Testing loading states: {states}")
    session.stream(f"# Testing Loading States  Testing {len(states)} loading state(s): {', '.join(states)}  ")
    
    try:
        for state in states:
            if state in ["thinking", "searching", "analyzing", "coding", "generating"]:
                await _demo_loading_state(session, state, f"{state.capitalize()} state tested!")
            else:
                session.stream(f" ⚠️ Unknown state: {state} ")
        
        return f" ✅ Tested {len(states)} loading state(s) successfully!", None
    except Exception as e:
        return _fail("test_loading_states", e)


async def handle_test_image(fc: dict, session):
    """Test image display"""
    args = _parse_args(fc)
    url = args.get("url", "https://picsum.photos/400/300")
    
    logger.info(f"🧪 [TEST] Testing image: {url}")
    session.stream(f"# Testing Image  Image URL: {url}  ")
    
    try:
        await _demo_loading_state(session, "image", "Image loaded successfully!")
        session.image.send(url)
        session.stream("\n")
        return f" ✅ Image test completed: {url}", None
    except Exception as e:
        return _fail("test_image", e)


async def handle_test_video(fc: dict, session):
    """Test video display"""
    args = _parse_args(fc)
    url = args.get("url", "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4")
    is_youtube = args.get("is_youtube", False)
    
    logger.info(f"🧪 [TEST] Testing video: {url} (YouTube: {is_youtube})")
    session.stream(f"# Testing Video  Video URL: {url} YouTube: {is_youtube}  ")
    
    try:
        await _demo_loading_state(session, "video" if not is_youtube else "youtube", "Video loaded!")
        if is_youtube:
            session.video.youtube(url)
        else:
            session.video.send(url)
        session.stream("\n")
        return f" ✅ Video test completed: {url}", None
    except Exception as e:
        return _fail("test_video", e)


async def handle_test_cards(fc: dict, session):
    """Test card list display"""
    args = _parse_args(fc)
    count = args.get("count", 3)
    
    logger.info(f"🧪 [TEST] Testing cards: {count} cards")
    session.stream(f"# Testing Cards  Generating {count} card(s)...  ")
    
    try:
        await _demo_loading_state(session, "card.list", f"{count} cards loaded!")
        cards = []
        for i in range(1, count + 1):
            cards.append({
                "photo": f"https://picsum.photos/300/200?random={i}",
                "header": f"Card {i}",
                "subheader": f"Result {i}",
                "text": f"Detailed information about card {i}"
            })
        session.card.send(cards)
        session.stream("\n")
        return f" ✅ Cards test completed: {count} cards", None
    except Exception as e:
        return _fail("test_cards", e)


async def handle_test_audio(fc: dict, session):
    """Test audio display"""
    args = _parse_args(fc)
    count = args.get("count", 2)
    
    logger.info(f"🧪 [TEST] Testing audio: {count} tracks")
    session.stream(f"# Testing Audio  Generating {count} audio track(s)...  ")
    
    try:
        await _demo_loading_state(session, "audio", "")
        
        # Use send() with list of tracks (correct API)
        tracks = []
        for i in range(1, count + 1):
            tracks.append({
                "url": f"https://www.soundhelix.com/examples/mp3/SoundHelix-Song-{i}.mp3",
                "label": f"Sample Audio {i}",
                "type": "audio/mpeg"
            })
        session.audio.send(tracks)
        session.stream("\n")
        return f" ✅ Audio test completed: {count} tracks", None
    except Exception as e:
        return _fail("test_audio", e)


async def handle_test_map(fc: dict, session):
    """Test map display"""
    args = _parse_args(fc)
    lat = args.get("lat", 35.6892)
    lng = args.get("lng", 51.3890)
    
    logger.info(f"🧪 [TEST] Testing map: {lat}, {lng}")
    session.stream(f"# Testing Map  Coordinates: {lat}, {lng}  ")
    
    try:
        await _demo_loading_state(session, "map", f"Map loaded! Coordinates: {lat}, {lng}")
        session.location.send_coordinates(lat, lng)
        session.stream("\n")
        return f" ✅ Map test completed: {lat}, {lng}", None
    except Exception as e:
        return _fail("test_map", e)


async def handle_test_buttons(fc: dict, session):
    """Test buttons display"""
    args = _parse_args(fc)
    count = args.get("count", 3)
    
    logger.info(f"🧪 [TEST] Testing buttons: {count} buttons")
    session.stream(f"# Testing Buttons  Generating {count} button(s)...  ")
    
    try:
        session.button.begin()
        for i in range(1, count + 1):
            if i % 2 == 0:
                session.button.add_link(f"Link {i}", f"https://example.com/{i}", row=(i + 1) // 2)
            else:
                session.button.add_action(f"Action {i}", str(i), row=(i + 1) // 2)
        session.button.end()
        session.stream("\n")
        return f" ✅ Buttons test completed: {count} buttons", None
    except Exception as e:
        return _fail("test_buttons", e)


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
    "complete_streaming_example": handle_complete_streaming_example,
    "test_loading_states": handle_test_loading_states,
    "test_image": handle_test_image,
    "test_video": handle_test_video,
    "test_cards": handle_test_cards,
    "test_audio": handle_test_audio,
    "test_map": handle_test_map,
    "test_buttons": handle_test_buttons,
}


async def execute_function_call(function_call: dict, session):
    fn_name = function_call["function"]["name"]
    fn_args = function_call.get("function", {}).get("arguments", "{}")
    
    logger.info(f"🔧 [TOOL CALL] Function: {fn_name}")
    logger.info(f"📋 [TOOL CALL] Arguments: {fn_args}")
    session.stream(f" 🔧 **Tool Called:** `{fn_name}` ")
    if fn_args and fn_args != "{}":
        try:
            import json
            parsed_args = json.loads(fn_args)
            if parsed_args:
                session.stream(f"📋 **Arguments:** {json.dumps(parsed_args, indent=2)} ")
        except:
            pass

    handler = FUNCTION_HANDLERS.get(fn_name)
    if not handler:
        logger.error(f"❌ [TOOL CALL] Unknown function: {fn_name}")
        return f" ❌ Unknown function: {fn_name}", None

    try:
        result, url = await handler(function_call, session)
        logger.info(f"✅ [TOOL CALL] Function {fn_name} completed successfully")
        return result, url
    except Exception as e:
        logger.error(f"❌ [TOOL CALL] Function {fn_name} failed: {e}", exc_info=True)
        return _fail(fn_name, e)


# =========================
# Public API
# =========================

def get_available_functions() -> list:
    return AVAILABLE_FUNCTIONS


async def process_function_calls(function_calls: list, session):
    if not function_calls:
        return "", None

    logger.info(f"🔄 [TOOL CALLS] Processing {len(function_calls)} function call(s)")
    session.stream(f"Executing {len(function_calls)} tool(s)...\n")
    
    output = ""
    file_url: Optional[str] = None

    for idx, fc in enumerate(function_calls, 1):
        fn_name = fc.get("function", {}).get("name", "unknown")
        logger.info(f"🔄 [TOOL CALLS] [{idx}/{len(function_calls)}] Executing: {fn_name}")
        session.stream(f"--- Tool {idx}/{len(function_calls)}: {fn_name} ---\n")
        
        result, url = await execute_function_call(fc, session)
        output += result
        file_url = file_url or url
        
        logger.info(f"✅ [TOOL CALLS] [{idx}/{len(function_calls)}] Completed: {fn_name}")

    logger.info(f"✅ [TOOL CALLS] All {len(function_calls)} function call(s) completed")
    session.stream(f"All {len(function_calls)} tool(s) completed!\n")
    
    return output, file_url