"""
Universal AWS Lambda Handler
============================

A unified entry point for AWS Lambda that handles:
1. HTTP (via FastAPI/Mangum)
2. SQS (Background Processing)
3. Cron (Scheduled Events)

Powered by the Orca create_hybrid_handler factory.
"""

import logging
from orca import create_hybrid_handler
from main import process_message

# Create the universal handler
# Note: This will automatically route events based on their source (HTTP, SQS, or Cron)
handler = create_hybrid_handler(
    process_message_func=process_message,
    app_title="Orca Dummy Lambda Agent"
)

if __name__ == "__main__":
    # For local quick testing (not production)
    print("🚀 Orca Hybrid Handler exported as 'handler'")
