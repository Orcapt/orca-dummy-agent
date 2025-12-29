# Orca Dummy Agent

A minimal AI agent demonstrating real-time streaming and dummy image generation using the **Orca Platform**.

## Features

- 🚀 **Orca SDK Integration**: Built with the modern `orcapt-sdk`.
- 💬 **Streaming Chat**: Real-time response streaming with loading indicators.
- 🎨 **Image Generation**: Automated dummy image simulation via OpenAI functions.
- 🏗️ **Minimalist Architecture**: Clean separation between logic and delivery.
- 📦 **Docker Ready**: Production-grade containerization.

## Setup

### Option 1: Docker (Recommended)

1. **Build and Run**:
   ```bash
   docker-compose up -d --build
   ```

2. **View Logs**:
   ```bash
   docker-compose logs -f
   ```

### Option 3: AWS Lambda Deployment

The agent is ready for Serverless deployment using the **Orca Hybrid Handler**.

1. **Build Lambda Image**:
   ```bash
   docker build -t orca-dummy-lambda -f Dockerfile.lambda .
   ```

2. **Push to ECR & Deploy**:
   - Push the image to AWS ECR.
   - Create a Lambda function using the "Container Image" option.
   - Set the handler to `lambda_handler.handler` (already default in Dockerfile).

3. **Features**:
   - **Hybrid Routing**: Auto-handles HTTP (FastAPI), SQS, and Cron events.
   - **Cold Start Optimized**: Uses minimalist factory patterns.

## Testing the Agent

Since the agent is now running (or you can start it with `python main.py`), you can test it using the provided test client:

1. **Install Httpx**:
   ```bash
   pip install httpx
   ```

2. **Run Test Client**:
   ```bash
   python test_client.py "Show me a cat video, a map of Tehran, and some buttons"
   ```

3. **Check Console Output**:
   If you have `ORCA_DEV_MODE=true` set, you will see the full Orca stream (including loading markers, video URLs, and button payloads) directly in the terminal where the agent is running.

## API Endpoints

- **POST** `/api/v1/send_message` - Core chat & logic endpoint.
- **GET** `/api/v1/health` - System health check.
- **GET** `/api/v1/docs` - Interactive documentation.

## How it Works

1. **Entry Point**: The agent uses `create_agent_app` to bootstrap a FastAPI server.
2. **Session Management**: Each message starts an `OrcaHandler` session.
3. **Streaming**: Content is streamed back instantly using `session.stream()`.
4. **Tools**: Uses OpenAI tool calling to trigger `generate_image` in `function_handler.py`.
5. **UI**: Sends loading markers and images via the Orca session API.

## License

MIT
