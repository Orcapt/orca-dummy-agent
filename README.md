# Simple AI Agent

A minimal AI agent that demonstrates basic chat completion functionality using the Lexia platform with streaming support and dummy image generation for testing purposes.

## Features

- **Simple Chat Completion**: Uses OpenAI's chat completion API
- **Dummy Image Generation**: Returns a fixed demo image URL for testing
- **Real-time Streaming**: Responses are streamed in real-time via Lexia
- **Conversation Memory**: Maintains conversation history per thread
- **Function Calling**: Supports OpenAI function calling for dummy image generation
- **Clean Implementation**: Minimal, easy-to-understand code

## Setup

### Option 1: Docker (Recommended)

1. **Build and Run with Docker Compose**:
   ```bash
   # Build and start the agent
   docker-compose up --build
   
   # Run in background
   docker-compose up -d --build
   
   # View logs
   docker-compose logs -f
   ```

2. **Production Setup with Nginx**:
   ```bash
   # Run with nginx reverse proxy
   docker-compose --profile production up -d --build
   ```

3. **Stop the Agent**:
   ```bash
   docker-compose down
   ```

### Option 2: Local Development

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set OpenAI API Key**:
   - Configure your OpenAI API key in the Lexia platform
   - Go to Admin Mode → Agents → Edit Agent → Variables section
   - Add `OPENAI_API_KEY` with your OpenAI API key

3. **Run the Agent**:
   ```bash
   python main.py
   ```

## Usage

### Docker Deployment
The agent will start on `http://localhost:8000` (or `http://localhost:80` with nginx) with the following endpoints:

### Local Development
The agent will start on `http://localhost:8000` with the following endpoints:

- **POST** `/api/v1/send_message` - Main chat endpoint
- **GET** `/api/v1/health` - Health check
- **GET** `/api/v1/` - Root information
- **GET** `/api/v1/docs` - Interactive API documentation

## API Examples

### Basic Chat
Send a POST request to `/api/v1/send_message`:

```json
{
  "message": "Hello, how are you?",
  "thread_id": "user_123",
  "model": "gpt-3.5-turbo",
  "variables": [
    {
      "name": "OPENAI_API_KEY",
      "value": "your-openai-api-key"
    }
  ]
}
```

### Image Generation
Ask the agent to generate an image:

```json
{
  "message": "Generate an image of a beautiful sunset over mountains",
  "thread_id": "user_123",
  "model": "gpt-4",
  "variables": [
    {
      "name": "OPENAI_API_KEY",
      "value": "your-openai-api-key"
    }
  ]
}
```

The agent will automatically use the `generate_image` function to return a fixed demo image URL for testing purposes.

## How It Works

1. **Message Processing**: Receives chat messages via the Lexia platform
2. **OpenAI Integration**: Sends messages to OpenAI's chat completion API
3. **Streaming**: Streams responses back to the user in real-time
4. **Memory**: Maintains conversation history for context
5. **Error Handling**: Provides clear error messages for common issues

## Customization

This is a minimal implementation perfect for:
- Learning how Lexia streaming works
- Building a foundation for more complex agents
- Understanding the basic chat completion flow

You can extend this agent by adding:
- Function calling capabilities
- File processing (PDFs, images)
- Custom system prompts
- Database integration for persistent memory
- Additional AI model integrations

## Docker

### Container Features
- **Multi-stage build**: Optimized for production
- **Non-root user**: Security best practices
- **Health checks**: Automatic container health monitoring
- **Volume mounting**: Persistent logs
- **Nginx reverse proxy**: Production-ready setup

### Docker Commands
```bash
# Build the image
docker build -t lexia-dummy-agent .

# Run the container
docker run -p 8000:8000 lexia-dummy-agent

# Run with docker-compose
docker-compose up --build

# Run in production mode with nginx
docker-compose --profile production up -d --build
```

### Environment Variables
- `PYTHONPATH=/app` - Python path configuration
- `PYTHONUNBUFFERED=1` - Unbuffered Python output

## License

MIT
