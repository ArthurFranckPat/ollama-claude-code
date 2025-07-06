# Ollama-to-Claude Proxy Server

A proxy server that mimics Ollama's API but routes requests to Claude Code CLI, enabling seamless integration with Zed Agent Panel.

## Features

- **Ollama API Compatibility**: Implements `/api/generate`, `/api/chat`, `/api/tags`, and `/api/show` endpoints
- **Claude CLI Integration**: Uses Claude Code CLI for actual AI processing
- **Session Management**: Proper session lifecycle management with cleanup
- **Streaming Support**: Real-time streaming responses
- **TypeScript**: Full TypeScript implementation with proper types
- **Logging**: Comprehensive logging with Winston
- **Health Monitoring**: Built-in health check endpoint

## Installation

```bash
npm install
```

## Usage

### Development

```bash
npm run dev
```

### Production

```bash
npm run build
npm start
```

### Environment Variables

- `PORT`: Server port (default: 11434)
- `DEBUG`: Enable debug logging (default: false)
- `LOG_LEVEL`: Winston log level (default: info)
- `NODE_ENV`: Environment (production/development)

## API Endpoints

### Generate Completion
```bash
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-sonnet-4","prompt":"Hello!"}'
```

### Chat Completion
```bash
curl -X POST http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-sonnet-4","messages":[{"role":"user","content":"Hello!"}]}'
```

### List Models
```bash
curl http://localhost:11434/api/tags
```

### Health Check
```bash
curl http://localhost:11434/health
```

## Zed Integration

Configure Zed to use this proxy server by adding to your `settings.json`:

```json
{
  "language_models": {
    "openai": {
      "api_url": "http://localhost:11434/v1",
      "available_models": [
        {
          "name": "claude-sonnet-4",
          "display_name": "Claude Sonnet 4",
          "max_tokens": 8192
        }
      ]
    }
  }
}
```

## Architecture

- **Express Server**: Main HTTP server with TypeScript
- **Claude Session Manager**: Manages Claude CLI processes with node-pty
- **Request Translation**: Converts Ollama API format to Claude CLI commands
- **Response Streaming**: Implements real-time response streaming
- **Session Cleanup**: Automatic cleanup of inactive sessions

## Development

### Project Structure

```
src/
├── server.ts              # Main Express server
├── claude-session-manager.ts  # Claude CLI session management
├── logger.ts              # Winston logging configuration
└── types.ts               # TypeScript interfaces
```

### Testing

```bash
# Test generation endpoint
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-sonnet-4","prompt":"Write a hello world in Python"}'

# Test chat endpoint
curl -X POST http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-sonnet-4","messages":[{"role":"user","content":"What is TypeScript?"}]}'
```

## Requirements

- Node.js 18+
- Claude Code CLI installed and accessible in PATH
- TypeScript 5.7+

## License

ISC