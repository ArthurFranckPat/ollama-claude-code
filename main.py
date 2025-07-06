#!/usr/bin/env python3
"""
Ollama-to-Claude Proxy Server
A FastAPI server that mimics Ollama's API but routes requests to Claude Code CLI
"""

import asyncio
import json
import logging
import os
import subprocess
import time
import uuid
from typing import List, Optional, Dict, Any, AsyncGenerator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Ollama-to-Claude Proxy",
    description="Proxy server that mimics Ollama's API but routes to Claude Code CLI",
    version="1.0.0"
)

# Add CORS middleware for Zed compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Auto-workspace detection based on common patterns
def auto_detect_workspace_from_context(prompt: str, user_agent: str) -> Optional[str]:
    """Automatically detect workspace from prompt context and patterns"""
    import re
    
    # Pattern 1: Look for file paths mentioned in the prompt
    file_patterns = [
        r'(?:file|fichier)[:\s]+([/~][^\s,\.]+)',
        r'(?:in|dans|at)\s+([/~][^\s,\.]+)',
        r'`([/~][^`]+)`',
        r'"([/~][^"]+)"',
        r'project[:\s]+([/~][^\s,\.]+)',
    ]
    
    for pattern in file_patterns:
        matches = re.findall(pattern, prompt, re.IGNORECASE)
        for match in matches:
            # Expand ~ to home directory
            expanded_path = os.path.expanduser(match)
            if os.path.isfile(expanded_path):
                # If it's a file, get its directory
                workspace = os.path.dirname(expanded_path)
            elif os.path.isdir(expanded_path):
                workspace = expanded_path
            else:
                continue
                
            # Validate it looks like a real workspace
            if any(os.path.exists(os.path.join(workspace, marker)) for marker in 
                  ['.git', 'package.json', 'Cargo.toml', '.zed', '.vscode', 'pom.xml']):
                logger.info(f"Auto-detected workspace from prompt: {workspace}")
                return workspace
    
    # Pattern 2: Common workspace keywords with smart guessing
    workspace_keywords = {
        'desktop': '/Users/arthur/Desktop',
        'hire-agentic': '/Users/arthur/Desktop/hire-agentic',
        'plugins': '/Users/arthur/Desktop/Plugins',
        'ollama-claude': '/Users/arthur/Desktop/Plugins/ollama-claude-code',
        'documents': '/Users/arthur/Documents',
    }
    
    prompt_lower = prompt.lower()
    for keyword, path in workspace_keywords.items():
        if keyword in prompt_lower and os.path.exists(path):
            logger.info(f"Auto-detected workspace from keyword '{keyword}': {path}")
            return path
    
    return None

def get_most_likely_workspace() -> str:
    """Get the most likely workspace based on recent activity"""
    
    # Check for recent git activity in common locations
    common_workspaces = [
        '/Users/arthur/Desktop/hire-agentic',
        '/Users/arthur/Desktop/Plugins/ollama-claude-code', 
        '/Users/arthur/Desktop',
        '/Users/arthur/Documents',
        '/Users/arthur/Projects',
    ]
    
    for workspace in common_workspaces:
        if os.path.exists(workspace):
            # Check if it's an active workspace (has recent activity)
            if os.path.exists(os.path.join(workspace, '.git')):
                return workspace
    
    # Fallback to Desktop
    return '/Users/arthur/Desktop'

# Store auto-detected workspaces globally
auto_workspaces = {}

# Commented out middleware temporarily for debugging
# @app.middleware("http")
# async def auto_workspace_middleware(request: Request, call_next):
#     response = await call_next(request)
#     return response

# Claude CLI path - adjust this to your installation
CLAUDE_CLI_PATH = "/Users/arthur/.claude/local/claude"

# Session storage for conversation continuity
sessions = {}  # session_id -> {"working_dir": path, "last_messages": [...], "context_summary": "..."}
import tempfile
import os
session_files = {}  # session_id -> temp_file_path for conversation storage

# Pydantic models for API
class ChatMessage(BaseModel):
    role: str  # "system", "user", "assistant"
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: Optional[bool] = False

class GenerateRequest(BaseModel):
    model: str
    prompt: str
    stream: Optional[bool] = False

class ChatResponse(BaseModel):
    model: str
    created_at: str
    message: ChatMessage
    done: bool
    total_duration: Optional[int] = 0
    load_duration: Optional[int] = 0
    prompt_eval_count: Optional[int] = 0
    prompt_eval_duration: Optional[int] = 0
    eval_count: Optional[int] = 0
    eval_duration: Optional[int] = 0

class GenerateResponse(BaseModel):
    model: str
    created_at: str
    response: str
    done: bool
    total_duration: Optional[int] = 0
    load_duration: Optional[int] = 0
    prompt_eval_count: Optional[int] = 0
    prompt_eval_duration: Optional[int] = 0
    eval_count: Optional[int] = 0
    eval_duration: Optional[int] = 0

class OpenAIChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str

class OpenAIUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class OpenAIChatResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[OpenAIChoice]
    usage: OpenAIUsage

class ModelInfo(BaseModel):
    name: str
    modified_at: str
    size: int
    digest: str
    details: Dict[str, Any]

class ModelsResponse(BaseModel):
    models: List[ModelInfo]

# Legacy context management functions removed - now using Claude CLI native session management

async def execute_claude_streaming(prompt: str, working_dir: Optional[str] = None, session_id: Optional[str] = None, messages: List[ChatMessage] = None) -> AsyncGenerator[str, None]:
    """Execute Claude CLI with streaming output using stream-json format"""
    try:
        logger.info(f"Executing Claude with streaming for prompt: {prompt[:100]}...")
        
        # Use provided working directory or current directory
        cwd = working_dir or os.getcwd()
        logger.info(f"Working directory: {cwd}")
        
        # Determine if this is a continuation or new session
        is_continuation = False
        if session_id and session_id in sessions:
            session_data = sessions[session_id]
            if "claude_session_started" in session_data:
                is_continuation = True
        
        # Build command arguments using Claude CLI native session flags
        # Use default model and rely on cwd for workspace access (more efficient)
        if is_continuation:
            # Continue existing conversation using -c flag
            cmd_args = [
                CLAUDE_CLI_PATH,
                "-c",  # Continue flag for conversation continuity
                "-p", prompt,
                "--output-format", "stream-json",
                "--verbose",
                "--dangerously-skip-permissions"
            ]
            logger.info(f"Continuing Claude session {session_id} in {cwd}")
        else:
            # Start new conversation
            cmd_args = [
                CLAUDE_CLI_PATH,
                "-p", prompt,
                "--output-format", "stream-json",
                "--verbose",
                "--dangerously-skip-permissions"
            ]
            logger.info(f"Starting new Claude session {session_id} in {cwd}")
            
            # Mark session as started
            if session_id:
                sessions[session_id]["claude_session_started"] = True
        
        # Start subprocess with streaming
        process = subprocess.Popen(
            cmd_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
            bufsize=1,  # Line buffered
            universal_newlines=True
        )
        
        logger.info(f"Started Claude process PID: {process.pid}")
        
        # Stream output line by line
        full_response = ""
        while True:
            line = process.stdout.readline()
            if not line:
                break
                
            line = line.strip()
            if line:
                try:
                    # Parse JSON line from stream-json format
                    data = json.loads(line)
                    
                    # Extract content based on Claude CLI stream-json format
                    if data.get('type') == 'assistant' and 'message' in data:
                        message = data['message']
                        if 'content' in message and isinstance(message['content'], list):
                            for content_item in message['content']:
                                if content_item.get('type') == 'text' and 'text' in content_item:
                                    content = content_item['text']
                                    full_response += content
                                    
                                    # Simulate streaming by splitting into words
                                    words = content.split(' ')
                                    for i, word in enumerate(words):
                                        if i > 0:
                                            yield ' '
                                        yield word
                                        # Small delay to simulate real-time streaming
                                        await asyncio.sleep(0.05)
                    elif data.get('type') == 'result' and 'result' in data:
                        # Final result - we've already got the content from assistant message
                        pass
                    elif 'content' in data and data['content']:
                        # Fallback for other formats
                        content = data['content']
                        full_response += content
                        yield content
                    elif 'text' in data and data['text']:
                        # Another fallback
                        content = data['text']
                        full_response += content
                        yield content
                        
                except json.JSONDecodeError:
                    # If not JSON, yield as-is (might be plain text)
                    full_response += line
                    yield line
        
        # Wait for process to complete
        return_code = process.wait()
        
        if return_code != 0:
            stderr_output = process.stderr.read() if process.stderr else ""
            logger.error(f"Claude CLI error (code {return_code}): {stderr_output}")
            yield f"\n[Error: Claude CLI returned code {return_code}]"
        
        logger.info(f"Claude streaming completed. Total response length: {len(full_response)}")
        
    except Exception as e:
        logger.error(f"Error in streaming Claude execution: {e}")
        yield f"\n[Error: {str(e)}]"

async def execute_claude(prompt: str, working_dir: Optional[str] = None, session_id: Optional[str] = None, messages: List[ChatMessage] = None) -> str:
    """Execute Claude CLI with native session management"""
    try:
        logger.info(f"Executing Claude with prompt: {prompt[:100]}...")
        
        # Use provided working directory or current directory
        cwd = working_dir or os.getcwd()
        logger.info(f"Working directory: {cwd}")
        
        # Determine if this is a continuation or new session
        is_continuation = False
        if session_id and session_id in sessions:
            session_data = sessions[session_id]
            if "claude_session_started" in session_data:
                is_continuation = True
        
        # Build command arguments using Claude CLI native session flags
        # Use default model and rely on cwd for workspace access (more efficient)
        if is_continuation:
            # Continue existing conversation using -c flag
            cmd_args = [
                CLAUDE_CLI_PATH,
                "-c",  # Continue flag for conversation continuity
                "-p", prompt,
                "--dangerously-skip-permissions"
            ]
            logger.info(f"Continuing Claude session {session_id} in {cwd}")
        else:
            # Start new conversation
            cmd_args = [
                CLAUDE_CLI_PATH,
                "-p", prompt,
                "--dangerously-skip-permissions"
            ]
            logger.info(f"Starting new Claude session {session_id} in {cwd}")
            
            # Mark session as started
            if session_id:
                sessions[session_id]["claude_session_started"] = True
        
        result = subprocess.run(
            cmd_args,
            capture_output=True,
            text=True,
            timeout=60,  # Increased timeout for better reliability
            cwd=cwd
        )
        
        logger.info(f"Claude CLI return code: {result.returncode}")
        if result.stderr:
            logger.info(f"Claude CLI stderr: {result.stderr}")
        
        if result.returncode == 0:
            response = result.stdout.strip()
            
            if not response:
                logger.warning("Claude CLI returned empty response")
                response = "Claude CLI returned an empty response."
            
            logger.info(f"Claude response: {response[:100]}...")
            return response
        else:
            error_msg = result.stderr.strip() if result.stderr else f"Process exited with code {result.returncode}"
            logger.error(f"Claude CLI error: {error_msg}")
            raise HTTPException(status_code=500, detail=f"Claude CLI error: {error_msg}")
            
    except subprocess.TimeoutExpired:
        logger.error("Claude CLI timeout")
        raise HTTPException(status_code=500, detail="Claude CLI timeout")
    except Exception as e:
        logger.error(f"Error executing Claude: {e}")
        raise HTTPException(status_code=500, detail=f"Error executing Claude: {e}")

def extract_user_message(messages: List[ChatMessage]) -> str:
    """Extract the last user message from the conversation"""
    # Find the last user message
    for message in reversed(messages):
        if message.role == "user":
            return message.content
    
    # Fallback to first message if no user message found
    if messages:
        return messages[0].content
    
    return ""

def extract_session_info(request: Request, prompt: str) -> tuple[Optional[str], Optional[str]]:
    """Extract session ID and working directory from request"""
    # Try to get session ID from headers (Zed might send this)
    session_id = request.headers.get("x-session-id") or request.headers.get("conversation-id")
    
    # Create session key for auto-detection
    user_agent = request.headers.get("user-agent", "")
    client_ip = request.client.host if request.client else ""
    request_key = f"{user_agent}_{client_ip}"
    
    # If no session ID, create one based on request characteristics
    if not session_id:
        session_id = f"session_{hash(user_agent + client_ip) % 10000}"
    
    # Check if we have an existing session
    if session_id in sessions:
        logger.info(f"Continuing existing session: {session_id}")
        working_dir = sessions[session_id]["working_dir"]
        
        # Update with auto-detected workspace if available and different
        if request_key in auto_workspaces:
            auto_workspace = auto_workspaces[request_key]
            if auto_workspace != working_dir:
                sessions[session_id]["working_dir"] = auto_workspace
                logger.info(f"Updated session workspace to auto-detected: {auto_workspace}")
                working_dir = auto_workspace
        
        return session_id, working_dir
    
    # New session - check for auto-detected workspace first
    working_dir = auto_workspaces.get(request_key)
    if working_dir:
        logger.info(f"Using auto-detected workspace for new session: {working_dir}")
    else:
        # Fallback to manual detection
        working_dir = None
    
    # 1. Check multiple possible headers that Zed/editors might send
    possible_headers = [
        "x-project-path",       # Custom header we can use
        "x-workspace",          # Another custom header
        "x-working-directory",  # Standard working dir header
        "x-cwd",                # Current working directory
        "x-project-root",       # Project root
        "workspace-root",       # Alternative workspace header
        "project-path"          # Alternative project header
    ]
    
    for header in possible_headers:
        path = request.headers.get(header)
        if path and os.path.exists(path) and os.path.isdir(path):
            working_dir = path
            logger.info(f"Found working directory from header '{header}': {working_dir}")
            break
    
    # 2. Try to extract from User-Agent if it contains path info
    if not working_dir:
        user_agent = request.headers.get("user-agent", "")
        if "zed" in user_agent.lower():
            # Look for path patterns in User-Agent
            import re
            path_match = re.search(r'path[=:]([^\s;]+)', user_agent, re.IGNORECASE)
            if path_match:
                potential_path = path_match.group(1)
                if os.path.exists(potential_path) and os.path.isdir(potential_path):
                    working_dir = potential_path
                    logger.info(f"Found working directory from User-Agent: {working_dir}")
    
    # 3. Try to extract from prompt with improved patterns
    if not working_dir:
        import re
        path_patterns = [
            r'(?:working in|in project|in directory|in folder|project at|workspace at)\s+([^\s,\.]+)',
            r'(?:in|@)\s+(/[^\s,\.]+)',
            r'(?:cd|directory|folder)\s+([^\s,\.]+)',
            r'(?:project|workspace)\s+([^\s,\.]+)',
            r'file[:\s]+([^\s]+/[^\s]+)',  # file: /path/to/file
            r'`([^`]+/[^`]+)`',            # `path/to/directory`
        ]
        
        for pattern in path_patterns:
            matches = re.findall(pattern, prompt, re.IGNORECASE)
            for match in matches:
                # Extract directory from file path if it's a file
                potential_dir = match
                if os.path.isfile(potential_dir):
                    potential_dir = os.path.dirname(potential_dir)
                
                if os.path.exists(potential_dir) and os.path.isdir(potential_dir):
                    working_dir = potential_dir
                    logger.info(f"Found working directory from prompt: {working_dir}")
                    break
            if working_dir:
                break
    
    # 4. Try to detect workspace from common Zed/editor patterns
    if not working_dir:
        # Look for common workspace indicators in the prompt
        workspace_indicators = [
            "workspace", "project", "repo", "repository", "codebase", 
            "directory", "folder", "working on", "dans le projet"
        ]
        
        for indicator in workspace_indicators:
            if indicator in prompt.lower():
                # Ask Claude to help detect workspace in the prompt itself
                workspace_hint = f"Working in a {indicator}"
                logger.info(f"Found workspace indicator: {indicator}")
                break
    
    # 5. Final fallback: use smart workspace detection
    if not working_dir:
        working_dir = get_most_likely_workspace()
        logger.info(f"Using smart fallback workspace: {working_dir}")
    
    # Store session info
    sessions[session_id] = {
        "working_dir": working_dir,
        "context_summary": "",
        "created_at": time.time()
    }
    
    logger.info(f"Created new session: {session_id} with working_dir: {working_dir}")
    return session_id, working_dir

@app.get("/")
async def root():
    """Root endpoint with server information"""
    return {
        "message": "Ollama-to-Claude Proxy Server",
        "version": "1.0.0",
        "features": [
            "Streaming support with word-by-word delivery",
            "Native Claude CLI session management",
            "Automatic workspace detection via cwd",
            "Multi-header workspace support",\n            "Efficient resource usage with default model"
        ],
        "endpoints": {
            "generate": "/api/generate",
            "chat": "/api/chat (supports streaming)",
            "openai_chat": "/v1/chat/completions",
            "tags": "/api/tags",
            "show": "/api/show",
            "ps": "/api/ps",
            "health": "/health",
            "set_workspace": "/api/set-working-directory",
            "get_workspace": "/api/get-working-directory",
            "sessions": "/api/sessions"
        },
        "workspace_headers": [
            "x-project-path",
            "x-workspace", 
            "x-working-directory",
            "x-cwd",
            "x-project-root",
            "workspace-root",
            "project-path"
        ],
        "usage": {
            "set_workspace": "POST /api/set-working-directory with {\"working_dir\": \"/path/to/project\"}",
            "streaming": "Add \"stream\": true to chat requests for real-time response"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "models": ["claude-sonnet-4", "claude-opus-4", "claude-3-5-sonnet-20241022"]
    }

@app.post("/api/generate")
async def generate(request: GenerateRequest) -> GenerateResponse:
    """Ollama-compatible generate endpoint"""
    logger.info(f"Generate request for model: {request.model}")
    
    response_text = await execute_claude(request.prompt)
    
    return GenerateResponse(
        model=request.model,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        response=response_text,
        done=True,
        prompt_eval_count=len(request.prompt.split()),
        eval_count=len(response_text.split())
    )

@app.post("/api/chat")
async def chat(request: ChatRequest, http_request: Request):
    """Ollama-compatible chat endpoint with streaming support"""
    logger.info(f"Chat request for model: {request.model}, streaming: {request.stream}")
    
    # Extract user message and session info
    user_prompt = extract_user_message(request.messages)
    session_id, working_dir = extract_session_info(http_request, user_prompt)
    
    if request.stream:
        # Streaming response
        async def generate_chat_stream():
            async for chunk in execute_claude_streaming(user_prompt, working_dir, session_id, request.messages):
                # Format as Ollama chat streaming response
                response = {
                    "model": request.model,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                    "message": {
                        "role": "assistant",
                        "content": chunk
                    },
                    "done": False
                }
                yield f"{json.dumps(response)}\n"
            
            # Send final chunk
            final_response = {
                "model": request.model,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                "message": {
                    "role": "assistant",
                    "content": ""
                },
                "done": True
            }
            yield f"{json.dumps(final_response)}\n"
        
        return StreamingResponse(
            generate_chat_stream(),
            media_type="application/x-ndjson"
        )
    else:
        # Non-streaming response
        response_text = await execute_claude(user_prompt, working_dir, session_id, request.messages)
        
        return ChatResponse(
            model=request.model,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            message=ChatMessage(role="assistant", content=response_text),
            done=True,
            prompt_eval_count=len(user_prompt.split()),
            eval_count=len(response_text.split())
        )

@app.post("/v1/chat/completions")
async def openai_chat(request: ChatRequest, http_request: Request) -> OpenAIChatResponse:
    """OpenAI-compatible chat completions endpoint"""
    logger.info(f"OpenAI chat request for model: {request.model}")
    
    # Log headers for debugging
    auth_header = http_request.headers.get("authorization", "")
    logger.info(f"Authorization header: {auth_header[:20]}...")
    
    # Extract user message and session info
    user_prompt = extract_user_message(request.messages)
    session_id, working_dir = extract_session_info(http_request, user_prompt)
    
    response_text = await execute_claude(user_prompt, working_dir, session_id, request.messages)
    
    return OpenAIChatResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
        object="chat.completion",
        created=int(time.time()),
        model=request.model,
        choices=[
            OpenAIChoice(
                index=0,
                message=ChatMessage(role="assistant", content=response_text),
                finish_reason="stop"
            )
        ],
        usage=OpenAIUsage(
            prompt_tokens=len(user_prompt.split()),
            completion_tokens=len(response_text.split()),
            total_tokens=len(user_prompt.split()) + len(response_text.split())
        )
    )

@app.get("/api/tags")
async def list_models() -> ModelsResponse:
    """List available models"""
    models = [
        ModelInfo(
            name="claude-sonnet-4",
            modified_at=time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            size=7_000_000_000,
            digest="sha256:claude-sonnet-4-proxy",
            details={
                "format": "claude",
                "family": "claude",
                "families": ["claude"],
                "parameter_size": "7B",
                "quantization_level": "Q4_0"
            }
        ),
        ModelInfo(
            name="claude-opus-4",
            modified_at=time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            size=175_000_000_000,
            digest="sha256:claude-opus-4-proxy",
            details={
                "format": "claude",
                "family": "claude",
                "families": ["claude"],
                "parameter_size": "175B",
                "quantization_level": "Q4_0"
            }
        )
    ]
    
    return ModelsResponse(models=models)

@app.get("/api/ps")
async def running_models():
    """Show running models"""
    return {
        "models": [
            {
                "name": "claude-sonnet-4",
                "model": "claude-sonnet-4",
                "size": 7_000_000_000,
                "digest": "sha256:claude-sonnet-4-proxy",
                "details": {
                    "format": "claude",
                    "family": "claude",
                    "families": ["claude"],
                    "parameter_size": "7B",
                    "quantization_level": "Q4_0"
                },
                "expires_at": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ", 
                                         time.gmtime(time.time() + 300)),  # 5 minutes
                "size_vram": 0
            }
        ]
    }

@app.post("/api/show")
async def show_model(request: dict):
    """Show model information"""
    model_name = request.get("name", "unknown")
    
    return {
        "modelfile": f"# Modelfile for {model_name}\nFROM claude-code-cli\nPARAMETER temperature 0.7\nPARAMETER top_p 0.9\n",
        "parameters": "temperature 0.7\ntop_p 0.9",
        "template": "{{ if .System }}System: {{ .System }}\n{{ end }}{{ if .Prompt }}Human: {{ .Prompt }}\n{{ end }}Assistant: ",
        "details": {
            "format": "claude",
            "family": "claude",
            "families": ["claude"],
            "parameter_size": "175B" if "opus" in model_name else "7B",
            "quantization_level": "Q4_0"
        }
    }

@app.post("/api/pull")
async def pull_model(request: dict):
    """Simulate model pulling"""
    model_name = request.get("name", "unknown")
    stream = request.get("stream", False)
    
    logger.info(f"Pull request for model: {model_name}")
    
    if stream:
        # For streaming, we'd need to implement Server-Sent Events
        # For now, return success
        return {"status": "success"}
    else:
        return {"status": "success"}

@app.post("/api/set-working-directory")
async def set_working_directory(request: dict, http_request: Request):
    """Set working directory for a session"""
    working_dir = request.get("working_dir") or request.get("workspace") or request.get("project_path")
    
    if not working_dir:
        raise HTTPException(status_code=400, detail="Missing working_dir, workspace, or project_path")
    
    # Expand user path and resolve relative paths
    working_dir = os.path.expanduser(working_dir)
    working_dir = os.path.abspath(working_dir)
    
    if not os.path.exists(working_dir):
        raise HTTPException(status_code=400, detail=f"Directory does not exist: {working_dir}")
    
    if not os.path.isdir(working_dir):
        raise HTTPException(status_code=400, detail=f"Path is not a directory: {working_dir}")
    
    # Get session ID from request or headers
    session_id = (request.get("session_id") or 
                 http_request.headers.get("x-session-id") or 
                 http_request.headers.get("conversation-id"))
    
    if not session_id:
        # Create new session ID
        user_agent = http_request.headers.get("user-agent", "")
        client_ip = http_request.client.host if http_request.client else ""
        session_id = f"session_{hash(user_agent + client_ip) % 10000}"
    
    # Update or create session
    if session_id in sessions:
        sessions[session_id]["working_dir"] = working_dir
        logger.info(f"Updated working directory for existing session {session_id}: {working_dir}")
    else:
        sessions[session_id] = {
            "working_dir": working_dir,
            "context_summary": "",
            "created_at": time.time()
        }
        logger.info(f"Created new session {session_id} with working directory: {working_dir}")
    
    return {
        "status": "success", 
        "session_id": session_id, 
        "working_dir": working_dir,
        "message": f"Working directory set to {working_dir}"
    }

@app.get("/api/get-working-directory")
async def get_working_directory(http_request: Request):
    """Get current working directory for a session"""
    session_id = (http_request.headers.get("x-session-id") or 
                 http_request.headers.get("conversation-id"))
    
    if session_id and session_id in sessions:
        working_dir = sessions[session_id]["working_dir"]
        return {
            "status": "success",
            "session_id": session_id,
            "working_dir": working_dir
        }
    else:
        return {
            "status": "not_found",
            "message": "No active session found",
            "working_dir": os.getcwd()
        }

@app.get("/api/sessions")
async def list_sessions():
    """List active sessions"""
    return {"sessions": {sid: info for sid, info in sessions.items()}}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=11435,
        reload=True,
        log_level="info"
    )