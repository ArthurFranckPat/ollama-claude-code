#!/usr/bin/env python3
"""
Ollama-to-Claude Proxy Server
A FastAPI server that mimics Ollama's API but routes requests to Claude Code CLI
"""

import asyncio
import logging
import os
import subprocess
import time
import uuid
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
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
        if is_continuation:
            # Continue existing conversation using -c flag
            cmd_args = [
                CLAUDE_CLI_PATH,
                "-c",  # Continue flag for conversation continuity
                "-p", prompt,
                "--dangerously-skip-permissions"
            ]
            logger.info(f"Continuing Claude session {session_id}")
        else:
            # Start new conversation
            cmd_args = [
                CLAUDE_CLI_PATH,
                "-p", prompt,
                "--dangerously-skip-permissions"
            ]
            logger.info(f"Starting new Claude session {session_id}")
            
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
    
    # If no session ID, create one based on request characteristics
    if not session_id:
        # Use a combination of headers to create a stable session ID
        user_agent = request.headers.get("user-agent", "")
        client_ip = request.client.host if request.client else ""
        session_id = f"session_{hash(user_agent + client_ip) % 10000}"
    
    # Check if we have an existing session
    if session_id in sessions:
        logger.info(f"Continuing existing session: {session_id}")
        working_dir = sessions[session_id]["working_dir"]
        return session_id, working_dir
    
    # New session - try to extract working directory
    working_dir = None
    
    # Check headers for project path
    project_path = request.headers.get("x-project-path")
    if project_path and os.path.exists(project_path):
        working_dir = project_path
    
    # Check for workspace headers
    if not working_dir:
        workspace = request.headers.get("x-workspace")
        if workspace and os.path.exists(workspace):
            working_dir = workspace
    
    # Try to extract from prompt
    if not working_dir:
        import re
        path_patterns = [
            r'(?:in|@)\s+(/[^\s]+)',
            r'(?:cd|directory|folder)\s+(/[^\s]+)',
            r'(?:project|workspace)\s+(/[^\s]+)'
        ]
        
        for pattern in path_patterns:
            matches = re.findall(pattern, prompt, re.IGNORECASE)
            for match in matches:
                if os.path.exists(match) and os.path.isdir(match):
                    working_dir = match
                    break
            if working_dir:
                break
    
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
        "endpoints": {
            "generate": "/api/generate",
            "chat": "/api/chat",
            "openai_chat": "/v1/chat/completions",
            "tags": "/api/tags",
            "show": "/api/show",
            "ps": "/api/ps",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "models": ["claude-sonnet-4", "claude-opus-4"]
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
async def chat(request: ChatRequest, http_request: Request) -> ChatResponse:
    """Ollama-compatible chat endpoint"""
    logger.info(f"Chat request for model: {request.model}")
    
    # Extract user message and session info
    user_prompt = extract_user_message(request.messages)
    session_id, working_dir = extract_session_info(http_request, user_prompt)
    
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
    working_dir = request.get("working_dir")
    if not working_dir or not os.path.exists(working_dir):
        raise HTTPException(status_code=400, detail="Invalid working directory")
    
    # Get or create session
    session_id, _ = extract_session_info(http_request, "")
    sessions[session_id]["working_dir"] = working_dir
    
    logger.info(f"Set working directory for session {session_id}: {working_dir}")
    return {"status": "success", "session_id": session_id, "working_dir": working_dir}

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