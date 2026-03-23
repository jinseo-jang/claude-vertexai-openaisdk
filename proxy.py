"""
pip install fastapi uvicorn httpx pydantic
uvicorn main:app --host 0.0.0.0 --port 8000
"""
import os
import time
import uuid
import json
from typing import Any, Dict, List, Optional, Literal

import httpx
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field


app = FastAPI(title="LLM Router")


# -----------------------------
# Config
# -----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# Vertex Claude endpoint example:
# https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT}/locations/{LOCATION}/publishers/anthropic/models/{MODEL}:streamRawPredict
# or :rawPredict depending on your integration path.
#
# This sample assumes your gateway/proxy can obtain Google auth and call Vertex.
VERTEX_PROJECT = os.getenv("VERTEX_PROJECT", "YOUR_PROJECT_ID")
VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", "us-east5")

from google.auth import default
import google.auth.transport.requests

def get_gcp_access_token() -> str:
    """Obtains a fresh GCP access token string using application default credentials."""
    credentials, _ = default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    credentials.refresh(google.auth.transport.requests.Request())
    return credentials.token


# -----------------------------
# OpenAI-compatible request/response models
# -----------------------------
class FunctionSpec(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)


class ToolSpec(BaseModel):
    type: Literal["function"]
    function: FunctionSpec


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[Any] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    tools: Optional[List[ToolSpec]] = None
    tool_choice: Optional[Any] = None
    stream: Optional[bool] = False


# -----------------------------
# Model routing
# -----------------------------
def route_provider(model: str) -> str:
    m = model.lower()
    if m.startswith("gpt-") or m.startswith("o1") or m.startswith("o3") or m.startswith("o4"):
        return "openai"
    if "claude" in m:
        return "claude_vertex"
    raise HTTPException(status_code=400, detail=f"Unsupported model: {model}")


# -----------------------------
# OpenAI passthrough
# -----------------------------
async def call_openai_chat(req: ChatCompletionRequest) -> Dict[str, Any]:
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not configured")

    payload = req.model_dump(exclude_none=True)

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(
            f"{OPENAI_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
        )
        if r.status_code >= 400:
            raise HTTPException(status_code=r.status_code, detail=r.text)
        return r.json()


# -----------------------------
# OpenAI -> Claude(Messages API on Vertex) transform
# -----------------------------
def openai_messages_to_anthropic(
    messages: List[ChatMessage],
) -> Dict[str, Any]:
    """
    Convert OpenAI chat messages into:
    - system: separate top-level string/list
    - messages: anthropic conversation messages
    """
    system_blocks: List[Dict[str, str]] = []
    anthropic_messages: List[Dict[str, Any]] = []

    for msg in messages:
        if msg.role == "system":
            if isinstance(msg.content, str):
                system_blocks.append({"type": "text", "text": msg.content})
            elif isinstance(msg.content, list):
                for block in msg.content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        system_blocks.append({"type": "text", "text": block.get("text", "")})
        elif msg.role in ("user", "assistant"):
            content = msg.content
            if isinstance(content, str):
                anthropic_messages.append(
                    {
                        "role": msg.role,
                        "content": [{"type": "text", "text": content}],
                    }
                )
            elif isinstance(content, list):
                # pass through text/image-ish blocks if already structured
                anthropic_messages.append({"role": msg.role, "content": content})
            else:
                anthropic_messages.append(
                    {
                        "role": msg.role,
                        "content": [{"type": "text", "text": json.dumps(content, ensure_ascii=False)}],
                    }
                )
        elif msg.role == "tool":
            # Claude expects tool_result to appear in a user turn
            anthropic_messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.tool_call_id or "",
                            "content": msg.content if isinstance(msg.content, str) else json.dumps(msg.content, ensure_ascii=False),
                        }
                    ],
                }
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported message role for Claude: {msg.role}")

    return {
        "system": system_blocks if system_blocks else None,
        "messages": anthropic_messages,
    }


def openai_tools_to_anthropic(tools: Optional[List[ToolSpec]]) -> Optional[List[Dict[str, Any]]]:
    if not tools:
        return None

    converted = []
    for tool in tools:
        if tool.type != "function":
            continue
        converted.append(
            {
                "name": tool.function.name,
                "description": tool.function.description or "",
                "input_schema": tool.function.parameters or {"type": "object", "properties": {}},
            }
        )
    return converted or None


def openai_tool_choice_to_anthropic(tool_choice: Any) -> Optional[Dict[str, Any]]:
    if tool_choice is None:
        return None
    if tool_choice in ("auto", "none"):
        return {"type": tool_choice}
    if tool_choice == "required":
        # OpenAI "required" means must use one or more tools.
        # Claude supports auto/any/tool semantics in native tool use docs.
        return {"type": "any"}
    if isinstance(tool_choice, dict):
        # OpenAI style:
        # {"type":"function","function":{"name":"get_weather"}}
        if tool_choice.get("type") == "function":
            fn = tool_choice.get("function", {})
            name = fn.get("name")
            if name:
                return {"type": "tool", "name": name}
    return None


def anthropic_response_to_openai(
    anthropic_json: Dict[str, Any],
    requested_model: str,
) -> Dict[str, Any]:
    content = anthropic_json.get("content", [])
    text_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []

    for block in content:
        block_type = block.get("type")
        if block_type == "text":
            text_parts.append(block.get("text", ""))
        elif block_type == "tool_use":
            tool_calls.append(
                {
                    "id": block.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input", {}), ensure_ascii=False),
                    },
                }
            )

    finish_reason = "tool_calls" if tool_calls else "stop"
    message: Dict[str, Any] = {
        "role": "assistant",
        "content": "\n".join([p for p in text_parts if p]).strip() or None,
    }
    if tool_calls:
        message["tool_calls"] = tool_calls

    usage_src = anthropic_json.get("usage", {}) or {}
    usage = {
        "prompt_tokens": usage_src.get("input_tokens", 0),
        "completion_tokens": usage_src.get("output_tokens", 0),
        "total_tokens": usage_src.get("input_tokens", 0) + usage_src.get("output_tokens", 0),
    }

    return {
        "id": f"chatcmpl_{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": requested_model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage,
    }


# -----------------------------
# Claude on Vertex call
# -----------------------------
async def call_claude_vertex_chat(req: ChatCompletionRequest) -> Dict[str, Any]:
    if not VERTEX_PROJECT or not VERTEX_LOCATION:
        raise HTTPException(
            status_code=500,
            detail="VERTEX_PROJECT and VERTEX_LOCATION must be configured",
        )
        
    try:
        access_token = get_gcp_access_token()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to obtain GCP token: {e}")

    transformed = openai_messages_to_anthropic(req.messages)
    anthropic_tools = openai_tools_to_anthropic(req.tools)
    anthropic_tool_choice = openai_tool_choice_to_anthropic(req.tool_choice)

    # Vertex Claude: model goes in URL, anthropic_version goes in request body.
    # This is documented by Anthropic for Vertex.
    payload: Dict[str, Any] = {
        "anthropic_version": "vertex-2023-10-16",
        "messages": transformed["messages"],
    }
    if transformed["system"] is not None:
        payload["system"] = transformed["system"]
    if req.max_tokens is not None:
        payload["max_tokens"] = req.max_tokens
    else:
        payload["max_tokens"] = 1024
    if req.temperature is not None:
        payload["temperature"] = req.temperature
    if anthropic_tools is not None:
        payload["tools"] = anthropic_tools
    if anthropic_tool_choice is not None:
        payload["tool_choice"] = anthropic_tool_choice

    model = req.model
    url = (
        f"https://{VERTEX_LOCATION}-aiplatform.googleapis.com/v1/"
        f"projects/{VERTEX_PROJECT}/locations/{VERTEX_LOCATION}/"
        f"publishers/anthropic/models/{model}:rawPredict"
    )

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(url, headers=headers, json=payload)
        if r.status_code >= 400:
            raise HTTPException(status_code=r.status_code, detail=r.text)
        anthropic_json = r.json()

    return anthropic_response_to_openai(anthropic_json, requested_model=req.model)


# -----------------------------
# Public endpoint
# -----------------------------
@app.post("/v1/chat/completions")
async def chat_completions(
    req: ChatCompletionRequest,
    authorization: Optional[str] = Header(default=None),
):
    provider = route_provider(req.model)

    if req.stream:
        raise HTTPException(
            status_code=501,
            detail="Streaming example omitted in this minimal sample",
        )

    if provider == "openai":
        return await call_openai_chat(req)

    if provider == "claude_vertex":
        return await call_claude_vertex_chat(req)

    raise HTTPException(status_code=400, detail=f"Unhandled provider: {provider}")