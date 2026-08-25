"""
n8n AI Assistant to vLLM Translation Proxy
=========================================
Bridges n8n's built-in AI Assistant to vLLM
Handles both /v1/chat/completions and /v1/responses endpoints.
"""

import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("n8n-vllm-proxy")

RAW_VLLM_URL = os.getenv("VLLM_BACKEND_URL", "http://localhost:8000").rstrip("/")
VLLM_BASE_URL = RAW_VLLM_URL if RAW_VLLM_URL.endswith("/v1") else f"{RAW_VLLM_URL}/v1"
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "")

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "Qwen/Qwen2.5-32B-Instruct-AWQ")
OVERRIDE_MODEL = os.getenv("OVERRIDE_MODEL", "").strip()

http_client: Optional[httpx.AsyncClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    timeout = httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=30.0)
    limits = httpx.Limits(max_keepalive_connections=100, max_connections=200)
    http_client = httpx.AsyncClient(timeout=timeout, limits=limits)
    logger.info(f"Proxy initialized. Upstream: {VLLM_BASE_URL}")
    yield
    await http_client.aclose()
    logger.info("Proxy HTTP client shut down.")


app = FastAPI(
    title="n8n-to-vLLM Translation Proxy",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def resolve_model_name(requested_model: Optional[str]) -> str:
    if OVERRIDE_MODEL:
        return OVERRIDE_MODEL
    if not requested_model:
        return DEFAULT_MODEL
    clean_model = requested_model
    if clean_model.startswith("openai/"):
        clean_model = clean_model[len("openai/"):]
    return clean_model or DEFAULT_MODEL


# ==============================================================================
# 1. Standard OpenAI Chat Completions Endpoint
# ==============================================================================
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Handles standard Chat Completions requests from n8n / Vercel AI SDK."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    # Resolve target model
    body["model"] = resolve_model_name(body.get("model"))

    upstream_headers = {"Content-Type": "application/json"}
    if VLLM_API_KEY:
        upstream_headers["Authorization"] = f"Bearer {VLLM_API_KEY}"
    elif "authorization" in request.headers:
        upstream_headers["Authorization"] = request.headers["authorization"]

    is_stream = bool(body.get("stream", False))

    if is_stream:
        async def stream_raw():
            try:
                async with http_client.stream(
                    "POST",
                    f"{VLLM_BASE_URL}/chat/completions",
                    json=body,
                    headers=upstream_headers,
                    timeout=300.0,
                ) as upstream_res:
                    async for chunk in upstream_res.aiter_bytes():
                        yield chunk
            except Exception as e:
                logger.error(f"Error streaming chat completions: {e}", exc_info=True)

        return StreamingResponse(
            stream_raw(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream; charset=utf-8",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        try:
            upstream_res = await http_client.post(
                f"{VLLM_BASE_URL}/chat/completions",
                json=body,
                headers=upstream_headers,
                timeout=300.0,
            )
            return Response(
                content=upstream_res.content,
                status_code=upstream_res.status_code,
                media_type="application/json",
            )
        except Exception as e:
            logger.error(f"Error forwarding chat completions: {e}")
            raise HTTPException(status_code=502, detail=f"Failed to reach vLLM: {e}")


# ==============================================================================
# 2. OpenAI Responses API Translation Endpoint
# ==============================================================================
def map_input_to_messages(
    input_data: Union[str, List[Any], None],
    instructions: Optional[Union[str, List[Any]]] = None,
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []

    if instructions:
        if isinstance(instructions, str) and instructions.strip():
            messages.append({"role": "system", "content": instructions.strip()})
        elif isinstance(instructions, list):
            inst_text = " ".join(
                item if isinstance(item, str) else item.get("text", "")
                for item in instructions
            ).strip()
            if inst_text:
                messages.append({"role": "system", "content": inst_text})

    if not input_data:
        return messages

    if isinstance(input_data, str):
        messages.append({"role": "user", "content": input_data})
        return messages

    if isinstance(input_data, list):
        for item in input_data:
            if isinstance(item, str):
                messages.append({"role": "user", "content": item})
                continue
            if not isinstance(item, dict):
                continue

            item_type = item.get("type")
            item_role = item.get("role")

            if item_role or item_type == "message":
                role = item_role or item.get("role", "user")
                raw_content = item.get("content", "")
                content = ""
                if isinstance(raw_content, str):
                    content = raw_content
                elif isinstance(raw_content, list):
                    text_parts = []
                    for part in raw_content:
                        if isinstance(part, str):
                            text_parts.append(part)
                        elif isinstance(part, dict) and "text" in part:
                            text_parts.append(part["text"])
                    content = "\n".join(text_parts)
                else:
                    content = str(raw_content)

                msg_obj: Dict[str, Any] = {"role": role, "content": content}
                if "tool_calls" in item and isinstance(item["tool_calls"], list):
                    msg_obj["tool_calls"] = item["tool_calls"]
                messages.append(msg_obj)

            elif item_type == "function_call":
                call_id = item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex[:12]}"
                fn_name = item.get("name", "")
                fn_args = item.get("arguments", "{}")
                if not isinstance(fn_args, str):
                    fn_args = json.dumps(fn_args)

                tool_call = {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": fn_name, "arguments": fn_args},
                }

                if messages and messages[-1].get("role") == "assistant" and "tool_calls" in messages[-1]:
                    messages[-1]["tool_calls"].append(tool_call)
                else:
                    messages.append({"role": "assistant", "content": None, "tool_calls": [tool_call]})

            elif item_type == "function_call_output" or item_role == "tool":
                call_id = item.get("call_id") or item.get("tool_call_id") or item.get("id", "")
                raw_output = item.get("output") if "output" in item else item.get("content", "")
                output_str = raw_output if isinstance(raw_output, str) else json.dumps(raw_output)
                messages.append({"role": "tool", "tool_call_id": call_id, "content": output_str})

    return messages


def translate_tools(raw_tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
    if not raw_tools or not isinstance(raw_tools, list):
        return None
    translated = []
    for tool in raw_tools:
        if not isinstance(tool, dict):
            continue
        tool_type = tool.get("type", "function")
        if tool_type == "function":
            if "function" in tool and isinstance(tool["function"], dict):
                translated.append(tool)
            elif "name" in tool:
                translated.append({
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
                    },
                })
    return translated if translated else None


def translate_tool_choice(tool_choice: Any) -> Any:
    if not tool_choice:
        return None
    if isinstance(tool_choice, str):
        return tool_choice
    if isinstance(tool_choice, dict):
        if "function" in tool_choice:
            return tool_choice
        if tool_choice.get("type") == "function" and "name" in tool_choice:
            return {"type": "function", "function": {"name": tool_choice["name"]}}
    return tool_choice


async def stream_responses_translator(
    vllm_payload: Dict[str, Any],
    upstream_headers: Dict[str, str],
    client_model_name: str,
) -> AsyncGenerator[str, None]:
    response_id = f"resp_{uuid.uuid4().hex[:24]}"
    msg_item_id = f"msg_{uuid.uuid4().hex[:24]}"
    created_at = int(time.time())
    seq = 0

    def sse(event_type: str, data: Dict[str, Any]) -> str:
        return f"event: {event_type}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"

    yield sse(
        "response.created",
        {
            "type": "response.created",
            "sequence_number": seq,
            "response": {
                "id": response_id,
                "object": "response",
                "created_at": created_at,
                "model": client_model_name,
                "status": "in_progress",
            },
        },
    )
    seq += 1

    text_started = False
    accumulated_text = ""
    output_items: List[Dict[str, Any]] = []
    current_output_index = 0
    tool_calls_map: Dict[int, Dict[str, Any]] = {}
    usage_data = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    try:
        async with http_client.stream(
            "POST",
            f"{VLLM_BASE_URL}/chat/completions",
            json=vllm_payload,
            headers=upstream_headers,
            timeout=300.0,
        ) as upstream_response:
            if upstream_response.status_code != 200:
                err_body = await upstream_response.aread()
                logger.error(f"vLLM upstream error ({upstream_response.status_code}): {err_body.decode('utf-8', errors='ignore')}")
                yield sse("error", {"type": "error", "code": f"vllm_error_{upstream_response.status_code}", "message": err_body.decode("utf-8", errors="ignore")})
                return

            async for line in upstream_response.aiter_lines():
                if not line:
                    continue

                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    if "usage" in chunk and chunk["usage"]:
                        u = chunk["usage"]
                        usage_data = {
                            "input_tokens": u.get("prompt_tokens", 0),
                            "output_tokens": u.get("completion_tokens", 0),
                            "total_tokens": u.get("total_tokens", 0),
                        }

                    choices = chunk.get("choices", [])
                    if not choices:
                        continue

                    delta = choices[0].get("delta", {})
                    text_delta = delta.get("content")
                    if text_delta:
                        if not text_started:
                            yield sse("response.output_item.added", {"type": "response.output_item.added", "sequence_number": seq, "output_index": current_output_index, "item": {"id": msg_item_id, "type": "message", "role": "assistant", "content": []}})
                            seq += 1
                            yield sse("response.content_part.added", {"type": "response.content_part.added", "sequence_number": seq, "output_index": current_output_index, "content_index": 0, "part": {"type": "output_text", "text": ""}})
                            seq += 1
                            text_started = True

                        yield sse("response.output_text.delta", {"type": "response.output_text.delta", "sequence_number": seq, "output_index": current_output_index, "content_index": 0, "delta": text_delta})
                        seq += 1
                        accumulated_text += text_delta

                    tool_deltas = delta.get("tool_calls")
                    if tool_deltas:
                        for td in tool_deltas:
                            tc_idx = td.get("index", 0)
                            if tc_idx not in tool_calls_map:
                                if text_started:
                                    yield sse("response.output_text.done", {"type": "response.output_text.done", "sequence_number": seq, "output_index": current_output_index, "content_index": 0, "text": accumulated_text})
                                    seq += 1
                                    yield sse("response.content_part.done", {"type": "response.content_part.done", "sequence_number": seq, "output_index": current_output_index, "content_index": 0, "part": {"type": "output_text", "text": accumulated_text}})
                                    seq += 1
                                    txt_item = {"id": msg_item_id, "type": "message", "status": "completed", "role": "assistant", "content": [{"type": "output_text", "text": accumulated_text}]}
                                    yield sse("response.output_item.done", {"type": "response.output_item.done", "sequence_number": seq, "output_index": current_output_index, "item": txt_item})
                                    seq += 1
                                    output_items.append(txt_item)
                                    current_output_index += 1
                                    text_started = False

                                call_id = td.get("id") or f"call_{uuid.uuid4().hex[:24]}"
                                fn_name = td.get("function", {}).get("name", "")
                                tool_calls_map[tc_idx] = {
                                    "id": call_id,
                                    "call_id": call_id,
                                    "name": fn_name,
                                    "arguments": "",
                                    "output_index": current_output_index,
                                }
                                current_output_index += 1

                                yield sse(
                                    "response.output_item.added",
                                    {
                                        "type": "response.output_item.added",
                                        "sequence_number": seq,
                                        "output_index": tool_calls_map[tc_idx]["output_index"],
                                        "item": {"id": call_id, "call_id": call_id, "type": "function_call", "name": fn_name, "arguments": "", "status": "in_progress"},
                                    },
                                )
                                seq += 1

                            args_delta = td.get("function", {}).get("arguments", "")
                            if args_delta:
                                tool_calls_map[tc_idx]["arguments"] += args_delta
                                yield sse(
                                    "response.function_call_arguments.delta",
                                    {
                                        "type": "response.function_call_arguments.delta",
                                        "sequence_number": seq,
                                        "output_index": tool_calls_map[tc_idx]["output_index"],
                                        "call_id": tool_calls_map[tc_idx]["call_id"],
                                        "delta": args_delta,
                                    },
                                )
                                seq += 1

    except Exception as e:
        logger.error(f"Error during stream forwarding: {e}", exc_info=True)
        yield sse("error", {"type": "error", "code": "stream_forward_exception", "message": str(e)})
        return

    if text_started:
        yield sse("response.output_text.done", {"type": "response.output_text.done", "sequence_number": seq, "output_index": current_output_index, "content_index": 0, "text": accumulated_text})
        seq += 1
        yield sse("response.content_part.done", {"type": "response.content_part.done", "sequence_number": seq, "output_index": current_output_index, "content_index": 0, "part": {"type": "output_text", "text": accumulated_text}})
        seq += 1
        txt_item = {"id": msg_item_id, "type": "message", "status": "completed", "role": "assistant", "content": [{"type": "output_text", "text": accumulated_text}]}
        yield sse("response.output_item.done", {"type": "response.output_item.done", "sequence_number": seq, "output_index": current_output_index, "item": txt_item})
        seq += 1
        output_items.append(txt_item)

    for tc in tool_calls_map.values():
        yield sse("response.function_call_arguments.done", {"type": "response.function_call_arguments.done", "sequence_number": seq, "output_index": tc["output_index"], "call_id": tc["call_id"], "arguments": tc["arguments"]})
        seq += 1
        t_item = {"id": tc["id"], "call_id": tc["call_id"], "type": "function_call", "name": tc["name"], "arguments": tc["arguments"], "status": "completed"}
        yield sse("response.output_item.done", {"type": "response.output_item.done", "sequence_number": seq, "output_index": tc["output_index"], "item": t_item})
        seq += 1
        output_items.append(t_item)

    yield sse(
        "response.completed",
        {
            "type": "response.completed",
            "sequence_number": seq,
            "response": {
                "id": response_id,
                "object": "response",
                "created_at": created_at,
                "model": client_model_name,
                "status": "completed",
                "output": output_items,
                "usage": usage_data,
            },
        },
    )
    yield "data: [DONE]\n\n"


@app.post("/v1/responses")
async def create_response(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    client_model = body.get("model", DEFAULT_MODEL)
    target_model = resolve_model_name(client_model)

    messages = map_input_to_messages(
        input_data=body.get("input"),
        instructions=body.get("instructions"),
    )

    vllm_payload: Dict[str, Any] = {
        "model": target_model,
        "messages": messages,
    }

    tools = translate_tools(body.get("tools"))
    if tools:
        vllm_payload["tools"] = tools

    tool_choice = translate_tool_choice(body.get("tool_choice"))
    if tool_choice:
        vllm_payload["tool_choice"] = tool_choice

    if "max_output_tokens" in body:
        vllm_payload["max_tokens"] = body["max_output_tokens"]
    elif "max_tokens" in body:
        vllm_payload["max_tokens"] = body["max_tokens"]

    for param in ["temperature", "top_p", "frequency_penalty", "presence_penalty", "stop", "seed"]:
        if param in body and body[param] is not None:
            vllm_payload[param] = body[param]

    upstream_headers = {"Content-Type": "application/json"}
    if VLLM_API_KEY:
        upstream_headers["Authorization"] = f"Bearer {VLLM_API_KEY}"
    elif "authorization" in request.headers:
        upstream_headers["Authorization"] = request.headers["authorization"]

    is_stream = bool(body.get("stream", False))

    if is_stream:
        vllm_payload["stream"] = True
        vllm_payload["stream_options"] = {"include_usage": True}

        return StreamingResponse(
            stream_responses_translator(
                vllm_payload=vllm_payload,
                upstream_headers=upstream_headers,
                client_model_name=client_model,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream; charset=utf-8",
                "X-Accel-Buffering": "no",
            },
        )

    vllm_payload["stream"] = False
    try:
        upstream_res = await http_client.post(
            f"{VLLM_BASE_URL}/chat/completions",
            json=vllm_payload,
            headers=upstream_headers,
            timeout=300.0,
        )
    except Exception as exc:
        logger.error(f"Failed to connect to vLLM: {exc}")
        raise HTTPException(status_code=502, detail=f"Failed to connect to upstream vLLM: {exc}")

    if upstream_res.status_code != 200:
        return Response(
            content=upstream_res.content,
            status_code=upstream_res.status_code,
            media_type="application/json",
        )

    res_json = upstream_res.json()
    choices = res_json.get("choices", [])
    output_items: List[Dict[str, Any]] = []

    if choices:
        msg = choices[0].get("message", {})
        if msg.get("content"):
            output_items.append({
                "id": f"msg_{uuid.uuid4().hex[:24]}",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": msg["content"]}],
            })

        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                call_id = tc.get("id") or f"call_{uuid.uuid4().hex[:24]}"
                fn = tc.get("function", {})
                output_items.append({
                    "id": call_id,
                    "call_id": call_id,
                    "type": "function_call",
                    "status": "completed",
                    "name": fn.get("name", ""),
                    "arguments": fn.get("arguments", "{}"),
                })

    usage = res_json.get("usage", {})
    response_payload = {
        "id": f"resp_{uuid.uuid4().hex[:24]}",
        "object": "response",
        "created_at": res_json.get("created", int(time.time())),
        "model": client_model,
        "status": "completed",
        "output": output_items,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        },
    }
    return JSONResponse(content=response_payload)


@app.get("/v1/models")
async def list_models(request: Request):
    upstream_headers = {}
    if VLLM_API_KEY:
        upstream_headers["Authorization"] = f"Bearer {VLLM_API_KEY}"
    elif "authorization" in request.headers:
        upstream_headers["Authorization"] = request.headers["authorization"]

    try:
        resp = await http_client.get(f"{VLLM_BASE_URL}/models", headers=upstream_headers, timeout=10.0)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        logger.warning(f"Upstream /v1/models failed ({e}), returning fallback model list.")

    model_name = OVERRIDE_MODEL or DEFAULT_MODEL
    return {
        "object": "list",
        "data": [
            {
                "id": model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "vllm",
            }
        ],
    }


@app.get("/v1/models/{model_id:path}")
async def get_model(model_id: str):
    return {"id": model_id, "object": "model", "created": int(time.time()), "owned_by": "vllm"}


@app.get("/health")
@app.get("/")
async def health_check():
    return {
        "status": "healthy",
        "service": "n8n-vllm-proxy",
        "target_vllm_endpoint": VLLM_BASE_URL,
        "target_model": OVERRIDE_MODEL or DEFAULT_MODEL,
    }


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run("proxy:app", host=host, port=port, log_level=LOG_LEVEL.lower(), reload=False)
