# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import os
import socket
import threading
import time
import uuid
from ipaddress import ip_address
from typing import Any

import aiohttp
import msgpack
import zmq
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

SERVICE_DISCOVERY_PORT = 30001
PROXY_HTTP_PORT = 10001
DEFAULT_PING_SECONDS = 5
AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

prefill_instances: dict[str, tuple[str, float]] = {}
decode_instances: dict[str, tuple[str, float]] = {}
prefill_cv = threading.Condition()
decode_cv = threading.Condition()
count = 0
app = FastAPI()


def random_uuid() -> str:
    return uuid.uuid4().hex


def _log(event: str, **fields: Any) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    items = [
        f"ts={timestamp}",
        f"pid={os.getpid()}",
        f"tid={threading.get_ident()}",
        f"event={event}",
    ]
    for key, value in fields.items():
        items.append(f"{key}={value!r}")
    print(" ".join(items), flush=True)


def _compact_id(value: str | None, limit: int = 80) -> str:
    if not value:
        return "none"
    sanitized = value.replace(" ", "_")
    if len(sanitized) <= limit:
        return sanitized
    return f"{sanitized[:limit]}..."


def _request_shape(data: dict[str, Any]) -> dict[str, Any]:
    prompt = data.get("prompt")
    messages = data.get("messages")
    return {
        "keys": sorted(data.keys()),
        "model": data.get("model"),
        "stream": data.get("stream"),
        "has_stream_options": "stream_options" in data,
        "max_tokens": data.get("max_tokens"),
        "max_completion_tokens": data.get("max_completion_tokens"),
        "prompt_chars": len(prompt) if isinstance(prompt, str) else None,
        "message_count": len(messages) if isinstance(messages, list) else None,
    }


def _attach_shared_request_id(
    data: dict[str, Any], request_id: str
) -> dict[str, Any]:
    updated = dict(data)
    kv_transfer_params = dict(updated.get("kv_transfer_params") or {})
    kv_transfer_params["shared_request_id"] = request_id
    updated["kv_transfer_params"] = kv_transfer_params
    return updated


def _split_host_port(address: str) -> tuple[str, int]:
    host, sep, port_text = address.rpartition(":")
    if not sep:
        return address, -1
    try:
        return host, int(port_text)
    except ValueError:
        return host, -1


def _normalize_host(host: str) -> tuple[int, Any]:
    try:
        return (0, ip_address(host))
    except ValueError:
        return (1, host)


def _instance_sort_key(item: tuple[str, tuple[str, float]]) -> tuple[Any, ...]:
    http_addr, (zmq_addr, _) = item
    zmq_host, zmq_port = _split_host_port(zmq_addr)
    http_host, http_port = _split_host_port(http_addr)
    return (
        _normalize_host(zmq_host),
        zmq_port,
        _normalize_host(http_host),
        http_port,
        zmq_addr,
        http_addr,
    )


def _remove_expired_instances(instances: dict[str, tuple[str, float]]) -> None:
    oldest_key = next(iter(instances), None)
    while oldest_key is not None:
        value = instances[oldest_key]
        if value[1] > time.time():
            break
        _log(
            "instance_expired",
            http_address=oldest_key,
            zmq_address=value[0],
            expires_at=value[1],
            now=time.time(),
        )
        instances.pop(oldest_key, None)
        oldest_key = next(iter(instances), None)


def _listen_for_register(poller, router_socket):
    while True:
        socks = dict(poller.poll())
        if router_socket not in socks:
            continue

        remote_address, message = router_socket.recv_multipart()
        data = msgpack.loads(message)

        if data["type"] == "P":
            global prefill_instances
            global prefill_cv
            with prefill_cv:
                node = prefill_instances.get(data["http_address"])
                prefill_instances[data["http_address"]] = (
                    data["zmq_address"],
                    time.time() + DEFAULT_PING_SECONDS,
                )
                _remove_expired_instances(prefill_instances)
            if node is None:
                _log(
                    "instance_registered",
                    node_type="prefill",
                    remote_address=remote_address.decode("utf-8", errors="replace"),
                    http_address=data["http_address"],
                    zmq_address=data["zmq_address"],
                    ttl_seconds=DEFAULT_PING_SECONDS,
                )
        elif data["type"] == "D":
            global decode_instances
            global decode_cv
            with decode_cv:
                node = decode_instances.get(data["http_address"])
                decode_instances[data["http_address"]] = (
                    data["zmq_address"],
                    time.time() + DEFAULT_PING_SECONDS,
                )
                _remove_expired_instances(decode_instances)
            if node is None:
                _log(
                    "instance_registered",
                    node_type="decode",
                    remote_address=remote_address.decode("utf-8", errors="replace"),
                    http_address=data["http_address"],
                    zmq_address=data["zmq_address"],
                    ttl_seconds=DEFAULT_PING_SECONDS,
                )
        else:
            _log(
                "instance_register_unexpected_message",
                remote_address=remote_address.decode("utf-8", errors="replace"),
                payload=data,
            )


def start_service_discovery(hostname: str, port: int):
    if not hostname:
        hostname = socket.gethostname()
    if port == 0:
        raise ValueError("Port cannot be 0")

    context = zmq.Context()
    router_socket = context.socket(zmq.ROUTER)
    router_socket.bind(f"tcp://{hostname}:{port}")

    poller = zmq.Poller()
    poller.register(router_socket, zmq.POLLIN)

    listener_thread = threading.Thread(
        target=_listen_for_register, args=(poller, router_socket), daemon=True
    )
    listener_thread.start()
    return listener_thread


async def stream_upstream(
    url: str,
    data: dict[str, Any],
    request_id: str,
    stage: str,
) -> tuple[aiohttp.ClientResponse, aiohttp.ClientSession]:
    session = aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT)
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id,
    }
    _log(
        "upstream_request_start",
        stage=stage,
        url=url,
        request_id=request_id,
        payload=_request_shape(data),
    )
    response = await session.post(url=url, json=data, headers=headers)
    _log(
        "upstream_response_headers",
        stage=stage,
        url=url,
        request_id=request_id,
        status=response.status,
        content_type=response.headers.get("content-type"),
        transfer_encoding=response.headers.get("transfer-encoding"),
        content_length=response.headers.get("content-length"),
    )
    return response, session


async def fully_consume_upstream(
    url: str,
    data: dict[str, Any],
    request_id: str,
    stage: str,
) -> None:
    _log(
        "upstream_consume_begin",
        stage=stage,
        url=url,
        request_id=request_id,
    )
    response, session = await stream_upstream(url, data, request_id, stage)
    try:
        payload = await response.read()
        if response.status >= 400:
            text = payload.decode("utf-8", errors="replace")
            raise HTTPException(
                status_code=response.status,
                detail=f"{stage} request failed: {text}",
            )
        _log(
            "upstream_consume_complete",
            stage=stage,
            url=url,
            request_id=request_id,
            status=response.status,
            payload_bytes=len(payload),
        )
    finally:
        response.release()
        await session.close()
        _log(
            "upstream_consume_session_closed",
            stage=stage,
            url=url,
            request_id=request_id,
        )


def _select_pair() -> tuple[str, str, str, str]:
    global count

    with prefill_cv:
        prefill_list = sorted(prefill_instances.items(),
                              key=_instance_sort_key)
    with decode_cv:
        decode_list = sorted(decode_instances.items(), key=_instance_sort_key)

    if not prefill_list or not decode_list:
        raise HTTPException(status_code=503, detail="No prefill/decode instances ready")

    idx = count
    prefill_addr, prefill_info = prefill_list[idx % len(prefill_list)]
    decode_addr, decode_info = decode_list[idx % len(decode_list)]
    count += 1

    prefill_zmq_addr = prefill_info[0]
    decode_zmq_addr = decode_info[0]
    _log(
        "pair_selected",
        pair_index=idx,
        prefill_http=prefill_addr,
        prefill_zmq=prefill_zmq_addr,
        decode_http=decode_addr,
        decode_zmq=decode_zmq_addr,
        prefill_pool_size=len(prefill_list),
        decode_pool_size=len(decode_list),
    )
    return prefill_addr, prefill_zmq_addr, decode_addr, decode_zmq_addr


async def _handle_openai_request(raw_request: Request, max_tokens_field: str):
    raw_request_data = await raw_request.json()
    client_host = raw_request.client.host if raw_request.client else "unknown"
    client_port = raw_request.client.port if raw_request.client else "unknown"
    incoming_request_id = raw_request.headers.get("x-request-id")
    incoming_user_agent = raw_request.headers.get("user-agent")
    prefill_addr, prefill_zmq_addr, decode_addr, decode_zmq_addr = _select_pair()

    incoming_request_id_suffix = _compact_id(incoming_request_id, limit=120)
    request_id = (
        f"client_req_{incoming_request_id_suffix}___prefill_addr_"
        f"{prefill_zmq_addr}___decode_addr_{decode_zmq_addr}_{random_uuid()}"
    )

    original_request_data = _attach_shared_request_id(raw_request_data, request_id)
    prefill_request = dict(original_request_data)
    prefill_request[max_tokens_field] = 1
    prefill_request["stream"] = False
    if max_tokens_field == "max_completion_tokens":
        prefill_request["max_tokens"] = 1
    prefill_request.pop("stream_options", None)
    prefill_request.pop("stream_include_usage", None)
    prefill_request.pop("stream_continuous_usage_stats", None)
    _log(
        "incoming_request",
        path=raw_request.url.path,
        method=raw_request.method,
        client_host=client_host,
        client_port=client_port,
        user_agent=incoming_user_agent,
        incoming_request_id=incoming_request_id,
        proxy_request_id=request_id,
        payload=_request_shape(original_request_data),
    )
    _log(
        "request_routed",
        proxy_request_id=request_id,
        prefill_url=f"http://{prefill_addr}{raw_request.url.path}",
        decode_url=f"http://{decode_addr}{raw_request.url.path}",
        prefill_payload=_request_shape(prefill_request),
        decode_payload=_request_shape(original_request_data),
    )

    await fully_consume_upstream(
        f"http://{prefill_addr}{raw_request.url.path}",
        prefill_request,
        request_id,
        stage="prefill",
    )

    _log(
        "prefill_stage_finished_switch_to_decode",
        proxy_request_id=request_id,
        decode_url=f"http://{decode_addr}{raw_request.url.path}",
    )

    decode_response, decode_session = await stream_upstream(
        f"http://{decode_addr}{raw_request.url.path}",
        original_request_data,
        request_id,
        stage="decode",
    )

    _log(
        "decode_stage_connected",
        proxy_request_id=request_id,
        status=decode_response.status,
        content_type=decode_response.headers.get("content-type"),
    )

    if decode_response.status >= 400:
        try:
            payload = await decode_response.text()
        finally:
            decode_response.release()
            await decode_session.close()
            _log(
                "decode_stage_error_session_closed",
                proxy_request_id=request_id,
                status=decode_response.status,
            )
        raise HTTPException(
            status_code=decode_response.status,
            detail=f"Decode request failed: {payload}",
        )

    headers = {}
    content_type = decode_response.headers.get("content-type")
    if content_type:
        headers["content-type"] = content_type

    async def body_iter():
        chunk_count = 0
        total_bytes = 0
        try:
            async for chunk in decode_response.content.iter_chunked(1024):
                chunk_count += 1
                total_bytes += len(chunk)
                if chunk_count == 1:
                    _log(
                        "decode_stream_first_chunk",
                        proxy_request_id=request_id,
                        chunk_bytes=len(chunk),
                    )
                yield chunk
        finally:
            decode_response.release()
            await decode_session.close()
            _log(
                "decode_stream_complete",
                proxy_request_id=request_id,
                chunk_count=chunk_count,
                total_bytes=total_bytes,
            )

    return StreamingResponse(
        body_iter(),
        status_code=decode_response.status,
        headers=headers,
        media_type=content_type,
    )


@app.get("/health")
async def health():
    return Response(status_code=200)


@app.get("/status")
async def status():
    with prefill_cv:
        prefill_nodes = list(prefill_instances.keys())
    with decode_cv:
        decode_nodes = list(decode_instances.keys())
    return {
        "prefill_node_count": len(prefill_nodes),
        "decode_node_count": len(decode_nodes),
        "prefill_nodes": prefill_nodes,
        "decode_nodes": decode_nodes,
    }


@app.get("/v1/models")
async def show_models():
    model = os.environ.get("DISAGG_MODEL", "unknown")
    return JSONResponse(
        content={
            "object": "list",
            "data": [
                {
                    "id": model,
                    "object": "model",
                    "owned_by": "vllm",
                }
            ],
        }
    )


@app.post("/v1/completions")
async def create_completion(raw_request: Request):
    return await _handle_openai_request(raw_request, "max_tokens")


@app.post("/v1/chat/completions")
async def create_chat_completion(raw_request: Request):
    return await _handle_openai_request(raw_request, "max_completion_tokens")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=PROXY_HTTP_PORT)
    parser.add_argument("--discovery-port", type=int, default=SERVICE_DISCOVERY_PORT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    listener_thread = start_service_discovery(args.host, args.discovery_port)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)
    listener_thread.join()
