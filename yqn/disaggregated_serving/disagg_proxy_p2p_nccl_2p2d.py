# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import os
import socket
import threading
import time
import uuid
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


def _remove_expired_instances(instances: dict[str, tuple[str, float]]) -> None:
    oldest_key = next(iter(instances), None)
    while oldest_key is not None:
        value = instances[oldest_key]
        if value[1] > time.time():
            break
        print(f"Remove expired instance [HTTP:{oldest_key}, ZMQ:{value[0]}]")
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
                print(
                    f"Add prefill instance [HTTP:{data['http_address']}, "
                    f"ZMQ:{data['zmq_address']}]"
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
                print(
                    f"Add decode instance [HTTP:{data['http_address']}, "
                    f"ZMQ:{data['zmq_address']}]"
                )
        else:
            print(f"Unexpected message from {remote_address}: {data}")


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
    url: str, data: dict[str, Any], request_id: str
) -> tuple[aiohttp.ClientResponse, aiohttp.ClientSession]:
    session = aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT)
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id,
    }
    response = await session.post(url=url, json=data, headers=headers)
    return response, session


async def fully_consume_upstream(url: str, data: dict[str, Any], request_id: str) -> None:
    response, session = await stream_upstream(url, data, request_id)
    try:
        payload = await response.read()
        if response.status >= 400:
            text = payload.decode("utf-8", errors="replace")
            raise HTTPException(
                status_code=response.status,
                detail=f"Prefill request failed: {text}",
            )
    finally:
        response.release()
        await session.close()


def _select_pair() -> tuple[str, str, str, str]:
    global count

    with prefill_cv:
        prefill_list = list(prefill_instances.items())
    with decode_cv:
        decode_list = list(decode_instances.items())

    if not prefill_list or not decode_list:
        raise HTTPException(status_code=503, detail="No prefill/decode instances ready")

    idx = count
    prefill_addr, prefill_info = prefill_list[idx % len(prefill_list)]
    decode_addr, decode_info = decode_list[idx % len(decode_list)]
    count += 1

    prefill_zmq_addr = prefill_info[0]
    decode_zmq_addr = decode_info[0]
    print(
        f"pair[{idx}] [HTTP:{prefill_addr}, ZMQ:{prefill_zmq_addr}] -> "
        f"[HTTP:{decode_addr}, ZMQ:{decode_zmq_addr}]"
    )
    return prefill_addr, prefill_zmq_addr, decode_addr, decode_zmq_addr


async def _handle_openai_request(raw_request: Request, max_tokens_field: str):
    original_request_data = await raw_request.json()
    prefill_addr, prefill_zmq_addr, decode_addr, decode_zmq_addr = _select_pair()

    prefill_request = dict(original_request_data)
    prefill_request[max_tokens_field] = 1
    if max_tokens_field == "max_completion_tokens":
        prefill_request["max_tokens"] = 1

    request_id = (
        f"___prefill_addr_{prefill_zmq_addr}___decode_addr_"
        f"{decode_zmq_addr}_{random_uuid()}"
    )

    await fully_consume_upstream(
        f"http://{prefill_addr}{raw_request.url.path}",
        prefill_request,
        request_id,
    )

    decode_response, decode_session = await stream_upstream(
        f"http://{decode_addr}{raw_request.url.path}",
        original_request_data,
        request_id,
    )

    if decode_response.status >= 400:
        try:
            payload = await decode_response.text()
        finally:
            decode_response.release()
            await decode_session.close()
        raise HTTPException(
            status_code=decode_response.status,
            detail=f"Decode request failed: {payload}",
        )

    headers = {}
    content_type = decode_response.headers.get("content-type")
    if content_type:
        headers["content-type"] = content_type

    async def body_iter():
        try:
            async for chunk in decode_response.content.iter_chunked(1024):
                yield chunk
        finally:
            decode_response.release()
            await decode_session.close()

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
