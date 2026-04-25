# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import itertools
import logging
import os
import sys
import uuid
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

PROXY_HTTP_PORT = 10001

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s",
            "%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
logger.propagate = False


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


def _build_prefill_request(req_data: dict[str, Any]) -> dict[str, Any]:
    prefill_request = req_data.copy()
    prefill_request["kv_transfer_params"] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
    }
    prefill_request["stream"] = False
    prefill_request["max_tokens"] = 1
    if "max_completion_tokens" in prefill_request:
        prefill_request["max_completion_tokens"] = 1
    prefill_request.pop("stream_options", None)
    prefill_request.pop("stream_include_usage", None)
    prefill_request.pop("stream_continuous_usage_stats", None)
    prefill_request.pop("min_tokens", None)
    prefill_request.pop("min_completion_tokens", None)
    return prefill_request


def _make_headers(request_id: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.prefill_clients = []
    app.state.decode_clients = []

    for idx, (host, port) in enumerate(global_args.prefiller_instances):
        app.state.prefill_clients.append(
            {
                "client": httpx.AsyncClient(
                    timeout=None,
                    base_url=f"http://{host}:{port}/v1",
                    limits=httpx.Limits(
                        max_connections=None,
                        max_keepalive_connections=None,
                    ),
                ),
                "host": host,
                "port": port,
                "id": idx,
            }
        )

    for idx, (host, port) in enumerate(global_args.decoder_instances):
        app.state.decode_clients.append(
            {
                "client": httpx.AsyncClient(
                    timeout=None,
                    base_url=f"http://{host}:{port}/v1",
                    limits=httpx.Limits(
                        max_connections=None,
                        max_keepalive_connections=None,
                    ),
                ),
                "host": host,
                "port": port,
                "id": idx,
            }
        )

    app.state.prefill_iterator = itertools.cycle(range(len(app.state.prefill_clients)))
    app.state.decode_iterator = itertools.cycle(range(len(app.state.decode_clients)))

    logger.info(
        "Initialized %d prefiller clients and %d decoder clients",
        len(app.state.prefill_clients),
        len(app.state.decode_clients),
    )

    yield

    for client_info in app.state.prefill_clients:
        await client_info["client"].aclose()
    for client_info in app.state.decode_clients:
        await client_info["client"].aclose()


app = FastAPI(lifespan=lifespan)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=PROXY_HTTP_PORT)
    parser.add_argument(
        "--prefiller-hosts",
        "--prefiller-host",
        type=str,
        nargs="+",
        default=["127.0.0.1", "127.0.0.1"],
    )
    parser.add_argument(
        "--prefiller-ports",
        "--prefiller-port",
        type=int,
        nargs="+",
        default=[8100, 8110],
    )
    parser.add_argument(
        "--decoder-hosts",
        "--decoder-host",
        type=str,
        nargs="+",
        default=["127.0.0.1", "127.0.0.1"],
    )
    parser.add_argument(
        "--decoder-ports",
        "--decoder-port",
        type=int,
        nargs="+",
        default=[8200, 8210],
    )
    parser.add_argument(
        "--discovery-port",
        type=int,
        default=None,
        help="Deprecated compatibility flag. Ignored for NixlConnector routing.",
    )

    args = parser.parse_args()

    if len(args.prefiller_hosts) != len(args.prefiller_ports):
        raise ValueError(
            "Number of prefiller hosts must match number of prefiller ports"
        )
    if len(args.decoder_hosts) != len(args.decoder_ports):
        raise ValueError("Number of decoder hosts must match number of decoder ports")

    args.prefiller_instances = list(zip(args.prefiller_hosts, args.prefiller_ports))
    args.decoder_instances = list(zip(args.decoder_hosts, args.decoder_ports))
    return args


def _get_next_client(app: FastAPI, service_type: str) -> dict[str, Any]:
    if service_type == "prefill":
        client_idx = next(app.state.prefill_iterator)
        return app.state.prefill_clients[client_idx]
    if service_type == "decode":
        client_idx = next(app.state.decode_iterator)
        return app.state.decode_clients[client_idx]
    raise ValueError(f"Unknown service type: {service_type}")


async def _send_prefill_request(
    client_info: dict[str, Any],
    endpoint: str,
    req_data: dict[str, Any],
    request_id: str,
) -> dict[str, Any]:
    response = await client_info["client"].post(
        endpoint,
        json=req_data,
        headers=_make_headers(request_id),
    )
    response.raise_for_status()
    payload = response.json()
    await response.aclose()

    kv_transfer_params = payload.get("kv_transfer_params")
    if not isinstance(kv_transfer_params, dict) or not kv_transfer_params:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Prefill response for {request_id} missing kv_transfer_params. "
                "This proxy expects NixlConnector remote decode metadata."
            ),
        )
    return payload


async def _stream_decode_response(
    client_info: dict[str, Any],
    endpoint: str,
    req_data: dict[str, Any],
    request_id: str,
):
    async with client_info["client"].stream(
        "POST",
        endpoint,
        json=req_data,
        headers=_make_headers(request_id),
    ) as response:
        response.raise_for_status()
        async for chunk in response.aiter_bytes():
            yield chunk


async def _handle_openai_request(endpoint: str, request: Request):
    req_data = await request.json()
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())

    prefill_client_info = _get_next_client(request.app, "prefill")
    prefill_request = _build_prefill_request(req_data)
    logger.info(
        "Routing request_id=%s endpoint=%s to prefill=%s:%s payload=%s",
        request_id,
        endpoint,
        prefill_client_info["host"],
        prefill_client_info["port"],
        _request_shape(prefill_request),
    )
    prefill_payload = await _send_prefill_request(
        prefill_client_info,
        endpoint,
        prefill_request,
        request_id,
    )

    decode_client_info = _get_next_client(request.app, "decode")
    decode_request = req_data.copy()
    decode_request["kv_transfer_params"] = prefill_payload["kv_transfer_params"]

    logger.info(
        "Switching request_id=%s to decode=%s:%s",
        request_id,
        decode_client_info["host"],
        decode_client_info["port"],
    )

    async def generate_stream():
        async for chunk in _stream_decode_response(
            decode_client_info,
            endpoint,
            decode_request,
            request_id,
        ):
            yield chunk

    return StreamingResponse(generate_stream(), media_type="application/json")


@app.get("/health")
async def health():
    return Response(status_code=200)


@app.get("/healthcheck")
async def healthcheck():
    return {
        "status": "ok",
        "prefill_instances": len(app.state.prefill_clients),
        "decode_instances": len(app.state.decode_clients),
    }


@app.get("/status")
async def status():
    prefill_nodes = [
        f"{client['host']}:{client['port']}" for client in app.state.prefill_clients
    ]
    decode_nodes = [
        f"{client['host']}:{client['port']}" for client in app.state.decode_clients
    ]
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
async def create_completion(request: Request):
    return await _handle_openai_request("/completions", request)


@app.post("/v1/chat/completions")
async def create_chat_completion(request: Request):
    return await _handle_openai_request("/chat/completions", request)


if __name__ == "__main__":
    global global_args
    global_args = parse_args()

    logger.info(
        "Starting proxy host=%s port=%s prefiller_instances=%s decoder_instances=%s",
        global_args.host,
        global_args.port,
        global_args.prefiller_instances,
        global_args.decoder_instances,
    )

    import uvicorn

    uvicorn.run(app, host=global_args.host, port=global_args.port)
