# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import logging
import os
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import msgpack
import torch
import zmq

from vllm.config.kv_transfer import KVTransferConfig
from vllm.distributed.device_communicators.pynccl_wrapper import (
    NCCLLibrary,
    buffer_type,
    cudaStream_t,
    ncclComm_t,
    ncclDataTypeEnum,
)
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.tensor_memory_pool import (  # noqa: E501
    TensorMemoryPool,
)
from vllm.utils.network_utils import get_ip
from vllm.utils.torch_utils import current_stream

logger = logging.getLogger(__name__)

DEFAULT_MEM_POOL_SIZE_GB = 32


def _thread_summary() -> str:
    thread = threading.current_thread()
    return f"{thread.name}[tid={thread.ident}]"


def _tensor_summary(tensor: torch.Tensor) -> str:
    bytes_size = tensor.element_size() * tensor.numel()
    return (
        f"shape={tuple(tensor.shape)} dtype={tensor.dtype} "
        f"device={tensor.device} bytes={bytes_size}"
    )


def _tensor_size_bytes(tensor: torch.Tensor) -> int:
    return tensor.element_size() * tensor.numel()


@contextmanager
def set_p2p_nccl_context(num_channels: str):
    original_values: dict[str, Any] = {}
    env_vars = [
        "NCCL_MAX_NCHANNELS",
        "NCCL_MIN_NCHANNELS",
        "NCCL_CUMEM_ENABLE",
        "NCCL_BUFFSIZE",
        "NCCL_PROTO",  # LL,LL128,SIMPLE
        "NCCL_ALGO",  # RING,TREE
    ]

    for var in env_vars:
        original_values[var] = os.environ.get(var)

    logger.info("set_p2p_nccl_context, original_values: %s", original_values)

    try:
        os.environ["NCCL_MAX_NCHANNELS"] = num_channels
        os.environ["NCCL_MIN_NCHANNELS"] = num_channels
        os.environ["NCCL_CUMEM_ENABLE"] = "1"
        yield
    finally:
        for var in env_vars:
            if original_values[var] is not None:
                os.environ[var] = original_values[var]
            else:
                os.environ.pop(var, None)


@dataclass
class SendQueueItem:
    tensor_id: str
    remote_address: str
    tensor: torch.Tensor


class P2pNcclEngine:
    def __init__(
        self,
        local_rank: int,
        config: KVTransferConfig,
        hostname: str = "",
        port_offset: int = 0,
        library_path: str | None = None,
    ) -> None:
        self.config = config
        self.rank = port_offset
        self.local_rank = local_rank
        self.device = torch.device(f"cuda:{self.local_rank}")
        self.nccl = NCCLLibrary(library_path)

        if not hostname:
            hostname = get_ip()
        port = int(self.config.kv_port) + port_offset
        if port == 0:
            raise ValueError("Port cannot be 0")
        self._hostname = hostname
        self._port = port

        # Each card corresponds to a ZMQ address.
        self.zmq_address = f"{self._hostname}:{self._port}"

        # If `proxy_ip` or `proxy_port` is `""`,
        # then the ping thread will not be enabled.
        proxy_ip = self.config.get_from_extra_config("proxy_ip", "")
        proxy_port = self.config.get_from_extra_config("proxy_port", "")
        if proxy_ip == "" or proxy_port == "":
            self.proxy_address = ""
            self.http_address = ""
        else:
            self.proxy_address = proxy_ip + ":" + proxy_port
            # the `http_port` must be consistent with the port of OpenAI.
            http_port = self.config.get_from_extra_config("http_port", None)
            if http_port is None:
                example_cfg = {
                    "kv_connector": "P2pNcclConnector",
                    "kv_connector_extra_config": {"http_port": 8000},
                }
                example = (
                    f"--port=8000 --kv-transfer-config='{json.dumps(example_cfg)}'"
                )
                raise ValueError(
                    "kv_connector_extra_config.http_port is required. "
                    f"Example: {example}"
                )
            self.http_address = f"{self._hostname}:{http_port}"

        self.context = zmq.Context()
        self.router_socket = self.context.socket(zmq.ROUTER)
        self.router_socket.bind(f"tcp://{self.zmq_address}")

        self.poller = zmq.Poller()
        self.poller.register(self.router_socket, zmq.POLLIN)

        self.send_store_cv = threading.Condition()
        self.send_queue_cv = threading.Condition()
        self.recv_store_cv = threading.Condition()

        self.send_stream = torch.cuda.Stream()
        self.recv_stream = torch.cuda.Stream()

        mem_pool_size_gb = float(
            self.config.get_from_extra_config(
                "mem_pool_size_gb", DEFAULT_MEM_POOL_SIZE_GB
            )
        )
        self.pool = TensorMemoryPool(
            max_block_size=int(mem_pool_size_gb * 1024**3)
        )  # GB

        # The sending type includes tree mutually exclusive options:
        # PUT, GET, PUT_ASYNC.
        self.send_type = self.config.get_from_extra_config("send_type", "PUT_ASYNC")
        if self.send_type == "GET":
            # tensor_id: torch.Tensor
            self.send_store: dict[str, torch.Tensor] = {}
        else:
            # PUT or PUT_ASYNC
            # tensor_id: torch.Tensor
            self.send_queue: deque[SendQueueItem] = deque()
            if self.send_type == "PUT_ASYNC":
                self._send_thread = threading.Thread(
                    target=self.send_async, daemon=True
                )
                self._send_thread.start()

        # tensor_id: torch.Tensor/(addr, dtype, shape)
        self.recv_store: dict[str, Any] = {}
        self.recv_request_id_to_tensor_ids: dict[str, set[str]] = {}
        self.send_request_id_to_tensor_ids: dict[str, set[str]] = {}
        self.socks: dict[str, Any] = {}  # remote_address: client socket
        self.comms: dict[str, Any] = {}  # remote_address: (ncclComm_t, rank)

        self.buffer_size = 0
        self.buffer_size_threshold = float(self.config.kv_buffer_size)

        self.nccl_num_channels = self.config.get_from_extra_config(
            "nccl_num_channels", "8"
        )

        self._listener_thread = threading.Thread(
            target=self.listen_for_requests, daemon=True
        )
        self._listener_thread.start()

        self._ping_thread = None
        if port_offset == 0 and self.proxy_address != "":
            self._ping_thread = threading.Thread(target=self.ping, daemon=True)
            self._ping_thread.start()

        logger.info(
            "💯P2pNcclEngine init, rank:%d, local_rank:%d, http_address:%s, "
            "zmq_address:%s, proxy_address:%s, send_type:%s, buffer_size_"
            "threshold:%.2f, nccl_num_channels:%s",
            self.rank,
            self.local_rank,
            self.http_address,
            self.zmq_address,
            self.proxy_address,
            self.send_type,
            self.buffer_size_threshold,
            self.nccl_num_channels,
        )
        logger.info(
            "P2P_ENGINE_STREAMS rank=%d send_stream=%s recv_stream=%s "
            "thread=%s device=%s",
            self.rank,
            self.send_stream.cuda_stream,
            self.recv_stream.cuda_stream,
            _thread_summary(),
            self.device,
        )

    def create_connect(self, remote_address: str | None = None):
        assert remote_address is not None
        logger.info(
            "P2P_CREATE_CONNECT_BEGIN local_zmq=%s remote_address=%s rank=%d "
            "thread=%s existing_sock=%s existing_comm=%s",
            self.zmq_address,
            remote_address,
            self.rank,
            _thread_summary(),
            remote_address in self.socks,
            remote_address in self.comms,
        )
        if remote_address not in self.socks:
            sock = self.context.socket(zmq.DEALER)
            sock.setsockopt_string(zmq.IDENTITY, self.zmq_address)
            sock.connect(f"tcp://{remote_address}")
            self.socks[remote_address] = sock
            if remote_address in self.comms:
                logger.info(
                    "👋comm exists, remote_address:%s, comms:%s",
                    remote_address,
                    self.comms,
                )
                logger.info(
                    "P2P_CREATE_CONNECT_REUSE local_zmq=%s remote_address=%s "
                    "rank=%d thread=%s",
                    self.zmq_address,
                    remote_address,
                    self.rank,
                    _thread_summary(),
                )
                return sock, self.comms[remote_address]

            unique_id = self.nccl.ncclGetUniqueId()
            data = {"cmd": "NEW", "unique_id": bytes(unique_id.internal)}
            logger.info(
                "P2P_CREATE_CONNECT_SEND_NEW local_zmq=%s remote_address=%s "
                "rank=%d thread=%s",
                self.zmq_address,
                remote_address,
                self.rank,
                _thread_summary(),
            )
            sock.send(msgpack.dumps(data))

            with torch.accelerator.device_index(self.device.index):
                rank = 0
                with set_p2p_nccl_context(self.nccl_num_channels):
                    comm: ncclComm_t = self.nccl.ncclCommInitRank(2, unique_id, rank)
                self.comms[remote_address] = (comm, rank)
                logger.info(
                    "🤝ncclCommInitRank Success, %s👉%s, MyRank:%s",
                    self.zmq_address,
                    remote_address,
                    rank,
                )

        logger.info(
            "P2P_CREATE_CONNECT_END local_zmq=%s remote_address=%s rank=%d "
            "thread=%s",
            self.zmq_address,
            remote_address,
            self.rank,
            _thread_summary(),
        )

        return self.socks[remote_address], self.comms[remote_address]

    def send_tensor(
        self,
        tensor_id: str,
        tensor: torch.Tensor,
        remote_address: str | None = None,
    ) -> bool:
        tensor_size = _tensor_size_bytes(tensor)
        logger.info(
            "P2P_SEND_TENSOR_BEGIN tensor_id=%s remote_address=%s rank=%d "
            "send_type=%s tensor=%s thread=%s",
            tensor_id,
            remote_address,
            self.rank,
            self.send_type,
            _tensor_summary(tensor),
            _thread_summary(),
        )
        if remote_address is None:
            with self.recv_store_cv:
                self.recv_store[tensor_id] = tensor
                self.recv_store_cv.notify()
                recv_store_size = len(self.recv_store)
            logger.info(
                "P2P_SEND_TENSOR_LOCAL_STORE tensor_id=%s rank=%d "
                "recv_store_size=%d thread=%s",
                tensor_id,
                self.rank,
                recv_store_size,
                _thread_summary(),
            )
            return True

        item = SendQueueItem(
            tensor_id=tensor_id, remote_address=remote_address, tensor=tensor
        )

        if self.send_type == "PUT":
            logger.info(
                "P2P_SEND_TENSOR_SYNC_DISPATCH tensor_id=%s remote_address=%s "
                "rank=%d thread=%s",
                tensor_id,
                remote_address,
                self.rank,
                _thread_summary(),
            )
            return self.send_sync(item)

        if self.send_type == "PUT_ASYNC":
            with self.send_queue_cv:
                self.send_queue.append(item)
                self.send_queue_cv.notify()
                queue_len = len(self.send_queue)
            logger.info(
                "P2P_SEND_TENSOR_ENQUEUED tensor_id=%s remote_address=%s rank=%d "
                "queue_len=%d bytes=%d thread=%s",
                tensor_id,
                remote_address,
                self.rank,
                queue_len,
                tensor_size,
                _thread_summary(),
            )
            return True

        # GET
        with self.send_store_cv:
            if tensor_size > self.buffer_size_threshold:
                logger.warning(
                    "❗[GET]tensor_id:%s, tensor_size:%d, is greater than"
                    "buffer size threshold :%d, skip send to %s, rank:%d",
                    tensor_id,
                    tensor_size,
                    self.buffer_size_threshold,
                    remote_address,
                    self.rank,
                )
                return False
            while self.buffer_size + tensor_size > self.buffer_size_threshold:
                assert len(self.send_store) > 0
                oldest_tensor_id = next(iter(self.send_store))
                oldest_tensor = self.send_store.pop(oldest_tensor_id)
                oldest_tensor_size = (
                    oldest_tensor.element_size() * oldest_tensor.numel()
                )
                self.buffer_size -= oldest_tensor_size
                logger.debug(
                    "⛔[GET]Send to %s, tensor_id:%s, tensor_size:%d,"
                    " buffer_size:%d, oldest_tensor_size:%d, rank:%d",
                    remote_address,
                    tensor_id,
                    tensor_size,
                    self.buffer_size,
                    oldest_tensor_size,
                    self.rank,
                )

            self.send_store[tensor_id] = tensor
            self.buffer_size += tensor_size
            logger.debug(
                "🔵[GET]Send to %s, tensor_id:%s, tensor_size:%d, "
                "shape:%s, rank:%d, buffer_size:%d(%.2f%%)",
                remote_address,
                tensor_id,
                tensor_size,
                tensor.shape,
                self.rank,
                self.buffer_size,
                self.buffer_size / self.buffer_size_threshold * 100,
            )
        logger.info(
            "P2P_SEND_TENSOR_GET_STORED tensor_id=%s remote_address=%s rank=%d "
            "buffer_size=%d bytes=%d send_store_size=%d thread=%s",
            tensor_id,
            remote_address,
            self.rank,
            self.buffer_size,
            tensor_size,
            len(self.send_store),
            _thread_summary(),
        )
        return True

    def recv_tensor(
        self,
        tensor_id: str,
        remote_address: str | None = None,
    ) -> torch.Tensor:
        if self.send_type == "PUT" or self.send_type == "PUT_ASYNC":
            start_time = time.time()
            logger.info(
                "P2P_RECV_WAIT_BEGIN tensor_id=%s remote_address=%s rank=%d "
                "thread=%s",
                tensor_id,
                remote_address,
                self.rank,
                _thread_summary(),
            )
            with self.recv_store_cv:
                while tensor_id not in self.recv_store:
                    logger.info(
                        "P2P_RECV_WAIT_BLOCK tensor_id=%s remote_address=%s "
                        "rank=%d recv_store_size=%d thread=%s",
                        tensor_id,
                        remote_address,
                        self.rank,
                        len(self.recv_store),
                        _thread_summary(),
                    )
                    self.recv_store_cv.wait()
                tensor = self.recv_store[tensor_id]

            if tensor is not None:
                if isinstance(tensor, tuple):
                    addr, dtype, shape = tensor
                    tensor = self.pool.load_tensor(addr, dtype, shape, self.device)
                    logger.info(
                        "P2P_RECV_WAIT_HIT_MEMPOOL tensor_id=%s remote_address=%s "
                        "rank=%d duration_ms=%.3f addr=%d shape=%s dtype=%s "
                        "thread=%s",
                        tensor_id,
                        remote_address,
                        self.rank,
                        (time.time() - start_time) * 1000,
                        addr,
                        tuple(shape),
                        dtype,
                        _thread_summary(),
                    )
                else:
                    self.buffer_size -= _tensor_size_bytes(tensor)
                    logger.info(
                        "P2P_RECV_WAIT_HIT tensor_id=%s remote_address=%s rank=%d "
                        "duration_ms=%.3f tensor=%s buffer_size=%d thread=%s",
                        tensor_id,
                        remote_address,
                        self.rank,
                        (time.time() - start_time) * 1000,
                        _tensor_summary(tensor),
                        self.buffer_size,
                        _thread_summary(),
                    )
            else:
                duration = time.time() - start_time
                logger.warning(
                    "🔴[PUT]Recv From %s, tensor_id:%s, duration:%.3fms, rank:%d",
                    remote_address,
                    tensor_id,
                    duration * 1000,
                    self.rank,
                )
            return tensor

        # GET
        if remote_address is None:
            return None

        if remote_address not in self.socks:
            self.create_connect(remote_address)

        sock = self.socks[remote_address]
        comm, rank = self.comms[remote_address]

        data = {"cmd": "GET", "tensor_id": tensor_id}
        logger.info(
            "P2P_GET_REQUEST_BEGIN tensor_id=%s remote_address=%s rank=%d "
            "thread=%s",
            tensor_id,
            remote_address,
            self.rank,
            _thread_summary(),
        )
        sock.send(msgpack.dumps(data))

        message = sock.recv()
        data = msgpack.loads(message)
        logger.info(
            "P2P_GET_RESPONSE tensor_id=%s remote_address=%s rank=%d ret=%s "
            "meta=%s thread=%s",
            tensor_id,
            remote_address,
            self.rank,
            data.get("ret"),
            data,
            _thread_summary(),
        )
        if data["ret"] != 0:
            logger.warning(
                "🔴[GET]Recv From %s, tensor_id: %s, ret: %d",
                remote_address,
                tensor_id,
                data["ret"],
            )
            return None

        with torch.cuda.stream(self.recv_stream):
            tensor = torch.empty(
                data["shape"], dtype=getattr(torch, data["dtype"]), device=self.device
            )
        logger.info(
            "P2P_GET_ALLOC_DONE tensor_id=%s remote_address=%s rank=%d tensor=%s "
            "stream=%s thread=%s",
            tensor_id,
            remote_address,
            self.rank,
            _tensor_summary(tensor),
            self.recv_stream.cuda_stream,
            _thread_summary(),
        )

        self.recv(comm, tensor, rank ^ 1, self.recv_stream)
        logger.info(
            "P2P_GET_RECV_DONE tensor_id=%s remote_address=%s rank=%d tensor=%s "
            "thread=%s",
            tensor_id,
            remote_address,
            self.rank,
            _tensor_summary(tensor),
            _thread_summary(),
        )

        return tensor

    def listen_for_requests(self):
        while True:
            socks = dict(self.poller.poll())
            if self.router_socket not in socks:
                continue

            remote_address, message = self.router_socket.recv_multipart()
            data = msgpack.loads(message)
            remote_address_str = remote_address.decode()
            logger.info(
                "P2P_LISTEN_MESSAGE local_zmq=%s remote_address=%s cmd=%s rank=%d "
                "thread=%s",
                self.zmq_address,
                remote_address_str,
                data.get("cmd"),
                self.rank,
                _thread_summary(),
            )
            if data["cmd"] == "NEW":
                unique_id = self.nccl.unique_id_from_bytes(bytes(data["unique_id"]))
                with torch.accelerator.device_index(self.device.index):
                    rank = 1
                    with set_p2p_nccl_context(self.nccl_num_channels):
                        comm: ncclComm_t = self.nccl.ncclCommInitRank(
                            2, unique_id, rank
                        )
                    self.comms[remote_address.decode()] = (comm, rank)
                    logger.info(
                        "🤝ncclCommInitRank Success, %s👈%s, MyRank:%s",
                        self.zmq_address,
                        remote_address_str,
                        rank,
                    )
            elif data["cmd"] == "PUT":
                tensor_id = data["tensor_id"]
                try:
                    logger.info(
                        "P2P_LISTEN_PUT_BEGIN remote_address=%s tensor_id=%s "
                        "rank=%d shape=%s dtype=%s thread=%s device=%s",
                        remote_address_str,
                        tensor_id,
                        self.rank,
                        tuple(data["shape"]),
                        data["dtype"],
                        _thread_summary(),
                        self.device,
                    )
                    alloc_start = time.perf_counter()
                    with torch.cuda.stream(self.recv_stream):
                        tensor = torch.empty(
                            data["shape"],
                            dtype=getattr(torch, data["dtype"]),
                            device=self.device,
                        )
                    logger.info(
                        "P2P_LISTEN_PUT_ALLOC_DONE remote_address=%s tensor_id=%s "
                        "rank=%d duration_ms=%.3f stream=%s tensor=%s "
                        "thread=%s",
                        remote_address_str,
                        tensor_id,
                        self.rank,
                        (time.perf_counter() - alloc_start) * 1000,
                        self.recv_stream.cuda_stream,
                        _tensor_summary(tensor),
                        _thread_summary(),
                    )
                    self.router_socket.send_multipart([remote_address, b"0"])
                    logger.info(
                        "P2P_LISTEN_PUT_ACK_SENT remote_address=%s tensor_id=%s "
                        "rank=%d thread=%s",
                        remote_address_str,
                        tensor_id,
                        self.rank,
                        _thread_summary(),
                    )
                    comm, rank = self.comms[remote_address_str]
                    recv_start = time.perf_counter()
                    self.recv(comm, tensor, rank ^ 1, self.recv_stream)
                    tensor_size = _tensor_size_bytes(tensor)
                    logger.info(
                        "P2P_LISTEN_PUT_RECV_DONE remote_address=%s tensor_id=%s "
                        "rank=%d duration_ms=%.3f tensor=%s thread=%s",
                        remote_address_str,
                        tensor_id,
                        self.rank,
                        (time.perf_counter() - recv_start) * 1000,
                        _tensor_summary(tensor),
                        _thread_summary(),
                    )
                    if self.buffer_size + tensor_size > self.buffer_size_threshold:
                        # Store Tensor in memory pool
                        addr = self.pool.store_tensor(tensor)
                        tensor = (addr, tensor.dtype, tensor.shape)
                        logger.warning(
                            "🔴[PUT]Recv Tensor, Out Of Threshold, "
                            "%s👈%s, data:%s, addr:%d",
                            self.zmq_address,
                            remote_address.decode(),
                            data,
                            addr,
                        )
                    else:
                        self.buffer_size += tensor_size
                        logger.info(
                            "P2P_LISTEN_PUT_BUFFERED remote_address=%s tensor_id=%s "
                            "rank=%d buffer_size=%d threshold=%.0f thread=%s",
                            remote_address_str,
                            tensor_id,
                            self.rank,
                            self.buffer_size,
                            self.buffer_size_threshold,
                            _thread_summary(),
                        )

                except torch.cuda.OutOfMemoryError:
                    self.router_socket.send_multipart([remote_address, b"1"])
                    tensor = None
                    logger.warning(
                        "🔴[PUT]Recv Tensor, Out Of Memory, %s👈%s, data:%s",
                        self.zmq_address,
                        remote_address_str,
                        data,
                    )
                except Exception:
                    self.router_socket.send_multipart([remote_address, b"2"])
                    tensor = None
                    logger.exception(
                        "P2P_LISTEN_PUT_EXCEPTION remote_address=%s tensor_id=%s "
                        "rank=%d thread=%s",
                        remote_address_str,
                        tensor_id,
                        self.rank,
                        _thread_summary(),
                    )

                with self.recv_store_cv:
                    self.recv_store[tensor_id] = tensor
                    self.have_received_tensor_id(tensor_id)
                    self.recv_store_cv.notify()
                    recv_store_size = len(self.recv_store)
                logger.info(
                    "P2P_LISTEN_PUT_STORE_DONE remote_address=%s tensor_id=%s "
                    "rank=%d recv_store_size=%d thread=%s",
                    remote_address_str,
                    tensor_id,
                    self.rank,
                    recv_store_size,
                    _thread_summary(),
                )

            elif data["cmd"] == "GET":
                tensor_id = data["tensor_id"]
                with self.send_store_cv:
                    tensor = self.send_store.pop(tensor_id, None)
                    if tensor is not None:
                        data = {
                            "ret": 0,
                            "shape": tensor.shape,
                            "dtype": str(tensor.dtype).replace("torch.", ""),
                        }
                        # LRU
                        self.send_store[tensor_id] = tensor
                        self.have_sent_tensor_id(tensor_id)
                    else:
                        data = {"ret": 1}

                self.router_socket.send_multipart([remote_address, msgpack.dumps(data)])
                logger.info(
                    "P2P_LISTEN_GET_RESPONSE remote_address=%s tensor_id=%s "
                    "rank=%d ret=%s send_store_size=%d thread=%s",
                    remote_address_str,
                    tensor_id,
                    self.rank,
                    data["ret"],
                    len(self.send_store),
                    _thread_summary(),
                )

                if data["ret"] == 0:
                    comm, rank = self.comms[remote_address_str]
                    logger.info(
                        "P2P_LISTEN_GET_SEND_BEGIN remote_address=%s tensor_id=%s "
                        "rank=%d tensor=%s thread=%s",
                        remote_address_str,
                        tensor_id,
                        self.rank,
                        _tensor_summary(tensor),
                        _thread_summary(),
                    )
                    self.send(comm, tensor.to(self.device), rank ^ 1, self.send_stream)
                    logger.info(
                        "P2P_LISTEN_GET_SEND_DONE remote_address=%s tensor_id=%s "
                        "rank=%d thread=%s",
                        remote_address_str,
                        tensor_id,
                        self.rank,
                        _thread_summary(),
                    )
            else:
                logger.warning(
                    "🚧Unexpected, Received message from %s, data:%s",
                    remote_address,
                    data,
                )

    def have_sent_tensor_id(self, tensor_id: str):
        request_id = tensor_id.split("#")[0]
        if request_id not in self.send_request_id_to_tensor_ids:
            self.send_request_id_to_tensor_ids[request_id] = set()
        self.send_request_id_to_tensor_ids[request_id].add(tensor_id)
        logger.info(
            "P2P_REQUEST_SEND_PROGRESS request_id=%s tensor_id=%s rank=%d "
            "sent_tensor_count=%d",
            request_id,
            tensor_id,
            self.rank,
            len(self.send_request_id_to_tensor_ids[request_id]),
        )

    def have_received_tensor_id(self, tensor_id: str):
        request_id = tensor_id.split("#")[0]
        if request_id not in self.recv_request_id_to_tensor_ids:
            self.recv_request_id_to_tensor_ids[request_id] = set()
        self.recv_request_id_to_tensor_ids[request_id].add(tensor_id)
        logger.info(
            "P2P_REQUEST_RECV_PROGRESS request_id=%s tensor_id=%s rank=%d "
            "recv_tensor_count=%d",
            request_id,
            tensor_id,
            self.rank,
            len(self.recv_request_id_to_tensor_ids[request_id]),
        )

    def send_async(self):
        while True:
            with self.send_queue_cv:
                while not self.send_queue:
                    self.send_queue_cv.wait()
                item = self.send_queue.popleft()
                if not self.send_queue:
                    self.send_queue_cv.notify()
                queue_len = len(self.send_queue)
            logger.info(
                "P2P_SEND_ASYNC_DEQUEUED tensor_id=%s remote_address=%s rank=%d "
                "queue_len=%d thread=%s",
                item.tensor_id,
                item.remote_address,
                self.rank,
                queue_len,
                _thread_summary(),
            )
            self.send_sync(item)

    def wait_for_sent(self):
        if self.send_type == "PUT_ASYNC":
            start_time = time.time()
            logger.info(
                "P2P_WAIT_FOR_SENT_BEGIN rank=%d queue_len=%d thread=%s",
                self.rank,
                len(self.send_queue),
                _thread_summary(),
            )
            with self.send_queue_cv:
                while self.send_queue:
                    logger.info(
                        "P2P_WAIT_FOR_SENT_BLOCK rank=%d queue_len=%d thread=%s",
                        self.rank,
                        len(self.send_queue),
                        _thread_summary(),
                    )
                    self.send_queue_cv.wait()
            duration = time.time() - start_time
            logger.info(
                "P2P_WAIT_FOR_SENT_END rank=%d duration_ms=%.3f thread=%s",
                self.rank,
                duration * 1000,
                _thread_summary(),
            )

    def send_sync(self, item: SendQueueItem) -> bool:
        if item.remote_address is None:
            return False
        if item.remote_address not in self.socks:
            self.create_connect(item.remote_address)

        tensor = item.tensor

        sock = self.socks[item.remote_address]
        comm, rank = self.comms[item.remote_address]
        data = {
            "cmd": "PUT",
            "tensor_id": item.tensor_id,
            "shape": tensor.shape,
            "dtype": str(tensor.dtype).replace("torch.", ""),
        }
        logger.info(
            "P2P_SEND_SYNC_RPC_BEGIN tensor_id=%s remote_address=%s rank=%d "
            "shape=%s dtype=%s bytes=%d thread=%s",
            item.tensor_id,
            item.remote_address,
            self.rank,
            tuple(tensor.shape),
            tensor.dtype,
            _tensor_size_bytes(tensor),
            _thread_summary(),
        )
        sock.send(msgpack.dumps(data))
        logger.info(
            "P2P_SEND_SYNC_RPC_CONTROL_SENT tensor_id=%s remote_address=%s "
            "rank=%d thread=%s",
            item.tensor_id,
            item.remote_address,
            self.rank,
            _thread_summary(),
        )

        response = sock.recv()
        logger.info(
            "P2P_SEND_SYNC_RPC_ACK tensor_id=%s remote_address=%s rank=%d "
            "response=%s thread=%s",
            item.tensor_id,
            item.remote_address,
            self.rank,
            response,
            _thread_summary(),
        )
        if response != b"0":
            logger.error(
                "🔴Send Tensor, Peer Out Of Memory/Threshold, %s 👉 %s, "
                "MyRank:%s, data:%s, tensor:%s, size:%fGB, response:%s",
                self.zmq_address,
                item.remote_address,
                rank,
                data,
                tensor.shape,
                tensor.element_size() * tensor.numel() / 1024**3,
                response.decode(),
            )
            return False

        self.send(comm, tensor.to(self.device), rank ^ 1, self.send_stream)
        logger.info(
            "P2P_SEND_SYNC_RPC_DONE tensor_id=%s remote_address=%s rank=%d",
            item.tensor_id,
            item.remote_address,
            self.rank,
        )

        if self.send_type == "PUT_ASYNC":
            self.have_sent_tensor_id(item.tensor_id)

        return True

    def get_finished(
        self, finished_req_ids: set[str], no_compile_layers
    ) -> tuple[set[str] | None, set[str] | None]:
        """
        Notifies worker-side connector ids of requests that have
        finished generating tokens.

        Returns:
            ids of requests that have finished asynchronous transfer,
            tuple of (sending/saving ids, recving/loading ids).
            The finished saves/sends req ids must belong to a set provided in a
            call to this method (this call or a prior one).
        """

        # Clear the buffer upon request completion.
        logger.info(
            "P2P_GET_FINISHED_BEGIN rank=%d finished_req_count=%d ids=%s",
            self.rank,
            len(finished_req_ids),
            sorted(finished_req_ids),
        )
        for request_id in finished_req_ids:
            for layer_name in no_compile_layers:
                tensor_id = request_id + "#" + layer_name
                if tensor_id in self.recv_store:
                    with self.recv_store_cv:
                        tensor = self.recv_store.pop(tensor_id, None)
                        self.send_request_id_to_tensor_ids.pop(request_id, None)
                        self.recv_request_id_to_tensor_ids.pop(request_id, None)
                    if isinstance(tensor, tuple):
                        addr, _, _ = tensor
                        self.pool.free(addr)
                    logger.info(
                        "P2P_GET_FINISHED_CLEANUP request_id=%s tensor_id=%s "
                        "rank=%d recv_store_size=%d",
                        request_id,
                        tensor_id,
                        self.rank,
                        len(self.recv_store),
                    )

        # TODO:Retrieve requests that have already sent the KV cache.
        finished_sending: set[str] = set()

        # TODO:Retrieve requests that have already received the KV cache.
        finished_recving: set[str] = set()

        logger.info(
            "P2P_GET_FINISHED_END rank=%d finished_sending=%s "
            "finished_recving=%s",
            self.rank,
            sorted(finished_sending),
            sorted(finished_recving),
        )
        return finished_sending or None, finished_recving or None

    def ping(self):
        sock = self.context.socket(zmq.DEALER)
        sock.setsockopt_string(zmq.IDENTITY, self.zmq_address)
        logger.info(
            "P2P_PING_THREAD_START zmq_address=%s proxy_address=%s rank=%d "
            "thread=%s",
            self.zmq_address,
            self.proxy_address,
            self.rank,
            _thread_summary(),
        )
        sock.connect(f"tcp://{self.proxy_address}")
        data = {
            "type": "P" if self.config.is_kv_producer else "D",
            "http_address": self.http_address,
            "zmq_address": self.zmq_address,
        }
        while True:
            sock.send(msgpack.dumps(data))
            time.sleep(3)

    def send(self, comm, tensor: torch.Tensor, dst: int, stream=None):
        assert tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}"
        )
        if stream is None:
            stream = current_stream()

        logger.info(
            "P2P_NCCL_SEND_BEGIN rank=%d dst=%d tensor=%s stream=%s thread=%s",
            self.rank,
            dst,
            _tensor_summary(tensor),
            stream.cuda_stream,
            _thread_summary(),
        )
        nccl_call_start = time.perf_counter()
        with torch.cuda.stream(stream):
            self.nccl.ncclSend(
                buffer_type(tensor.data_ptr()),
                tensor.numel(),
                ncclDataTypeEnum.from_torch(tensor.dtype),
                dst,
                comm,
                cudaStream_t(stream.cuda_stream),
            )
        logger.info(
            "P2P_NCCL_SEND_CALL_RETURN rank=%d dst=%d duration_ms=%.3f "
            "stream=%s thread=%s",
            self.rank,
            dst,
            (time.perf_counter() - nccl_call_start) * 1000,
            stream.cuda_stream,
            _thread_summary(),
        )
        sync_start = time.perf_counter()
        logger.info(
            "P2P_NCCL_SEND_SYNC_BEGIN rank=%d dst=%d stream=%s thread=%s",
            self.rank,
            dst,
            stream.cuda_stream,
            _thread_summary(),
        )
        stream.synchronize()
        logger.info(
            "P2P_NCCL_SEND_DONE rank=%d dst=%d tensor=%s sync_ms=%.3f "
            "stream=%s thread=%s",
            self.rank,
            dst,
            _tensor_summary(tensor),
            (time.perf_counter() - sync_start) * 1000,
            stream.cuda_stream,
            _thread_summary(),
        )

    def recv(self, comm, tensor: torch.Tensor, src: int, stream=None):
        assert tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}"
        )
        if stream is None:
            stream = current_stream()

        logger.info(
            "P2P_NCCL_RECV_BEGIN rank=%d src=%d tensor=%s stream=%s thread=%s",
            self.rank,
            src,
            _tensor_summary(tensor),
            stream.cuda_stream,
            _thread_summary(),
        )
        nccl_call_start = time.perf_counter()
        with torch.cuda.stream(stream):
            self.nccl.ncclRecv(
                buffer_type(tensor.data_ptr()),
                tensor.numel(),
                ncclDataTypeEnum.from_torch(tensor.dtype),
                src,
                comm,
                cudaStream_t(stream.cuda_stream),
            )
        logger.info(
            "P2P_NCCL_RECV_CALL_RETURN rank=%d src=%d duration_ms=%.3f "
            "stream=%s thread=%s",
            self.rank,
            src,
            (time.perf_counter() - nccl_call_start) * 1000,
            stream.cuda_stream,
            _thread_summary(),
        )
        sync_start = time.perf_counter()
        logger.info(
            "P2P_NCCL_RECV_SYNC_BEGIN rank=%d src=%d stream=%s thread=%s",
            self.rank,
            src,
            stream.cuda_stream,
            _thread_summary(),
        )
        stream.synchronize()
        logger.info(
            "P2P_NCCL_RECV_DONE rank=%d src=%d tensor=%s sync_ms=%.3f "
            "stream=%s thread=%s",
            self.rank,
            src,
            _tensor_summary(tensor),
            (time.perf_counter() - sync_start) * 1000,
            stream.cuda_stream,
            _thread_summary(),
        )

    def close(self) -> None:
        self._listener_thread.join()
        if self.send_type == "PUT_ASYNC":
            self._send_thread.join()
        if self._ping_thread is not None:
            self._ping_thread.join()
