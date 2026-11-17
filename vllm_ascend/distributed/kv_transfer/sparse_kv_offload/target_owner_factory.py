# SPDX-License-Identifier: Apache-2.0
"""Explicit role factories for vLLM's independently launched TP processes."""

import math
from dataclasses import dataclass

import zmq

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_contract import ContractError
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_coordinator import MultiRankTargetProvider
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_owner import (
    TargetOwnerCall,
    TargetOwnerDispatcher,
    TargetProviderClient,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_owner_rpc import (
    OwnerRpcClient,
    OwnerRpcServer,
    RpcRetentionAuthority,
)

DEFAULT_OWNER_RPC_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True)
class TargetOwnerRpcConfig:
    owner_endpoints: tuple[str, ...]
    retention_endpoint: str
    timeout: float

    @classmethod
    def from_extra(cls, extra: dict, tp_size: int):
        if type(tp_size) is not int or tp_size < 1:
            raise ContractError("Owner RPC requires the actual positive TP size")
        base = extra.get("target_owner_rpc_base_port")
        retain = extra.get("target_retention_rpc_port")
        if base is None and retain is None:
            return None
        if type(base) is not int or type(retain) is not int or not (1 <= base <= base + tp_size - 1 <= 65535):
            raise ContractError("Owner RPC base port and retention port must be explicit valid integers")
        if not 1 <= retain <= 65535 or base <= retain < base + tp_size:
            raise ContractError("Retention RPC port must be valid and distinct from every TP owner port")
        timeout = extra.get("target_owner_rpc_timeout_s", DEFAULT_OWNER_RPC_TIMEOUT_SECONDS)
        if type(timeout) not in (float, int) or not math.isfinite(timeout) or timeout <= 0:
            raise ContractError("Owner RPC timeout must be positive and finite")
        return cls(
            tuple(f"tcp://127.0.0.1:{base + rank}" for rank in range(tp_size)),
            f"tcp://127.0.0.1:{retain}",
            float(timeout),
        )


class DispatcherOwnerTransport:
    """An explicit instance, never a module-level scheduler/worker rendezvous."""

    def __init__(self, dispatcher) -> None:
        self._dispatcher = dispatcher

    def call(self, command):
        return self._dispatcher.dispatch(command)


class SchedulerTargetRpc:
    def __init__(self, config: TargetOwnerRpcConfig, registry) -> None:
        self._context = zmq.Context()
        self._clients = ()
        self._server = None
        self._closed = False
        try:
            self._server = OwnerRpcServer.retention(config.retention_endpoint, registry)
            self._clients = tuple(
                OwnerRpcClient(endpoint, timeout=config.timeout, context=self._context)
                for endpoint in config.owner_endpoints
            )
            self.provider = MultiRankTargetProvider(
                tuple(TargetProviderClient(client) for client in self._clients), parallel_owner_calls=True
            )
            # The scheduler-local coordinator already returns defensive
            # snapshots. A second copy is unnecessary on this transport.
            self.transport = DispatcherOwnerTransport(TargetOwnerDispatcher(self.provider, copy_reply=False))
        except Exception:
            for client in self._clients:
                client.close()
            if self._server is not None:
                self._server.close()
            self._context.term()
            raise

    def close(self) -> None:
        if self._closed:
            return
        self.provider.assert_can_close()
        for client in self._clients:
            client.close()
        self._server.close()
        self._context.term()
        self._closed = True


class WorkerTargetRpc:
    def __init__(self, config: TargetOwnerRpcConfig, rank: int, facade) -> None:
        if type(rank) is not int or not 0 <= rank < len(config.owner_endpoints):
            raise ContractError("Worker target RPC rank is outside the explicit TP owner set")
        self.retention = RpcRetentionAuthority(config.retention_endpoint, timeout=config.timeout)
        self._endpoint = config.owner_endpoints[rank]
        self._facade = facade
        self._server = None
        self._closed = False

    def serve_registered_owner(self, dispatcher) -> None:
        if self._closed or self._server is not None or not self._facade.ready:
            raise ContractError("Serve exactly one fully registered target owner per worker generation")
        self._server = OwnerRpcServer(self._endpoint, dispatcher, command_type=TargetOwnerCall)

    def close(self) -> None:
        if self._closed:
            return
        # Active calls/leases reject close before its reverse authority or
        # dispatch endpoint can disappear. No process-global transport exists.
        self._facade.close()
        if self._server is not None:
            self._server.close()
        self.retention.close()
        self._closed = True
