from typing import TYPE_CHECKING, Any, cast

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorRole,
    SupportsHMA,
    supports_hma,
)
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiConnector
from vllm.v1.worker import mamba_utils

from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.dspark_context import uses_sfa_dspark_kv_transfer

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.kv_events import KVConnectorKVEvents
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request


class AscendMultiConnector(MultiConnector, SupportsHMA):
    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole, kv_cache_config: "KVCacheConfig"):
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )

        transfer = vllm_config.kv_transfer_config
        self._requires_full_dspark_prompt = (
            uses_sfa_dspark_kv_transfer(vllm_config) and transfer.is_kv_producer and not transfer.is_kv_consumer
        )
        self._dspark_prefix_provider_index: int | None = None
        if self._requires_full_dspark_prompt:
            self.configure_dspark_prefix_cache(vllm_config)

        self._all_support_hma = all(supports_hma(c) for c in self._connectors)
        assert vllm_config.scheduler_config.disable_hybrid_kv_cache_manager or self._all_support_hma, (
            "HMA should not be enabled unless all sub-connectors support it"
        )
        self._configure_layerwise_reuse_completion()
        self._configure_target_provider()
        self._mamba_copy_bufs = None
        # Handle to the (V2) mamba hybrid model state while its per-layer
        # align pre-copy is deferred behind a layerwise child connector.
        self._mamba_state: Any = None
        self.requires_mamba_state_copy_after_layer_load = any(
            getattr(
                connector,
                "requires_mamba_state_copy_after_layer_load",
                False,
            )
            for connector in self._connectors
        )

        self.requires_mamba_state_copy_after_load = any(
            getattr(connector, "requires_mamba_state_copy_after_load", False) for connector in self._connectors
        )

    def configure_dspark_prefix_cache(self, vllm_config: "VllmConfig") -> bool:
        """Select a store that restores both target and persistent draft KV."""
        providers = []
        for index, connector in enumerate(self._connectors):
            configure = getattr(connector, "configure_dspark_prefix_cache", None)
            if callable(configure) and configure(vllm_config):
                providers.append(index)
        if len(providers) > 1:
            raise ValueError("DSpark prefix reuse requires exactly one paired target/draft store")
        self._dspark_prefix_provider_index = providers[0] if providers else None
        if providers:
            self._requires_full_dspark_prompt = False
        return bool(providers)

    def _configure_layerwise_reuse_completion(self) -> None:
        # Producers that report when a shared physical KV slot is safe to reuse.
        self._layerwise_slot_release_providers = [
            connector
            for connector in self._connectors
            if getattr(connector, "is_producer", False)
            and getattr(connector, "connector_worker", None) is not None
            and getattr(connector, "supports_layerwise_buffer_reuse", False)
            and callable(getattr(connector, "wait_for_layer_reuse", None))
        ]
        # All remaining connectors, which run after the slot-release providers.
        self._non_slot_release_connectors = [
            connector
            for connector in self._connectors
            if all(connector is not provider for provider in self._layerwise_slot_release_providers)
        ]
        self._external_slot_release_sink_configured = False
        if not self._layerwise_slot_release_providers:
            return
        for connector in self._connectors:
            set_waiter = getattr(connector, "set_external_slot_release_waiter", None)
            if callable(set_waiter) and set_waiter(self._wait_for_external_slot_release) is not False:
                self._external_slot_release_sink_configured = True
        self._configure_reuse_source_contract()

    def _configure_reuse_source_contract(self) -> None:
        providers = [
            p for p in self._layerwise_slot_release_providers if getattr(p, "reuse_source_contract_version", None) == 1
        ]
        if not providers:
            return
        if len(providers) != 1:
            raise RuntimeError("Reuse source contract v1 requires exactly one transport consumer")
        sinks = []
        for connector in self._connectors:
            if not callable(getattr(connector, "set_reuse_source_consumer", None)):
                continue
            wants_contract = getattr(connector, "wants_reuse_source_contract", None)
            if callable(wants_contract) and not wants_contract():
                continue
            sinks.append(connector)
        # A transport capability does not opt a plain, Store-less engine in
        # to buffer reuse. Preserve its legacy send path.
        if not sinks:
            return
        if len(sinks) != 1 or not self._external_slot_release_sink_configured:
            raise RuntimeError("Reuse source contract requires one wired Store sink")
        sinks[0].set_reuse_source_consumer(providers[0])

    def _wait_for_external_slot_release(self, layer_idx: int) -> None:
        for provider in self._layerwise_slot_release_providers:
            provider.wait_for_layer_reuse(layer_idx)

    @staticmethod
    def _receipt_mode_requested(connector) -> bool:
        """Layerwise receipt mode is opted in per engine via the kv-transfer
        extra config key ``layerwise_receipt_mode``."""
        config = getattr(connector, "vllm_config", None)
        kv_transfer = getattr(config, "kv_transfer_config", None) if config is not None else None
        extra = getattr(kv_transfer, "kv_connector_extra_config", None) or {}
        return bool(extra.get("layerwise_receipt_mode", False))

    def _configure_target_provider(self) -> None:
        """TargetProvider injection. Three-way decision:
        - mode off + no provider -> skip (legal, plain mode);
        - mode on + no provider -> startup failure (no silent
          degradation when the feature was requested);
        - more than one provider -> startup failure (exactly one D-target store).
        The discovered provider must pass the facade's capability probe, which
        also aborts startup on any gap."""
        providers = []
        for connector in self._connectors:
            worker = getattr(connector, "connector_worker", None)
            candidates = [connector, worker]
            for candidate in (connector, worker):
                if candidate is not None:
                    # A stable facade can be exposed before its owner catalog
                    # is registered. Do not serialize/duplicate the owner.
                    exposed = getattr(candidate, "__dict__", {}).get("target_provider")
                    if exposed is not None:
                        candidates.append(exposed)
            for candidate in candidates:
                if candidate is not None and callable(getattr(candidate, "acquire_target", None)):
                    if all(candidate is not existing for existing in providers):
                        providers.append(candidate)
        if len(providers) > 1:
            raise ValueError(f"Expected at most one TargetProvider, found {len(providers)}")
        if not providers:
            if any(self._receipt_mode_requested(connector) for connector in self._connectors):
                raise ValueError(
                    "layerwise_receipt_mode is enabled but no connector exposes a TargetProvider "
                    "(acquire_target); refusing to start"
                )
            return
        for connector in self._connectors:
            setter = getattr(connector, "set_target_provider", None)
            if callable(setter):
                setter(providers[0])
                return
        if any(self._receipt_mode_requested(connector) for connector in self._connectors):
            raise ValueError("Found a TargetProvider but no connector accepts set_target_provider")

    def shutdown(self) -> None:
        # Release provider waits before joining the store's receive threads.
        self._connectors[:] = [*self._layerwise_slot_release_providers, *self._non_slot_release_connectors]
        super().shutdown()

    def _receipt_restore_only_requests(self):
        """Collect request ids marked restore-only across layerwise child metadata.

        A receipt-mode request whose entire prompt is covered by the target
        lease is marked restore-only by the D scheduler. Those requests must
        bypass the normal forward load path and go through the Store restore
        driver instead.
        """
        marked = set()
        for connector in self._connectors:
            metadata = getattr(connector, "_connector_metadata", None)
            requests = getattr(metadata, "requests", None)
            if not requests:
                continue
            for req_id, meta in requests.items():
                if getattr(meta, "restore_only", False):
                    marked.add(req_id)
        return marked

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        # Receipt-mode restore-only requests are routed to the Store restore
        # driver instead of the normal per-child forward load. The Store child
        # already built its own AscendConnectorMetadata with the full-range
        # restore bounds; hand it to the restore driver and skip the upstream
        # per-child start_load_kv for this step's restore-only requests.
        restore_ids = self._receipt_restore_only_requests()
        if restore_ids:
            store = self._store_restore_connector()
            if store is None:
                raise RuntimeError(
                    "Receipt restore-only requests present but no AscendStore connector exposes a restore driver"
                )
            getter = getattr(store, "_get_connector_metadata", None)
            store_metadata = getter() if callable(getter) else getattr(store, "_connector_metadata", None)
            store.restore_kv_for_transfer(store_metadata)
            return
        super().start_load_kv(forward_context, **kwargs)

    def wait_for_layer_load(self, layer_name: str) -> None:
        if getattr(self, "_external_slot_release_sink_configured", False):
            # AscendStore owns the layer-entry reuse wait after accepting the
            # composite waiter, so provider layer-entry waits are redundant.
            connectors = self._non_slot_release_connectors
        else:
            # Without a sink, preserve the original protection by waiting on
            # providers before any sibling connector can write shared slots.
            connectors = [*self._layerwise_slot_release_providers, *self._non_slot_release_connectors]
        for connector in connectors:
            connector.wait_for_layer_load(layer_name)
        if (copy_bufs := getattr(self, "_mamba_copy_bufs", None)) is not None:
            mamba_utils.do_mamba_copy_block_for_layer(
                copy_bufs,
                layer_name,
            )
        # Mamba state copy runs after every child connector finished this
        # layer's load so the copy never reads half-loaded state. The model
        # state no-ops for non-mamba layers and for already-copied layers.
        if (mamba_state := getattr(self, "_mamba_state", None)) is not None:
            mamba_state.do_mamba_copy_for_layer(layer_name)

    def restore_kv_for_transfer(self, connector_metadata) -> None:
        """Route a receipt-mode restore-only step to the Store restore driver.

        The D scheduler marks a request whose entire prompt is covered by the
        target lease. The MooncakeLayerwise connector carries that mark in its
        metadata, but the restore driver lives on the AscendStore connector
        (its pool_worker owns the layerwise load/save loop without a model
        forward). Forward the Store's own metadata to that driver here, so the
        request restores from the Store instead of entering normal attention.
        """
        store = self._store_restore_connector()
        if store is None:
            raise RuntimeError("Restore-only transfer requires an AscendStore connector")
        store.restore_kv_for_transfer(connector_metadata)

    def _store_restore_connector(self):
        """The child connector that owns the Store restore driver."""
        for connector in self._connectors:
            worker = getattr(connector, "connector_worker", None)
            if worker is not None and callable(getattr(worker, "restore_kv_for_transfer", None)):
                return connector
        return None

    def prepare_mamba_state_copy(self, mamba_state_or_copy_bufs) -> bool:
        """Take over the mamba align pre-copy for this step.

        The V1 model runner passes its mamba copy buffers; the V2 model runner
        passes its mamba hybrid model state. Returns True when any child
        connector executes per-layer copies from its ``wait_for_layer_load``;
        see ``AscendStoreConnector.prepare_mamba_state_copy``.
        """
        if not self.requires_mamba_state_copy_after_layer_load:
            return False
        if hasattr(mamba_state_or_copy_bufs, "do_mamba_copy_for_layer"):
            self._mamba_state = mamba_state_or_copy_bufs
        else:
            mamba_utils.prepare_mamba_copy_by_layer(mamba_state_or_copy_bufs)
            self._mamba_copy_bufs = mamba_state_or_copy_bufs
        return True

    def finish_mamba_state_copy(self) -> None:
        if self._mamba_copy_bufs is not None:
            try:
                mamba_utils.finish_mamba_copy_by_layer(self._mamba_copy_bufs)
            finally:
                self._mamba_copy_bufs = None
        self._mamba_state = None

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer,
        attn_metadata: Any,
        **kwargs,
    ) -> None:
        # Phase 1: providers must close any new slot gate before returning.
        for connector in self._layerwise_slot_release_providers:
            connector.save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)
        # Phase 2: siblings may now publish work that can enable slot reuse.
        for connector in self._non_slot_release_connectors:
            connector.save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)

    def on_kv_cache_written(self, layer_name: str = "") -> None:
        # Phase 1: providers close their gates at the earliest cache-write hook.
        for connector in self._layerwise_slot_release_providers:
            hook = getattr(connector, "on_kv_cache_written", None)
            if callable(hook):
                hook(layer_name)
        # Phase 2: only then may sibling hooks publish reuse-enabling work.
        for connector in self._non_slot_release_connectors:
            hook = getattr(connector, "on_kv_cache_written", None)
            if callable(hook):
                hook(layer_name)

    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int):
        chosen_connector = self._requests_to_connector.get(request.request_id, -1)
        empty_blocks = blocks.new_empty()
        for i, connector in enumerate(self._connectors):
            needs_full_blocks = i == chosen_connector or bool(
                getattr(connector, "requires_full_blocks_on_update_after_alloc", False)
            )
            connector.update_state_after_alloc(
                request,
                blocks if needs_full_blocks else empty_blocks,
                num_external_tokens if needs_full_blocks else 0,
            )

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        # A target-only child cannot initialize the skipped draft context.
        if getattr(self, "_requires_full_dspark_prompt", False):
            return 0, False
        provider_index = getattr(self, "_dspark_prefix_provider_index", None)
        if provider_index is not None:
            tokens, load_async = self._connectors[provider_index].get_num_new_matched_tokens(
                request, num_computed_tokens
            )
            if tokens is not None and tokens > 0:
                self._requests_to_connector[request.request_id] = provider_index
            return tokens, load_async

        # Recompute offload may contain an unhashed partial block that other
        # prefix-cache connectors cannot restore. Give its request state
        # priority regardless of connector ordering.
        for i, connector in enumerate(self._connectors):
            has_preempted_request = getattr(connector, "has_preempted_request", None)
            if has_preempted_request is None or not has_preempted_request(request.request_id):
                continue
            tokens, load_async = connector.get_num_new_matched_tokens(request, num_computed_tokens)
            if tokens is None:
                return None, False
            if tokens > 0:
                self._requests_to_connector[request.request_id] = i
                return tokens, load_async
            break

        return super().get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_before_preempt(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
        num_computed_tokens: int,
    ) -> bool:
        offloaded = False
        for c in self._connectors:
            hook = getattr(c, "update_state_before_preempt", None)
            if hook is not None:
                offloaded = bool(hook(request, block_ids, num_computed_tokens)) or offloaded
        return offloaded

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        if not self._all_support_hma:
            assert len(block_ids) == 1, "HMA with multiple kv_cache_groups requires all sub-connectors to support HMA"
            return super().request_finished(request, block_ids[0])

        async_saves = 0
        kv_txfer_params = None
        for c in self._connectors:
            async_save, txfer_params = cast(SupportsHMA, c).request_finished_all_groups(request, block_ids)
            if async_save:
                async_saves += 1
            if txfer_params is not None:
                if kv_txfer_params is not None:
                    raise RuntimeError("Only one connector can produce KV transfer params")
                kv_txfer_params = txfer_params
        if async_saves > 1:
            self._extra_async_saves[request.request_id] = async_saves - 1

        self._requests_to_connector.pop(request.request_id, None)

        return async_saves > 0, kv_txfer_params

    def get_kv_connector_kv_cache_events(self) -> "KVConnectorKVEvents | None":
        """Collect worker-side KV cache events from all child connectors."""
        combined = None
        for connector in self._connectors:
            events = connector.get_kv_connector_kv_cache_events()
            if events is None:
                continue
            if combined is None:
                combined = events
            else:
                combined.add_events(events.get_all_events())
                combined.increment_workers(events.get_number_of_workers())
        return combined
