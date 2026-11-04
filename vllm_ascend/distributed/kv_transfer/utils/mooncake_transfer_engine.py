import threading
from dataclasses import dataclass
from uuid import uuid4

from vllm.logger import logger

from vllm_ascend.distributed.kv_transfer.utils.allocation_lifetime import AllocationLifetime, AllocationPin

_WILDCARD_LOCATION = "*"


@dataclass(frozen=True)
class RegistrationSnapshot:
    engine_instance: str
    generation: int
    regions: tuple[tuple[int, int, str], ...]
    # "host:rpc_port" the remote side opens a mooncake session to. Per-engine,
    # shared by every region; carried into RegisteredRegion so the producer can
    # reach the target without a separate zmq handshake.
    transport_endpoint: str = ""

    def __getstate__(self):
        raise TypeError("Local TE addresses must not enter target wire metadata")


class GlobalTE:
    def __init__(self):
        self.transfer_engine = None
        self.is_register_buffer: bool = False
        self.transfer_engine_lock = threading.Lock()
        self.register_buffer_lock = threading.Lock()
        self._registered_regions: list[tuple[int, int, str]] = []
        self._registration_instance = uuid4().hex
        self._registration_generation = 0
        self._registration_complete = False
        self._handshake_host: str = ""
        self._target_lifetime = AllocationLifetime("Mooncake TE registration")

    def get_transfer_engine(self, hostname: str, device_name: str | None):
        if self.transfer_engine is None:
            with self.transfer_engine_lock:
                # Double-Checked Locking
                if self.transfer_engine is None:
                    try:
                        from mooncake.engine import TransferEngine  # type: ignore
                    except ImportError as e:
                        raise ImportError(
                            "Please install mooncake by following the instructions at "
                            "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "  # noqa: E501
                            "to run vLLM with MooncakeConnector."
                        ) from e
                    self.transfer_engine = TransferEngine()
                    device_name = device_name if device_name is not None else ""
                    ret_value = self.transfer_engine.initialize(hostname, "P2PHANDSHAKE", "ascend", device_name)
                    if ret_value != 0:
                        raise RuntimeError(f"TransferEngine initialization failed with ret_value: {ret_value}")
                self._handshake_host = hostname
        return self.transfer_engine

    def transport_endpoint(self) -> str:
        """The 'host:rpc_port' a remote producer opens a mooncake session to.

        Empty until the engine is initialized; the catalog rejects an empty
        endpoint so a target lease never carries an unreachable address.
        """
        if self.transfer_engine is None or not self._handshake_host:
            return ""
        return f"{self._handshake_host}:{self.transfer_engine.get_rpc_port()}"

    def _unregister_regions(
        self,
        regions: list[tuple[int, int, str]],
    ) -> tuple[list[tuple[int, int, str]], list[str]]:
        """Unregister regions in reverse order and return any failures."""
        assert self.transfer_engine is not None
        failed_regions: list[tuple[int, int, str]] = []
        failure_messages: list[str] = []
        for region in reversed(regions):
            ptr, _, _ = region
            try:
                ret_value = self.transfer_engine.unregister_memory(ptr)
            except Exception as exc:  # noqa: BLE001
                failed_regions.append(region)
                failure_messages.append(f"ptr={ptr}: {exc}")
                continue
            if ret_value != 0:
                failed_regions.append(region)
                failure_messages.append(f"ptr={ptr}: ret_value={ret_value}")

        failed_regions.reverse()
        return failed_regions, failure_messages

    def register_buffer(
        self,
        ptrs: list[int],
        sizes: list[int],
        locations: list[str] | None = None,
    ) -> None:
        normalized_locations = locations if locations is not None else [_WILDCARD_LOCATION] * len(ptrs)
        regions = list(zip(ptrs, sizes, normalized_locations))

        with self.register_buffer_lock:
            assert self.transfer_engine is not None, "Transfer engine must be initialized"
            if self.is_register_buffer:
                return

            self._registration_complete = False
            registered_regions: list[tuple[int, int, str]] = []
            try:
                for ptr, size, location in regions:
                    ret_value = self.transfer_engine.register_memory(ptr, size, location)
                    if ret_value != 0 and location != _WILDCARD_LOCATION:
                        # Older mooncake wheels may not support every explicit
                        # location kind (e.g. registering a host-registered
                        # shared segment under an "npu:*" location). Retry once
                        # with the wildcard location so the transfer engine
                        # auto-detects the memory type.
                        logger.warning(
                            "Mooncake register_memory ret_value=%s for ptr=%#x size=%s location=%s; "
                            "retrying with wildcard location",
                            ret_value,
                            ptr,
                            size,
                            location,
                        )
                        ret_value = self.transfer_engine.register_memory(ptr, size, _WILDCARD_LOCATION)
                        if ret_value == 0:
                            location = _WILDCARD_LOCATION
                    if ret_value != 0:
                        raise RuntimeError(
                            "Mooncake memory registration failed: "
                            f"ptr={ptr:#x} size={size} location={location} ret_value={ret_value}"
                        )
                    registered_regions.append((ptr, size, location))
            except Exception as exc:
                failed_regions, failure_messages = self._unregister_regions(registered_regions)
                self._registered_regions = failed_regions
                self.is_register_buffer = bool(failed_regions)
                if failure_messages:
                    raise RuntimeError(
                        "Mooncake memory registration failed and rollback failed "
                        "for regions: " + "; ".join(failure_messages)
                    ) from exc
                raise
            self._registered_regions = registered_regions
            self.is_register_buffer = True
            self._registration_generation += 1
            self._registration_complete = True
            self._target_lifetime = AllocationLifetime("Mooncake TE registration")

    def registration_snapshot(self) -> RegistrationSnapshot:
        """Describe only a completely successful, still-live registration."""
        with self.register_buffer_lock:
            if not self._registration_complete or not self.is_register_buffer or not self._registered_regions:
                raise RuntimeError("Target catalog requires a complete live TE registration")
            return RegistrationSnapshot(
                self._registration_instance,
                self._registration_generation,
                tuple(self._registered_regions),
                self.transport_endpoint(),
            )

    def retain_registration(self, snapshot: RegistrationSnapshot) -> AllocationPin:
        with self.register_buffer_lock:
            current = RegistrationSnapshot(
                self._registration_instance,
                self._registration_generation,
                tuple(self._registered_regions),
                self.transport_endpoint(),
            )
            if not self._registration_complete or current != snapshot:
                raise RuntimeError("Target registration changed or is no longer live")
            return self._target_lifetime.retain()

    def unregister_buffer(self) -> None:
        """Unregister only after all local target owners release their pins."""
        with self.register_buffer_lock:
            if not self._registered_regions:
                self.is_register_buffer = False
                self._registration_complete = False
                return
            assert self.transfer_engine is not None, "Transfer engine must be initialized"

            def destroy() -> None:
                failed_regions, messages = self._unregister_regions(self._registered_regions)
                self._registered_regions = failed_regions
                self.is_register_buffer = bool(failed_regions)
                self._registration_complete = False
                if messages:
                    raise RuntimeError("Mooncake memory unregistration failed for regions: " + "; ".join(messages))

            self._target_lifetime.close(destroy)


global_te = GlobalTE()
