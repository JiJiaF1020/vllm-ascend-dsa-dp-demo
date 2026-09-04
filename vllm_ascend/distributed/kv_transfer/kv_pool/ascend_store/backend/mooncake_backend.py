# Standard
import functools
import json
import os
import threading
from dataclasses import dataclass
from typing import Any

import regex as re
import torch

# Third Party
from mooncake.store import ReplicateConfig  # type: ignore
from vllm.config import ParallelConfig
from vllm.distributed.parallel_state import get_world_group
from vllm.logger import logger
from vllm.utils.network_utils import get_ip

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.backend import Backend
from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import global_te
from vllm_ascend.distributed.parallel_state import get_global_rank

DEFAULT_GLOBAL_SEGMENT_SIZE = 1073741824  # 1.0 GiB
DEFAULT_LOCAL_BUFFER_SIZE = 1073741824  # 1.0 GiB
TCP_SCHEDULER_SEGMENT_PORT_OFFSET = 10000
MAX_TCP_SEGMENT_PORT_BASE = 45000


def _get_mooncake_client_rank(
    parallel_config: ParallelConfig,
) -> int:
    try:
        return int(get_global_rank(parallel_config))
    except Exception as exc:
        if int(getattr(parallel_config, "data_parallel_size", 1)) > 1:
            raise RuntimeError("Mooncake DP worker identity requires an initialized global worker group") from exc
        for name in ("LOCAL_RANK", "RANK"):
            value = os.getenv(name)
            if value is not None:
                return int(value)
        try:
            return int(torch.npu.current_device())
        except Exception:
            return 0


@functools.lru_cache(maxsize=1)
def _mooncake_setup_supports_ssd_offload() -> bool:
    """True when installed Mooncake exposes SSD kwargs on setup() (v0.3.11+)."""
    from mooncake.store import MooncakeDistributedStore  # type: ignore

    setup = MooncakeDistributedStore.setup
    try:
        import inspect

        sig = inspect.signature(setup)
        return "enable_ssd_offload" in sig.parameters
    except (TypeError, ValueError):
        # pybind11 overloaded bindings often reject inspect.signature
        doc = setup.__doc__ or ""
        return "enable_ssd_offload" in doc


def _ssd_setup_kwargs(config: "MooncakeStoreConfig") -> dict[str, object]:
    """Keyword args for store.setup(); empty on old Mooncake or when SSD is off."""
    if not config.enable_ssd_offload:
        return {}
    if not _mooncake_setup_supports_ssd_offload():
        raise RuntimeError(
            "mooncake.json has enable_ssd_offload=true, but the installed "
            "Mooncake does not support enable_ssd_offload/ssd_offload_path in "
            "MooncakeDistributedStore.setup(). Upgrade Mooncake to v0.3.11 or "
            "later (see Mooncake ssd-offload.md Step 3A), or set "
            "enable_ssd_offload to false."
        )
    return {
        "enable_ssd_offload": config.enable_ssd_offload,
        "ssd_offload_path": config.ssd_offload_path,
    }


class MooncakeBackend(Backend):
    def __init__(
        self,
        parallel_config: ParallelConfig,
        lazy_init: bool = False,
        contribute_memory: bool = True,
        config: "MooncakeStoreConfig | None" = None,
    ):
        self.parallel_config = parallel_config
        self.config = config or MooncakeStoreConfig.load_from_env()

        self.store: Any | None = None
        self.local_seg: str | None = None
        self._store_registered_buffers: dict[int, int] = {}
        self._store_owns_transfer_engine = self.config.protocol == "tcp"
        self._use_fabric_mem = (
            self.config.protocol == "ascend" and os.getenv("ASCEND_ENABLE_USE_FABRIC_MEM", "0") == "1"
        )
        self._lazy_init = lazy_init and self._use_fabric_mem
        self._contribute_memory = contribute_memory
        self._store_initialized = False
        self._store_init_lock = threading.Lock()

        if not self._lazy_init:
            self.store = self._setup_store()
            self._store_initialized = True

    def ensure_initialized(self):
        if self._store_initialized:
            return

        with self._store_init_lock:
            if self._store_initialized:
                return

            logger.info("Initializing Mooncake store. metadata_server=%s", self.config.metadata_server)
            self.store = self._setup_store()
            self._store_initialized = True

    def _setup_store(self):
        try:
            from mooncake.store import MooncakeDistributedStore  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "  # noqa: E501
                "to run vLLM with MooncakeConnector."
            ) from e

        store = MooncakeDistributedStore()
        local_hostname = get_ip()
        ssd_kwargs = _ssd_setup_kwargs(self.config)
        # Scheduler-only clients (contribute_memory=False) do not contribute
        # KV cache memory and therefore do not need SSD offload. Passing
        # enable_ssd_offload=True for them would cause Mooncake to register
        # an extra active client on the master, inflating both the client
        # count and the reported SSD storage usage.
        if ssd_kwargs and not self._contribute_memory:
            ssd_kwargs = {}
        # Each rank that contributes memory to the pool uses its own SSD
        # directory to avoid bucket file collisions. Key by the globally unique
        # rank so that DP/TP/PP/CP replicas never share a directory (dense and
        # MoE alike); only ranks that contribute memory need an offload dir.
        if ssd_kwargs and ssd_kwargs.get("ssd_offload_path"):
            global_rank = get_global_rank(self.parallel_config)
            rank_path = os.path.join(str(ssd_kwargs["ssd_offload_path"]), f"rank_{global_rank}")
            try:
                os.makedirs(rank_path, exist_ok=True)
            except OSError as e:
                raise RuntimeError(f"Failed to create per-rank SSD offload directory: {rank_path!r} ({e})")
            ssd_kwargs["ssd_offload_path"] = rank_path
        # TCP follows Mooncake/LMCache's Store-owned engine path. Ascend keeps
        # the existing external GlobalTE path used by the NPU KV pool.
        if self._store_owns_transfer_engine:
            # CPU-only schedulers have no worker process group/device identity.
            client_rank = (
                _get_mooncake_client_rank(self.parallel_config)
                if self._contribute_memory
                else int(self.parallel_config.data_parallel_rank)
            )
            if not 0 <= client_rank < TCP_SCHEDULER_SEGMENT_PORT_OFFSET:
                raise ValueError(f"Mooncake TCP client rank is outside its endpoint range: {client_rank}")
            segment_rank = client_rank
            role_offset = 0 if self._contribute_memory else TCP_SCHEDULER_SEGMENT_PORT_OFFSET
            tcp_segment_port = self.config.tcp_segment_port_base + segment_rank + role_offset
            self.local_seg = f"{local_hostname}:{tcp_segment_port}"
            ret = store.setup(
                local_hostname=self.local_seg,
                metadata_server=self.config.metadata_server,
                global_segment_size=self.config.global_segment_size if self._contribute_memory else 0,
                local_buffer_size=self.config.local_buffer_size if self._contribute_memory else 0,
                protocol=self.config.protocol,
                rdma_devices="",
                master_server_addr=self.config.master_server_address,
                **ssd_kwargs,
            )
        # ASCEND_ENABLE_USE_FABRIC_MEM: Enable unified memory address direct transmission scheme
        # and only can be used for 800 I/T A3 series.
        # Required supporting hardware versions are as follows:
        elif not self._use_fabric_mem:
            transfer_engine = global_te.get_transfer_engine(local_hostname, device_name=None)
            self.local_seg = local_hostname + ":" + str(transfer_engine.get_rpc_port())
            ret = store.setup(
                local_hostname=self.local_seg,
                metadata_server=self.config.metadata_server,
                global_segment_size=self.config.global_segment_size if self._contribute_memory else 0,
                local_buffer_size=self.config.local_buffer_size if self._contribute_memory else 0,
                protocol=self.config.protocol,
                rdma_devices=self.config.device_name,
                master_server_addr=self.config.master_server_address,
                engine=transfer_engine.get_engine(),
                **ssd_kwargs,
            )
        else:
            self.local_seg = local_hostname
            ret = store.setup(
                local_hostname=self.local_seg,
                metadata_server=self.config.metadata_server,
                global_segment_size=self.config.global_segment_size if self._contribute_memory else 0,
                local_buffer_size=0,
                protocol=self.config.protocol,
                rdma_devices=self.config.device_name,
                master_server_addr=self.config.master_server_address,
                **ssd_kwargs,
            )

        if ret != 0:
            msg = "Initialize mooncake failed."
            logger.error(
                "Initialize mooncake failed. ret=%d, metadata_server=%s. Check mooncake config and network.",
                ret,
                self.config.metadata_server,
            )
            raise RuntimeError(msg)
        if self._store_owns_transfer_engine:
            self.local_seg = str(store.get_hostname())
            logger.info(
                "Mooncake Store-owned TCP engine initialized: segment=%s",
                self.local_seg,
            )
        if ssd_kwargs:
            logger.info(
                "Mooncake SSD offload enabled (Mode A): path=%s",
                self.config.ssd_offload_path,
            )
        return store

    @classmethod
    def create_scheduler_client(cls, parallel_config: ParallelConfig):
        torch.npu.set_device(0)
        return cls(parallel_config, contribute_memory=False)

    def set_device(self):
        local_rank = get_world_group().local_rank
        device = torch.device(f"npu:{local_rank}")
        torch.npu.set_device(device)

    def register_buffer(self, ptrs: list[int], lengths: list[int]):
        if len(ptrs) != len(lengths):
            raise ValueError("Mooncake buffer pointer/size counts do not match")
        if self._store_owns_transfer_engine:
            self.ensure_initialized()
            assert self.store is not None
            newly_registered: list[int] = []
            try:
                for ptr, length in zip(ptrs, lengths):
                    ptr = int(ptr)
                    length = int(length)
                    if ptr <= 0 or length <= 0:
                        raise ValueError(f"Mooncake buffer range is invalid: ptr={ptr}, size={length}")
                    previous = self._store_registered_buffers.get(ptr)
                    if previous is not None:
                        if previous != length:
                            raise ValueError(
                                "Mooncake buffer was already registered with "
                                f"a different size: ptr=0x{ptr:x}, "
                                f"existing={previous}, requested={length}"
                            )
                        continue
                    end = ptr + length
                    for registered_ptr, registered_size in self._store_registered_buffers.items():
                        registered_end = registered_ptr + registered_size
                        if ptr < registered_end and registered_ptr < end:
                            raise ValueError(
                                "Mooncake incremental registration contains "
                                "an overlapping non-identical range: "
                                f"new=[0x{ptr:x},0x{end:x}), existing=["
                                f"0x{registered_ptr:x},0x{registered_end:x})"
                            )
                    result = int(self.store.register_buffer(ptr, length))
                    if result != 0:
                        raise RuntimeError(
                            f"Mooncake Store memory registration failed: ptr=0x{ptr:x}, size={length}, ret={result}"
                        )
                    self._store_registered_buffers[ptr] = length
                    newly_registered.append(ptr)
            except Exception:
                for ptr in reversed(newly_registered):
                    try:
                        rollback_result = int(self.store.unregister_buffer(ptr))
                        if rollback_result != 0:
                            logger.error(
                                "Mooncake Store registration rollback failed: ptr=0x%x ret=%d",
                                ptr,
                                rollback_result,
                            )
                    except Exception:
                        logger.exception(
                            "Mooncake Store registration rollback raised: ptr=0x%x",
                            ptr,
                        )
                    finally:
                        self._store_registered_buffers.pop(ptr, None)
                raise
            return
        if not self._use_fabric_mem:
            local_hostname = get_ip()
            global_te.get_transfer_engine(local_hostname, device_name=None)
            global_te.register_buffer(ptrs, lengths)

    def unregister_buffer(self, ptrs: list[int]):
        if self._store_owns_transfer_engine:
            assert self.store is not None
            for ptr in reversed(ptrs):
                ptr = int(ptr)
                if ptr not in self._store_registered_buffers:
                    continue
                result = int(self.store.unregister_buffer(ptr))
                if result != 0:
                    raise RuntimeError(f"Mooncake Store memory unregistration failed: ptr=0x{ptr:x}, ret={result}")
                self._store_registered_buffers.pop(ptr)
            return
        if not self._use_fabric_mem:
            global_te.unregister_buffer(ptrs)

    def exists(self, keys: list[str]) -> list[int]:
        if self._lazy_init and not self._store_initialized:
            logger.debug(
                "MooncakeBackend.exists called before store initialization; treating %d keys as missing.",
                len(keys),
            )
            return [0] * len(keys)
        assert self.store is not None
        return self.store.batch_is_exist(keys)

    def put(self, keys: list[str], addrs: list[list[int]], sizes: list[list[int]]):
        self.ensure_initialized()
        assert self.store is not None
        try:
            config = ReplicateConfig()
            if self.config.preferred_segment:
                config.preferred_segment = self.local_seg
            config.prefer_alloc_in_same_node = (
                self.config.prefer_alloc_in_same_node if self.config.protocol == "ascend" else False
            )
            res = self.store.batch_put_from_multi_buffers(keys, addrs, sizes, config)
            res_list = [int(value) for value in res]
            failed_codes = [value for value in res_list if value < 0]
            failed_count = len(failed_codes)
            if failed_count:
                error_codes = sorted(set(failed_codes))
                logger.error(
                    "Failed to put %d keys out of %d. error_codes=%s. Check memory and store capacity.",
                    failed_count,
                    len(keys),
                    error_codes,
                )
                logger.debug("Failed to put key details. keys=%s, result=%s", keys, res_list)
                if self._lazy_init:
                    logger.warning("First DSV4(compress) request failure is expected. This is normal behavior.")
            return res_list
        except Exception as e:
            logger.error(
                "Failed to put %d keys out of %d. type=%s, error=%s. Check store state and memory.",
                len(keys),
                len(keys),
                type(e).__name__,
                e,
            )
            logger.debug("Failed to put key details. keys=%s", keys)
            if self._lazy_init:
                logger.warning("First DSV4(compress) request failure is expected. This is normal behavior.")
            return None

    def close(self) -> None:
        try:
            if self.store is not None and self._store_registered_buffers:
                self.unregister_buffer(list(self._store_registered_buffers))
        finally:
            if self.store is not None:
                self.store.close()
                self.store = None
            self._store_initialized = False

    def get_with_sizes(
        self,
        keys: list[str],
        addrs: list[list[int]],
        sizes: list[list[int]],
    ):
        """Return Mooncake's per-key byte counts without normalization."""

        if self._lazy_init and not self._store_initialized:
            logger.error(
                "Failed to get %d keys out of %d. Store is not initialized; "
                "call put() first to trigger initialization.",
                len(keys),
                len(keys),
            )
            logger.debug("Failed to get key details. keys=%s", keys)
            return
        assert self.store is not None
        logger.debug(
            "MooncakeBackend.get enter keys=%d sample_keys=%s",
            len(keys),
            keys[:3],
        )
        try:
            res = self.store.batch_get_into_multi_buffers(keys, addrs, sizes)
            res_list = list(res)
            failed_codes = [int(value) for value in res_list if value < 0]
            failed_count = len(failed_codes)
            error_codes = sorted(set(failed_codes))
            if failed_count:
                logger.error(
                    "Failed to get %d keys out of %d. error_codes=%s. Check key existence and memory state.",
                    failed_count,
                    len(keys),
                    error_codes,
                )
                logger.debug("Failed to get key details. keys=%s, result=%s", keys, res_list)
            return res_list
        except Exception as e:
            logger.error(
                "Failed to get %d keys out of %d. type=%s, error=%s. Check store state and network.",
                len(keys),
                len(keys),
                type(e).__name__,
                e,
            )
            logger.debug("Failed to get key details. keys=%s", keys)
            return None

    def get(self, keys: list[str], addrs: list[list[int]], sizes: list[list[int]]):
        res_list = self.get_with_sizes(keys, addrs, sizes)
        if res_list is None:
            return None
        return [0 if int(value) > 0 else int(value) for value in res_list]


@dataclass
class MooncakeStoreConfig:
    metadata_server: str
    global_segment_size: int | str
    local_buffer_size: int
    protocol: str
    device_name: str
    master_server_address: str
    preferred_segment: bool
    prefer_alloc_in_same_node: bool
    tcp_segment_port_base: int = 45000
    enable_ssd_offload: bool = False
    ssd_offload_path: str = ""

    def __post_init__(self) -> None:
        if self.protocol not in {"ascend", "tcp"}:
            raise ValueError(f"Mooncake protocol must be 'ascend' or 'tcp', got {self.protocol!r}")
        if not 1 <= self.tcp_segment_port_base <= MAX_TCP_SEGMENT_PORT_BASE:
            raise ValueError(
                f"tcp_segment_port_base must be in [1, {MAX_TCP_SEGMENT_PORT_BASE}], got {self.tcp_segment_port_base}"
            )
        if not self.enable_ssd_offload:
            return
        if not self.ssd_offload_path:
            raise ValueError(
                "enable_ssd_offload is true but ssd_offload_path is empty. Set ssd_offload_path in mooncake.json."
            )
        if not os.path.isabs(self.ssd_offload_path):
            raise ValueError(f"ssd_offload_path must be an absolute path, got: {self.ssd_offload_path!r}")

    @staticmethod
    def from_file(file_path: str) -> "MooncakeStoreConfig":
        with open(file_path) as file:
            config = json.load(file)
        master_server_address = os.getenv("MOONCAKE_MASTER", None)
        global_segment_size_env = os.getenv("MOONCAKE_GLOBAL_SEGMENT_SIZE", None)
        return MooncakeStoreConfig(
            metadata_server=config.get("metadata_server"),
            global_segment_size=_parse_global_segment_size(
                global_segment_size_env
                if global_segment_size_env is not None
                else config.get("global_segment_size", DEFAULT_GLOBAL_SEGMENT_SIZE)
            ),
            local_buffer_size=_parse_global_segment_size(config.get("local_buffer_size", DEFAULT_LOCAL_BUFFER_SIZE)),
            protocol=config.get("protocol", "ascend"),
            device_name=config.get("device_name", ""),
            master_server_address=master_server_address
            if master_server_address is not None
            else config.get("master_server_address"),
            preferred_segment=config.get("preferred_segment", False),
            prefer_alloc_in_same_node=config.get("prefer_alloc_in_same_node", True),
            tcp_segment_port_base=int(
                os.getenv(
                    "MOONCAKE_TCP_SEGMENT_PORT_BASE",
                    config.get("tcp_segment_port_base", 45000),
                )
            ),
            enable_ssd_offload=bool(config.get("enable_ssd_offload", False)),
            ssd_offload_path=config.get("ssd_offload_path", ""),
        )

    @staticmethod
    def load_from_env() -> "MooncakeStoreConfig":
        config_path = os.getenv("MOONCAKE_CONFIG_PATH")
        if not config_path:
            raise ValueError("The environment variable 'MOONCAKE_CONFIG_PATH' is not set.")
        return MooncakeStoreConfig.from_file(config_path)


def _parse_global_segment_size(value) -> int:
    """
    Parse storage size strings with support for units: GB, MB, KB, B

    Args:
        value: Input value (int, str, or other convertible types)

    Returns:
        int: Size in bytes

    Raises:
        ValueError: For invalid format, missing number, or negative values
        TypeError: For unsupported input types
    """

    if isinstance(value, int):
        return value
    elif not isinstance(value, str):
        try:
            return int(value)
        except (TypeError, ValueError) as e:
            raise TypeError(f"Unsupported type for global_segment_size: {type(value)}") from e

    cleaned_input = value.strip().lower()
    if not cleaned_input:
        raise ValueError("global segment size cannot be empty.")

    UNIT_MULTIPLIERS = {
        "gb": 1024**3,  # 1 GB = 1024^3 bytes
        "mb": 1024**2,  # 1 MB = 1024^2 bytes
        "kb": 1024,  # 1 KB = 1024 bytes
        "b": 1,  # 1 B = 1 byte
    }
    pattern = r"^\s*([\d.]+)\s*(gb|mb|kb|b)?\s*$"
    match = re.match(pattern, cleaned_input)

    if not match:
        raise ValueError(f"Invalid format: '{value}'")

    number_str = match.group(1)
    unit = match.group(2) or "b"

    multiplier = UNIT_MULTIPLIERS[unit]
    return _convert_to_bytes(number_str, multiplier, value)


def _convert_to_bytes(number_str: str, multiplier: int, original_input: str) -> int:
    """
    Convert numeric string to byte count

    Args:
        number_str: Numeric portion of input
        multiplier: Unit conversion factor
        original_input: Original input string (for error messages)

    Returns:
        int: Byte count

    Raises:
        ValueError: For invalid numbers or negative results
    """
    try:
        numeric_value = float(number_str)
    except ValueError:
        raise ValueError(f"Invalid numeric value '{number_str}' in: '{original_input}'")
    # Calculate byte count
    try:
        byte_count = int(numeric_value * multiplier)
    except OverflowError:
        raise ValueError(f"Storage size too large: '{original_input}'")
    return byte_count
