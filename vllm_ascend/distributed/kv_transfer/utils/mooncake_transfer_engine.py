import threading


class GlobalTE:
    def __init__(self):
        self.transfer_engine = None
        self.registered_buffers: dict[int, int] = {}
        self.transfer_engine_lock = threading.Lock()
        self.register_buffer_lock = threading.Lock()

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
        return self.transfer_engine

    def register_buffer(self, ptrs: list[int], sizes: list[int]):
        with self.register_buffer_lock:
            assert self.transfer_engine is not None, "Transfer engine must be initialized"
            if len(ptrs) != len(sizes):
                raise ValueError(
                    "Mooncake buffer pointer/size counts do not match"
                )
            newly_registered: list[int] = []
            for ptr, size in zip(ptrs, sizes):
                ptr = int(ptr)
                size = int(size)
                if ptr <= 0 or size <= 0:
                    raise ValueError(
                        f"Mooncake buffer range is invalid: ptr={ptr}, size={size}"
                    )
                existing_size = self.registered_buffers.get(ptr)
                if existing_size is not None:
                    if existing_size != size:
                        raise ValueError(
                            "Mooncake buffer was already registered with a "
                            f"different size: ptr=0x{ptr:x}, "
                            f"existing={existing_size}, requested={size}"
                        )
                    continue
                end = ptr + size
                for registered_ptr, registered_size in self.registered_buffers.items():
                    registered_end = registered_ptr + registered_size
                    if ptr < registered_end and registered_ptr < end:
                        raise ValueError(
                            "Mooncake incremental registration contains an "
                            "overlapping non-identical range: "
                            f"new=[0x{ptr:x},0x{end:x}), "
                            "existing=["
                            f"0x{registered_ptr:x},0x{registered_end:x})"
                        )
                ret_value = self.transfer_engine.register_memory(ptr, size)
                if ret_value != 0:
                    for registered_ptr in reversed(newly_registered):
                        self.transfer_engine.unregister_memory(registered_ptr)
                        self.registered_buffers.pop(registered_ptr, None)
                    raise RuntimeError(
                        "Mooncake memory registration failed: "
                        f"ptr=0x{ptr:x}, size={size}, ret={ret_value}"
                    )
                self.registered_buffers[ptr] = size
                newly_registered.append(ptr)

    def unregister_buffer(self, ptrs: list[int]) -> None:
        with self.register_buffer_lock:
            assert self.transfer_engine is not None, "Transfer engine must be initialized"
            for ptr in reversed(ptrs):
                ptr = int(ptr)
                if ptr not in self.registered_buffers:
                    continue
                ret_value = self.transfer_engine.unregister_memory(ptr)
                if ret_value != 0:
                    raise RuntimeError(
                        "Mooncake memory unregistration failed: "
                        f"ptr=0x{ptr:x}, ret={ret_value}"
                    )
                self.registered_buffers.pop(ptr)


global_te = GlobalTE()
