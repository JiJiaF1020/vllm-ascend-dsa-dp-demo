#!/usr/bin/env python3
"""最小复现 MemCache device_urma 两卡建链超时。"""

import multiprocessing as mp
import os
import sys
import traceback
from pathlib import Path
from threading import BrokenBarrierError


DEVICES = (0, 1)
CONFIG = Path(__file__).parent / "baseline_e2e/configs/mmc-local.conf"


def worker(device: int, barrier) -> None:
    try:
        os.environ["ASCEND_VISIBLE_DEVICES"] = "0,1"
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0,1"
        os.environ["MMC_LOCAL_CONFIG_PATH"] = str(CONFIG.resolve())

        import torch
        import torch_npu  # noqa: F401
        from memcache_hybrid import DistributedObjectStore

        torch.npu.set_device(device)
        store = DistributedObjectStore()
        print(f"[repro] device={device} calling store.init", flush=True)
        ret = int(store.init(device, init_bm=True))
        print(f"[repro] device={device} store.init ret={ret}", flush=True)
        if ret == 0:
            try:
                barrier.wait(timeout=150.0)
            except BrokenBarrierError:
                pass
            store.close()
        else:
            barrier.abort()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0 if ret == 0 else 1)
    except Exception:
        try:
            barrier.abort()
        except Exception:
            pass
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)


if __name__ == "__main__":
    context = mp.get_context("spawn")
    barrier = context.Barrier(len(DEVICES))
    processes = [
        context.Process(target=worker, args=(device, barrier))
        for device in DEVICES
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    exitcodes = [process.exitcode for process in processes]
    print(f"[repro] child exitcodes={exitcodes}")
    raise SystemExit(0 if exitcodes == [0, 0] else 1)
