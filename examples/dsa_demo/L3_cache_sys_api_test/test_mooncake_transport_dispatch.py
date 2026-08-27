#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verify that a staged Mooncake wheel initializes the requested transport."""

from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=("tcp", "ascend"), required=True)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--expected-site", type=Path, required=True)
    args = parser.parse_args()

    if args.protocol == "ascend":
        import torch
        import torch_npu  # noqa: F401

        torch.npu.set_device(args.device_id)

    import mooncake
    import mooncake.engine
    from mooncake.engine import TransferEngine

    package_path = Path(mooncake.__file__).resolve()
    engine_path = Path(mooncake.engine.__file__).resolve()
    expected_site = args.expected_site.resolve()
    if not package_path.is_relative_to(expected_site):
        raise RuntimeError(
            f"Mooncake was not imported from staged site: "
            f"package={package_path}, expected={expected_site}"
        )

    print(
        "[mooncake-dispatch] "
        f"distribution={importlib.metadata.version('mooncake-transfer-engine-npu')} "
        f"package={package_path} engine={engine_path} protocol={args.protocol}",
        flush=True,
    )
    engine = TransferEngine()
    ret = int(
        engine.initialize(
            "127.0.0.1",
            "P2PHANDSHAKE",
            args.protocol,
            "",
        )
    )
    if ret != 0:
        raise RuntimeError(
            f"TransferEngine.initialize protocol={args.protocol!r} ret={ret}"
        )
    print(
        f"[mooncake-dispatch] PASSED protocol={args.protocol} "
        f"rpc_port={int(engine.get_rpc_port())}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
