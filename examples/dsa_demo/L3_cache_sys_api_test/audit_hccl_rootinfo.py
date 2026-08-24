#!/usr/bin/env python3
"""Summarize and validate the HCCL rootinfo fields consumed by MemFabric."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_port_count(topo_path: Path) -> int | None:
    if not topo_path.is_file():
        return None
    text = topo_path.read_text(encoding="utf-8", errors="replace")
    if "Atlas 950 SuperPoD" in text:
        return 6
    if "Atlas 850" in text:
        return 8
    return None


def eid_entries(device: dict) -> list[dict]:
    entries: list[dict] = []
    for level in device.get("level_list", []):
        for entry in level.get("rank_addr_list", []):
            if entry.get("addr_type") == "EID":
                entries.append(entry)
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default="/etc/hccl_rootinfo.json")
    parser.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    args = parser.parse_args()

    rootinfo_path = Path(args.path).expanduser().resolve()
    devices = [int(item) for item in args.devices.split(",") if item.strip()]
    data = json.loads(rootinfo_path.read_text(encoding="utf-8"))
    topo_path = Path(str(data.get("topo_file_path", ""))).expanduser()
    port_count = expected_port_count(topo_path)

    print(f"rootinfo={rootinfo_path}")
    print(f"rootinfo_sha256={file_sha256(rootinfo_path)}")
    print(f"topo_file_path={topo_path}")
    print(f"topo_exists={topo_path.is_file()}")
    if topo_path.is_file():
        print(f"topo_sha256={file_sha256(topo_path)}")
    print(f"expected_eid_port_count={port_count}")

    by_device: dict[int, dict] = {}
    for item in data.get("rank_list", []):
        device_id = item.get("device_id")
        if isinstance(device_id, int):
            by_device[device_id] = item

    failed = False
    for device_id in devices:
        device = by_device.get(device_id)
        if device is None:
            print(f"device={device_id} status=MISSING")
            failed = True
            continue
        candidates = []
        for entry in eid_entries(device):
            ports = entry.get("ports", [])
            if port_count is None or len(ports) == port_count:
                candidates.append((entry.get("addr"), len(ports), ports))
        status = "OK" if len(candidates) == 1 else "INVALID"
        print(
            f"device={device_id} status={status} "
            f"candidate_count={len(candidates)} candidates={candidates}"
        )
        failed |= status != "OK"
    return 1 if failed or port_count is None else 0


if __name__ == "__main__":
    raise SystemExit(main())
