#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate per-device HiXLEP LocalCommRes files used by A5 workers."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _parse_devices(value: str) -> list[int]:
    devices = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not devices:
        raise ValueError("visible device list is empty")
    if len(devices) != len(set(devices)):
        raise ValueError(f"visible device list contains duplicates: {devices}")
    return devices


def _summarize_endpoint(endpoint: object) -> tuple[str, str]:
    if not isinstance(endpoint, dict):
        return "<invalid>", "<invalid>"
    return str(endpoint.get("protocol", "<missing>")), str(
        endpoint.get("placement", "<missing>")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=Path("/etc/hixlep"))
    parser.add_argument("--visible-devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument(
        "--require-placement",
        action="append",
        default=[],
        help="Require every per-device file to expose this endpoint placement.",
    )
    args = parser.parse_args()

    config_dir = args.path.expanduser().resolve()
    devices = _parse_devices(args.visible_devices)
    failures: list[str] = []
    digests: dict[str, list[int]] = {}

    print(
        f"[hixlep-check] path={config_dir} "
        f"physical_devices={devices}"
    )
    for device_id in devices:
        config_file = config_dir / f"ub_endpoint_npu_{device_id}.json"
        try:
            raw = config_file.read_bytes()
        except OSError as exc:
            failures.append(f"device={device_id} cannot read {config_file}: {exc}")
            continue

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            failures.append(f"device={device_id} invalid JSON {config_file}: {exc}")
            continue
        if not isinstance(payload, dict):
            failures.append(f"device={device_id} root must be a JSON object")
            continue

        endpoints = payload.get("endpoint_list")
        if not isinstance(endpoints, list) or not endpoints:
            failures.append(
                f"device={device_id} endpoint_list must be a non-empty list"
            )
            continue

        protocols: set[str] = set()
        placements: set[str] = set()
        for endpoint in endpoints:
            protocol, placement = _summarize_endpoint(endpoint)
            protocols.add(protocol)
            placements.add(placement)
        missing_placements = sorted(set(args.require_placement) - placements)
        if missing_placements:
            failures.append(
                f"device={device_id} missing endpoint placements "
                f"{missing_placements}; available={sorted(placements)}"
            )
        digest = hashlib.sha256(raw).hexdigest()[:12]
        digests.setdefault(digest, []).append(device_id)
        print(
            f"[hixlep-check] device={device_id} file={config_file.name} "
            f"version={payload.get('version')} endpoints={len(endpoints)} "
            f"protocols={sorted(protocols)} placements={sorted(placements)} "
            f"sha256={digest}"
        )

    for digest, digest_devices in digests.items():
        if len(digest_devices) > 1:
            print(
                "[hixlep-check] warning: identical per-device files "
                f"devices={digest_devices} sha256={digest}"
            )

    if failures:
        for failure in failures:
            print(f"[hixlep-check] ERROR {failure}")
        print(
            "[hixlep-check] FAILED. Generate LocalCommRes files from this "
            "host's real topology with all required placements before "
            "starting TP workers."
        )
        return 1

    print("[hixlep-check] PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
