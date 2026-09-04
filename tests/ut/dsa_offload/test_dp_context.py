# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pickle
from dataclasses import replace

import pytest

import vllm_ascend.dsa_offload.dp_context as dp_context
from vllm_ascend.dsa_offload.dp_context import (
    coordinate_dsa_node_memory_budget,
    install_dsa_engine_dp_context,
    require_dsa_shared_l3_hash_seed,
    reset_dsa_engine_dp_context,
)

BOOT_A = "11111111-1111-4111-8111-111111111111"
BOOT_B = "22222222-2222-4222-8222-222222222222"


@pytest.fixture(autouse=True)
def _reset_context(monkeypatch, tmp_path):
    reset_dsa_engine_dp_context()
    path = tmp_path / "boot_id"
    path.write_text(BOOT_A + "\n", encoding="ascii")
    monkeypatch.setattr(dp_context, "_HOST_BOOT_ID_PATH", path)
    monkeypatch.setattr(dp_context.socket, "gethostname", lambda: "node-a")
    yield
    reset_dsa_engine_dp_context()


def test_single_dp_uses_the_observed_host_budget(monkeypatch) -> None:
    monkeypatch.setattr(dp_context.socket, "gethostname", lambda: "node-a")
    monkeypatch.setattr(dp_context, "_read_host_boot_id", lambda: pytest.fail("DP1 must not require boot ID"))

    budget = coordinate_dsa_node_memory_budget(
        data_parallel_size=1,
        data_parallel_rank=0,
        observed_available_bytes=1024,
        l1_indexer_blocks=8,
        complete_bundle_bytes=4,
        tensor_parallel_size=2,
        compatibility_signature="sig-a",
    )

    assert budget.data_parallel_size == 1
    assert budget.observed_available_bytes_by_dp == (1024,)
    assert budget.node_available_bytes == 1024
    assert budget.per_dp_quota_bytes == 1024
    assert budget.minimum_per_dp_quota_bytes == 72


def test_single_node_dp_uses_min_snapshot_and_equal_byte_quota(
    monkeypatch,
) -> None:
    process_group = object()
    install_dsa_engine_dp_context(
        process_group=process_group,
        rank=1,
        world_size=2,
        local_rank=1,
    )
    monkeypatch.setattr(dp_context.socket, "gethostname", lambda: "node-a")

    def fake_all_gather_object(output, _sample, *, group) -> None:
        assert group is process_group
        output[:] = [
            replace(_sample, rank=0, available_bytes=1000, l1_indexer_blocks=8),
            replace(_sample, rank=1, available_bytes=900, l1_indexer_blocks=10),
        ]

    monkeypatch.setattr(
        dp_context.dist,
        "all_gather_object",
        fake_all_gather_object,
    )

    budget = coordinate_dsa_node_memory_budget(
        data_parallel_size=2,
        data_parallel_rank=1,
        observed_available_bytes=900,
        l1_indexer_blocks=10,
        complete_bundle_bytes=4,
        tensor_parallel_size=2,
        compatibility_signature="sig-a",
    )

    assert budget.data_parallel_rank == 1
    assert budget.observed_available_bytes_by_dp == (1000, 900)
    assert budget.node_available_bytes == 900
    assert budget.per_dp_quota_bytes == 450
    assert budget.l1_indexer_blocks_by_dp == (8, 10)
    assert budget.minimum_per_dp_quota_bytes == 88


def test_dp_budget_rejects_multiple_host_scopes(monkeypatch) -> None:
    install_dsa_engine_dp_context(
        process_group=object(),
        rank=0,
        world_size=2,
        local_rank=0,
    )

    def fake_all_gather_object(output, _sample, *, group) -> None:
        output[:] = [
            _sample,
            replace(_sample, rank=1, boot_id=BOOT_B),
        ]

    monkeypatch.setattr(
        dp_context.dist,
        "all_gather_object",
        fake_all_gather_object,
    )

    with pytest.raises(NotImplementedError, match="one shared host scope"):
        coordinate_dsa_node_memory_budget(
            data_parallel_size=2,
            data_parallel_rank=0,
            observed_available_bytes=1000,
            l1_indexer_blocks=8,
            complete_bundle_bytes=4,
            tensor_parallel_size=2,
            compatibility_signature="sig-a",
        )


def test_dp_budget_requires_the_engine_core_group() -> None:
    with pytest.raises(RuntimeError, match="no EngineCore DP context"):
        coordinate_dsa_node_memory_budget(
            data_parallel_size=2,
            data_parallel_rank=0,
            observed_available_bytes=1000,
            l1_indexer_blocks=8,
            complete_bundle_bytes=4,
            tensor_parallel_size=2,
            compatibility_signature="sig-a",
        )


def test_dp_budget_rejects_quota_smaller_than_largest_l1_mirror(
    monkeypatch,
) -> None:
    install_dsa_engine_dp_context(
        process_group=object(),
        rank=0,
        world_size=2,
        local_rank=0,
    )

    def fake_all_gather_object(output, _sample, *, group) -> None:
        output[:] = [
            _sample,
            replace(_sample, rank=1, l1_indexer_blocks=12),
        ]

    monkeypatch.setattr(
        dp_context.dist,
        "all_gather_object",
        fake_all_gather_object,
    )

    with pytest.raises(RuntimeError, match="largest local L1 Indexer"):
        coordinate_dsa_node_memory_budget(
            data_parallel_size=2,
            data_parallel_rank=0,
            observed_available_bytes=100,
            l1_indexer_blocks=8,
            complete_bundle_bytes=4,
            tensor_parallel_size=2,
            compatibility_signature="sig-a",
        )


def test_dp_budget_rejects_incompatible_signatures(monkeypatch) -> None:
    install_dsa_engine_dp_context(
        process_group=object(),
        rank=0,
        world_size=2,
        local_rank=0,
    )

    def fake_all_gather_object(output, _sample, *, group) -> None:
        output[:] = [
            _sample,
            replace(_sample, rank=1, signature="sig-b"),
        ]

    monkeypatch.setattr(
        dp_context.dist,
        "all_gather_object",
        fake_all_gather_object,
    )

    with pytest.raises(RuntimeError, match="configuration signatures"):
        coordinate_dsa_node_memory_budget(
            data_parallel_size=2,
            data_parallel_rank=0,
            observed_available_bytes=1000,
            l1_indexer_blocks=8,
            complete_bundle_bytes=4,
            tensor_parallel_size=2,
            compatibility_signature="sig-a",
        )


def _coordinate(rank=0):
    return coordinate_dsa_node_memory_budget(
        data_parallel_size=2,
        data_parallel_rank=rank,
        observed_available_bytes=1000,
        l1_indexer_blocks=8,
        complete_bundle_bytes=4,
        tensor_parallel_size=2,
        compatibility_signature="sig-a",
    )


def test_same_kernel_accepts_different_hostnames_and_spmd_local_ranks(monkeypatch):
    plans = []

    def gather(output, sample, *, group):
        assert pickle.loads(pickle.dumps(sample)) == sample
        output[:] = [replace(sample, rank=0, hostname="container-a"), replace(sample, rank=1, hostname="container-b")]

    monkeypatch.setattr(dp_context.dist, "all_gather_object", gather)
    for rank, host in enumerate(("container-a", "container-b")):
        reset_dsa_engine_dp_context()
        install_dsa_engine_dp_context(process_group=object(), rank=rank, world_size=2, local_rank=0)
        monkeypatch.setattr(dp_context.socket, "gethostname", lambda value=host: value)
        plans.append(_coordinate(rank))
    assert plans[0].node_scope_id == plans[1].node_scope_id
    assert plans[0].snapshot_id == plans[1].snapshot_id
    assert plans[0].per_dp_quota_bytes == plans[1].per_dp_quota_bytes == 500


@pytest.mark.parametrize("rank", [0, 1])
def test_boot_probe_failure_is_gathered_then_rejected_by_every_dp(monkeypatch, rank):
    install_dsa_engine_dp_context(process_group=object(), rank=rank, world_size=2, local_rank=rank)

    def read_id():
        if rank == 0:
            raise PermissionError("blocked")
        return BOOT_A

    calls = []

    def gather(output, sample, *, group):
        calls.append(sample)
        output[:] = [
            replace(sample, rank=0, boot_id=None, probe_error="PermissionError: blocked"),
            replace(sample, rank=1, boot_id=BOOT_A, probe_error=None),
        ]

    monkeypatch.setattr(dp_context, "_read_host_boot_id", read_id)
    monkeypatch.setattr(dp_context.dist, "all_gather_object", gather)
    with pytest.raises(RuntimeError, match="boot-ID probe failures"):
        _coordinate(rank)
    assert len(calls) == 1
    assert (calls[0].probe_error is not None) == (rank == 0)


@pytest.mark.parametrize("bad_id", ["", "node-a", "00000000-0000-0000-0000-000000000000", "错误"])
def test_boot_id_reader_rejects_invalid_or_masked_values(bad_id):
    dp_context._HOST_BOOT_ID_PATH.write_text(bad_id, encoding="utf-8")
    with pytest.raises(ValueError):
        dp_context._read_host_boot_id()


def test_boot_id_reader_does_not_fallback_when_file_missing():
    dp_context._HOST_BOOT_ID_PATH.unlink()
    with pytest.raises(OSError):
        dp_context._read_host_boot_id()


@pytest.mark.parametrize("bad_peer", [None, (1, "node-a", 1000, 8, 4, 2, "sig-a")])
def test_budget_rejects_old_or_missing_observation_protocol(monkeypatch, bad_peer):
    install_dsa_engine_dp_context(process_group=object(), rank=0, world_size=2, local_rank=0)

    def gather(output, sample, *, group):
        output[:] = [sample, bad_peer]

    monkeypatch.setattr(dp_context.dist, "all_gather_object", gather)
    with pytest.raises(RuntimeError, match="invalid/empty observation"):
        _coordinate()


@pytest.mark.parametrize("problem", ["duplicate_rank", "changed_local"])
def test_budget_rejects_wrong_participants_or_changed_local_sample(monkeypatch, problem):
    install_dsa_engine_dp_context(process_group=object(), rank=0, world_size=2, local_rank=0)

    def gather(output, sample, *, group):
        if problem == "duplicate_rank":
            output[:] = [sample, sample]
        else:
            output[:] = [replace(sample, available_bytes=999), replace(sample, rank=1)]

    monkeypatch.setattr(dp_context.dist, "all_gather_object", gather)
    with pytest.raises(RuntimeError, match="invalid ranks|local observation"):
        _coordinate()


@pytest.mark.parametrize("seed", [None, "", "random", "-1", " 1", "1.0", "4294967296", "١", "9" * 5000])
def test_shared_l3_requires_an_explicit_numeric_seed(monkeypatch, seed):
    if seed is None:
        monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    else:
        monkeypatch.setenv("PYTHONHASHSEED", seed)
    with pytest.raises(ValueError, match="PYTHONHASHSEED"):
        require_dsa_shared_l3_hash_seed()


@pytest.mark.parametrize("seed", ["0", "114514", "4294967295", "0007"])
def test_shared_l3_preserves_raw_seed_and_stable_vllm_root(monkeypatch, seed):
    from vllm.utils.hashing import sha256
    from vllm.v1.core import kv_cache_utils

    monkeypatch.setenv("PYTHONHASHSEED", seed)
    monkeypatch.setattr(kv_cache_utils, "NONE_HASH", b"previous-root", raising=False)
    hashes = []
    for _ in range(2):
        assert require_dsa_shared_l3_hash_seed() == seed
        kv_cache_utils.init_none_hash(sha256)
        hashes.append(kv_cache_utils.hash_block_tokens(sha256, None, list(range(128))))
    assert hashes[0] == hashes[1]
