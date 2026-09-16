# LIM-MTP SIMT / full-visible TopK2048 migration

Source: `ops_lidu_scattercopy_sfa_a5`, commit
`6732a649a36e4829994aa6cbdbd7ad6616333180`.

The host C++ and kernel tree match the standalone operator, except for the
retained SPDX header in `vllm_a5_li_manage_c8_fast_union_vf.h`. Framework CMake
and `_C_ascend` registration remain framework-owned.

The MTP ABI now uses `offloaded_lens`, `topk_slots`, `topk_index`; paired
outputs are caller-owned `[T,1,2048]`. Each query selects TopK over its complete
causal KV domain. Selected tail tokens are hits at `C + token % 256`, never
union-copy sources or reverse-table entries. Only the durable prefix is managed.
Both steady paths seed eviction from `hash(final_seq_lengths_kv[b])`.

This integration upgrades the MTP3-configured target path, retaining ragged
R=1..4 support, PAD/DENSE and first-fill/steady. Non-MTP, KSC, two parity tail
allocation/dump/write logic, and resident span C+256 are unchanged.
MTP QSFA uses sparse_mode=0; the L3 attention-semantics fingerprint is versioned.

Rebuild **both** CANN operators and the Torch extension, restart workers and
recapture graphs. Host-only checks do not validate A5 synchronization or
full-model output. Hardware acceptance:

```bash
python examples/dsa_demo/run_a5_mtp_operator_tests.py
```

The native checks compare against independent full-visible LI TopK, logical KV
attention golden, real swapped-memory copies, and fixed-address graph replay.
Then run MTP3 inference with split/fused, eager/graph, and prefix/L3 cold/warm
paths. Single-operator test results are not framework hardware acceptance.
