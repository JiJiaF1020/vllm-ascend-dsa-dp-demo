# LIM-MTP SIMT migration

Source: `ops_lidu_scattercopy_sfa_a5`, commit
`2f80de6fb5df1d941164ad5cec7f144f5025e4ce`
(includes SIMT `0fc1c89` and victim scan rotation `eb80083`).

Only LIM-MTP sources are migrated. Framework CMake/registration, seven-output
2304 ABI, KSC guards, scheduler, prefix/L3 lifecycle and Python calls are retained.
The existing SPDX header in `vllm_a5_li_manage_c8_fast_union_vf.h` is retained.

Both fast union and the mixed cold/steady manager seed eviction scans from the
current `final_seq_lengths_kv[b]`, never physical pool row. Candidate length still
bounds the scanned durable prefix; tails do not enter that domain. A new length
may rotate the start, but modulo collisions are allowed. This is a policy change
from the previous candidate-only seed, not a promise of identical old/new slots
or full-model token outputs. Request-local SIMT metadata is refreshed per call.

SIMT migration includes the 216 KiB user-UB reservation and cold QLI 8K chunk.
Rebuild the CANN operators and framework extension on the target Ascend machine.

Host checks:

```bash
python -m unittest discover -s tests/ut/dsa_offload -p test_li_manage_c8_simt.py -v
```

These cover unsigned victim unpack, update/index bounds, batched repair, UB
arithmetic and scan rotation/coverage. They cannot validate A5 synchronization.
Before accepting on hardware, run the existing A5 tests in
`tests/e2e/nightly/single_node/ops/singlecard_ops/test_dsa_a5_c8_ops.py`, especially
`test_packed_c8_mtp_pool_row_relocation_diagnostic` and
`test_packed_c8_mtp_lim_mixed_2304_graph_replay`, plus prefix/L3 cold/warm/reentry
inference comparisons. Do not relax assertions to accept unexplained differences.
