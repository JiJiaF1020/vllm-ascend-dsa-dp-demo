#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.config import CUDAGraphMode

import vllm_ascend.spec_decode.llm_base_proposer as proposer_module
from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer

# CUDAGraphMode values whose ``has_full_cudagraphs()`` is True: FULL plus the
# two composite modes that mix FULL with NONE / PIECEWISE.
FULL_CUDAGRAPH_MODES = [
    CUDAGraphMode.FULL,
    CUDAGraphMode.FULL_DECODE_ONLY,
    CUDAGraphMode.FULL_AND_PIECEWISE,
]

# Modes without a full cudagraph.
NON_FULL_CUDAGRAPH_MODES = [
    CUDAGraphMode.NONE,
    CUDAGraphMode.PIECEWISE,
]


@pytest.mark.parametrize("peer_tokens", [4, 16, 16384])
@pytest.mark.parametrize("local_reqs", [1, 4])
@pytest.mark.parametrize("dsa_enabled", [False, True])
def test_dsa_dp_mtp_idle_uses_local_rows_and_no_attention(monkeypatch, peer_tokens, local_reqs, dsa_enabled):
    proposer = object.__new__(AscendSpecDecodeBaseProposer)
    padded_tokens = max(peer_tokens, 4 * local_reqs)
    sync = Mock(return_value=(padded_tokens, torch.tensor([padded_tokens] * 2), CUDAGraphMode.NONE))
    proposer.runner = SimpleNamespace(
        _sync_metadata_across_dp=sync, max_num_reqs=4, pcp_manager=None, dsa_offload_enabled=dsa_enabled,
    )
    proposer.pcp_size = proposer.dcp_size = 1
    proposer.use_cuda_graph = True  # Runtime idle must not inherit target FULL.
    proposer.supports_mm_inputs = False
    proposer.num_speculative_tokens = 3
    proposer.extra_slots_per_request = 1
    proposer.token_indices_to_sample = torch.zeros(padded_tokens, dtype=torch.int32)
    proposer._get_positions = lambda count: torch.zeros(count, dtype=torch.int64)
    proposer._runnable = Mock()
    proposer.vllm_config = SimpleNamespace()
    context = SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.NONE, moe_layer_index=9)
    context_calls = []

    @contextmanager
    def set_context(metadata, config, **kwargs):
        context_calls.append((metadata, kwargs))
        yield context

    monkeypatch.setattr(proposer_module, "set_ascend_forward_context", set_context)
    monkeypatch.setattr(proposer_module, "get_forward_context", lambda: context)
    proposer.dummy_run(
        num_tokens=4 * local_reqs,
        num_reqs=local_reqs,
        aclgraph_runtime_mode=CUDAGraphMode.FULL,
        runtime_idle=True,
    )
    sync.assert_called_once_with(4 * local_reqs, is_draft_model=True)
    kwargs = proposer._runnable.call_args.kwargs
    assert kwargs["num_input_tokens"] == padded_tokens
    assert kwargs["batch_size"] == local_reqs
    assert kwargs["token_indices_to_sample"].shape == (local_reqs,)
    assert kwargs["multi_steps_attn_metadata"] == []
    assert context_calls[0][0] is None
    assert context_calls[0][1]["num_actual_tokens"] == 0
    assert context_calls[0][1]["aclgraph_runtime_mode"] == CUDAGraphMode.NONE
    assert context_calls[0][1]["skip_kv_transfer"] is dsa_enabled


@pytest.mark.parametrize("local_reqs", [1, 4])
def test_mtp_idle_merged_draft_keeps_all_three_model_calls(monkeypatch, local_reqs):
    proposer = object.__new__(AscendSpecDecodeBaseProposer)
    num_tokens = 16
    proposer.runner = SimpleNamespace(pcp_manager=None)
    proposer.device = torch.device("cpu")
    proposer.vllm_config = SimpleNamespace(model_config=SimpleNamespace(max_model_len=512))
    proposer.method = "mtp"
    proposer.num_speculative_tokens = 3
    proposer.parallel_drafting = False
    proposer.use_cuda_graph = False
    proposer.pass_hidden_states_to_model = True
    proposer.supports_mm_inputs = proposer.uses_mrope = False
    proposer.pcp_size = proposer.dcp_size = 1
    proposer.input_ids = torch.zeros(num_tokens, dtype=torch.int32)
    proposer.positions = torch.zeros(num_tokens, dtype=torch.int64)
    proposer.hidden_states = torch.zeros(num_tokens, 2)
    proposer.arange = torch.arange(num_tokens)
    proposer._get_positions = lambda count: proposer.positions[:count]
    proposer._set_positions = lambda count, values: proposer.positions[:count].copy_(values)
    proposer.maybe_pad_and_reduce = lambda hidden, positions: (hidden, positions)
    proposer.maybe_all_gather_and_unpad = lambda last, positions, hidden: (last, positions, hidden)
    proposer.model_returns_tuple = lambda: False
    proposer._share_mtp_indices = True
    context = SimpleNamespace(attn_metadata=None, moe_layer_index=9)
    calls = []
    skip_topk = []

    class Model:
        model = SimpleNamespace(set_skip_topk=lambda value: skip_topk.append(value))

        def __call__(self, **kwargs):
            calls.append((context.attn_metadata, kwargs["input_ids"].shape[0], context.moe_layer_index))
            return torch.ones(num_tokens, 2)

        def compute_logits(self, hidden):
            return torch.zeros(hidden.shape[0], 8)

    proposer.model = Model()
    monkeypatch.setattr(proposer_module, "get_forward_context", lambda: context)
    monkeypatch.setattr(proposer_module, "_EXTRA_CTX", SimpleNamespace())
    monkeypatch.setattr(proposer_module, "lmhead_tp_enable", lambda: False)
    monkeypatch.setattr(proposer_module, "get_ascend_config", lambda: SimpleNamespace(enable_reduce_sample=False))
    result = proposer._run_merged_draft(
        num_input_tokens=num_tokens,
        batch_size=local_reqs,
        token_indices_to_sample=torch.zeros(local_reqs, dtype=torch.int32),
        target_positions=proposer.positions,
        inputs_embeds=None,
        multi_steps_attn_metadata=[],
        num_tokens=num_tokens,
    )
    assert result.shape == (local_reqs, 3)
    assert len(calls) == 3
    assert all(metadata is None and tokens == num_tokens for metadata, tokens, _ in calls)
    assert [index for _, _, index in calls[1:]] == [0, 0]
    assert skip_topk == [False, True]


class TestDisablePaddedDrafterBatchWithFullGraph:
    """Guard: ``disable_padded_drafter_batch=True`` + cuda graph + any full
    cudagraph mode must raise ``NotImplementedError``.
    """

    @staticmethod
    def _make_proposer(
        *,
        disable_padded_drafter_batch: bool,
        use_cuda_graph: bool,
        cudagraph_mode: CUDAGraphMode,
    ) -> AscendSpecDecodeBaseProposer:
        """Bypass ``__init__`` and set only the three attrs the guard reads.

        ``cudagraph_mode`` is a real enum value so ``has_full_cudagraphs()`` is
        exercised, not stubbed.
        """
        proposer = AscendSpecDecodeBaseProposer.__new__(AscendSpecDecodeBaseProposer)
        proposer.speculative_config = SimpleNamespace(
            disable_padded_drafter_batch=disable_padded_drafter_batch,
        )
        proposer.use_cuda_graph = use_cuda_graph
        proposer.compilation_config = SimpleNamespace(cudagraph_mode=cudagraph_mode)
        return proposer

    @pytest.mark.parametrize("cudagraph_mode", FULL_CUDAGRAPH_MODES)
    def test_guard_raises_when_padded_drafter_batch_disabled_with_full_cudagraph(self, cudagraph_mode: CUDAGraphMode):
        """The bad combo: disable_padded + cuda graph + any full-cudagraph mode
        is intercepted with ``NotImplementedError``."""
        proposer = self._make_proposer(
            disable_padded_drafter_batch=True,
            use_cuda_graph=True,
            cudagraph_mode=cudagraph_mode,
        )

        with pytest.raises(NotImplementedError, match="disable_padded_drafter_batch"):
            proposer._raise_if_padded_drafter_batch_disabled_and_full_graph_enabled()

    @pytest.mark.parametrize("cudagraph_mode", NON_FULL_CUDAGRAPH_MODES)
    def test_guard_does_not_raise_without_full_cudagraph(self, cudagraph_mode: CUDAGraphMode):
        """NONE / PIECEWISE never trip the guard, even with disable_padded + cuda graph."""
        proposer = self._make_proposer(
            disable_padded_drafter_batch=True,
            use_cuda_graph=True,
            cudagraph_mode=cudagraph_mode,
        )

        # Must not raise.
        proposer._raise_if_padded_drafter_batch_disabled_and_full_graph_enabled()

    @pytest.mark.parametrize("cudagraph_mode", FULL_CUDAGRAPH_MODES)
    def test_guard_does_not_raise_when_padded_drafter_batch_enabled(self, cudagraph_mode: CUDAGraphMode):
        """Padded drafter batch on (the default) is fine with any full cudagraph."""
        proposer = self._make_proposer(
            disable_padded_drafter_batch=False,
            use_cuda_graph=True,
            cudagraph_mode=cudagraph_mode,
        )

        proposer._raise_if_padded_drafter_batch_disabled_and_full_graph_enabled()

    def test_guard_does_not_raise_when_eager(self):
        """``enforce_eager`` -> ``use_cuda_graph=False`` short-circuits the guard."""
        proposer = self._make_proposer(
            disable_padded_drafter_batch=True,
            use_cuda_graph=False,
            cudagraph_mode=CUDAGraphMode.FULL,
        )

        proposer._raise_if_padded_drafter_batch_disabled_and_full_graph_enabled()
