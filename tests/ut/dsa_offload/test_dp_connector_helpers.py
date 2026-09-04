# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""有 attention metadata 的图 dummy 也必须跳过 connector 层级回调。"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_ascend.attention import utils


@pytest.mark.parametrize("skip", [False, True])
def test_layerwise_callbacks_obey_explicit_dummy_marker(monkeypatch, skip):
    connector = Mock()
    metadata = {"layer": object()}
    context = SimpleNamespace(attn_metadata=metadata, skip_kv_transfer=skip)
    monkeypatch.setattr(utils, "has_kv_transfer_group", lambda: True)
    monkeypatch.setattr(utils, "is_v1_kv_transfer_group", lambda: True)
    monkeypatch.setattr(utils, "get_kv_transfer_group", lambda: connector)
    monkeypatch.setattr(utils, "get_forward_context", lambda: context)
    kv = [object()]
    utils.wait_for_kv_layer_from_connector("layer")
    utils.maybe_save_kv_layer_to_connector("layer", kv)
    utils.notify_kv_cache_written("layer")
    if skip:
        assert connector.mock_calls == []
    else:
        connector.wait_for_layer_load.assert_called_once_with("layer")
        connector.save_kv_layer.assert_called_once_with("layer", kv, metadata)
        connector.on_kv_cache_written.assert_called_once_with("layer")


def test_absent_notification_hook_does_not_require_forward_context(monkeypatch):
    monkeypatch.setattr(utils, "has_kv_transfer_group", lambda: True)
    monkeypatch.setattr(utils, "is_v1_kv_transfer_group", lambda: True)
    monkeypatch.setattr(utils, "get_kv_transfer_group", lambda: SimpleNamespace())
    monkeypatch.setattr(utils, "get_forward_context", lambda: pytest.fail("no notification hook"))
    utils.notify_kv_cache_written("layer")
