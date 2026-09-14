# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from vllm_ascend.ascend_config import AscendConfig
from vllm_ascend.utils import AscendDeviceType


class TestDSALIC8Config(unittest.TestCase):
    @staticmethod
    def _config(description, *, enabled=True, offload=True):
        config = AscendConfig.__new__(AscendConfig)
        quant_config = SimpleNamespace(quant_description=description)
        config.vllm_config = SimpleNamespace(quant_config=quant_config)
        config.dsa_offload_config = SimpleNamespace(enabled=offload)
        config.enable_sparse_li_c8 = enabled
        config._sparse_li_c8_layer_filter_enabled = config._has_sparse_li_c8_layer_config(quant_config)
        (
            config._sparse_li_c8_layer_ids,
            config._sparse_li_c8_layer_names,
        ) = config._parse_sparse_li_c8_layers_from_quant_config(quant_config)
        return config

    @staticmethod
    def _public_description():
        description = {
            "indexer_quant_type": "INT8_DYNAMIC",
            "model.layers.0.self_attn.indexer.wq_b.weight": "W8A8_MXFP8",
            "model.layers.0.self_attn.indexer.k_norm.weight": "FLOAT",
        }
        description.update({f"model.layers.{i}.self_attn.indexer.quant_type": "FP8_DYNAMIC" for i in range(6, 67, 4)})
        return description

    def _is_enabled(self, config, layer=0, device=AscendDeviceType.A5):
        with patch("vllm_ascend.utils.get_ascend_device_type", return_value=device):
            return config.is_sparse_li_c8_layer(f"model.layers.{layer}.self_attn.attn")

    def test_public_checkpoint_keeps_all_target_full_indexers_c8(self):
        config = self._config(self._public_description())
        self.assertTrue(config._sparse_li_c8_layer_filter_enabled)
        for layer in (0, 2, *range(6, 75, 4), 76):
            with self.subTest(layer=layer):
                self.assertTrue(self._is_enabled(config, layer))

    def test_legacy_checkpoint_keeps_global_switch_behavior(self):
        config = self._config({"model.layers.0.self_attn.indexer.wq_b.weight": "W8A8_MXFP8"})
        self.assertFalse(config._sparse_li_c8_layer_filter_enabled)
        self.assertTrue(self._is_enabled(config))

    def test_global_type_does_not_require_projection_metadata(self):
        for quant_type in ("INT8_DYNAMIC", "FP8_DYNAMIC", "W8A8_MXFP8"):
            with self.subTest(quant_type=quant_type):
                config = self._config({"indexer_quant_type": quant_type})
                self.assertTrue(self._is_enabled(config))

    def test_layer_override_takes_precedence_over_global_and_weight(self):
        for quant_type in ("FLOAT", "UNKNOWN", "", None):
            with self.subTest(quant_type=quant_type):
                description = self._public_description()
                description["model.layers.0.self_attn.indexer.quant_type"] = quant_type
                # Neither projection spelling can override an explicit cache type.
                description["model.layers.0.self_attn.indexer.wq_b_weight"] = "W8A8_MXFP8"
                config = self._config(description)
                self.assertFalse(self._is_enabled(config))
                self.assertTrue(self._is_enabled(config, 6))

    def test_quantized_layer_can_override_float_global_default(self):
        config = self._config(
            {
                "indexer_quant_type": "FLOAT",
                "model.layers.6.self_attn.indexer.quant_type": "FP8_DYNAMIC",
            }
        )
        self.assertFalse(self._is_enabled(config))
        self.assertTrue(self._is_enabled(config, 6))

    def test_invalid_global_type_does_not_fall_back_to_projection(self):
        for quant_type in ("FLOAT", "UNKNOWN", "", None):
            with self.subTest(quant_type=quant_type):
                config = self._config(
                    {
                        "indexer_quant_type": quant_type,
                        "model.layers.0.self_attn.indexer.wq_b.weight": "W8A8_MXFP8",
                    }
                )
                self.assertFalse(self._is_enabled(config))

    def test_explicit_only_metadata_does_not_enable_unlisted_layers(self):
        config = self._config({"model.layers.6.self_attn.indexer.quant_type": "FP8_DYNAMIC"})
        self.assertFalse(self._is_enabled(config))
        self.assertTrue(self._is_enabled(config, 6))

    def test_global_switch_off_still_wins(self):
        config = self._config(self._public_description(), enabled=False)
        self.assertFalse(self._is_enabled(config))
        self.assertFalse(self._is_enabled(config, 6))

    def test_non_offload_keeps_existing_filter(self):
        config = self._config(self._public_description(), offload=False)
        self.assertFalse(self._is_enabled(config))
        self.assertFalse(self._is_enabled(config, 6))

    def test_non_a5_keeps_existing_filter(self):
        config = self._config(self._public_description())
        self.assertFalse(self._is_enabled(config, device=AscendDeviceType.A3))
        self.assertFalse(self._is_enabled(config, 6, device=AscendDeviceType.A3))

    def test_missing_quant_description_keeps_legacy_behavior(self):
        for description in (None, {}):
            with self.subTest(description=description):
                self.assertTrue(self._is_enabled(self._config(description)))

    def test_trailing_dot_in_attention_name(self):
        config = self._config({"model.layers.6.self_attn.indexer.quant_type": "FP8_DYNAMIC"})
        with patch("vllm_ascend.utils.get_ascend_device_type", return_value=AscendDeviceType.A5):
            self.assertTrue(config.is_sparse_li_c8_layer("model.layers.6.self_attn.attn."))


if __name__ == "__main__":
    unittest.main()
