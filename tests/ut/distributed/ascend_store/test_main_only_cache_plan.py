# SPDX-License-Identifier: Apache-2.0
"""Fixed-capacity Main scratch planning, independent of P/D transport."""

import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import torch
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSFAIndexerCacheSpec
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend import mooncake_layerwise
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_cache_layout import (
    apply_layerwise_kv_cache_plan,
    build_main_only_kv_cache_plan,
)


class TestMainOnlyCachePlan(unittest.TestCase):
    def setUp(self):
        # Store's lightweight test harness mocks utils; keep this adapter scoped
        # to these tests so real descriptor consumers are not globally replaced.
        patcher = patch(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_cache_layout."
            "get_kv_cache_tensor_layers",
            side_effect=lambda tensor: tensor.layers,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def make_config(self, layers=8, blocks=4):
        main = AscendMLAAttentionSpec(block_size=2, num_kv_heads=1, head_size=8, dtype=torch.int8)
        indexer = AscendSFAIndexerCacheSpec(block_size=2, num_kv_heads=1, head_size=8, dtype=torch.int8)
        specs = {
            **{f"model.layers.{layer}.attn": main for layer in range(layers)},
            # Names intentionally do not identify the component; spec types do.
            **{f"model.layers.{layer}.lookup": indexer for layer in range(layers)},
        }
        return self.config_from_specs(specs, blocks), main, indexer

    @staticmethod
    def config_from_specs(specs, blocks=4):
        return KVCacheConfig(
            num_blocks=blocks,
            kv_cache_groups=[KVCacheGroupSpec(layer_names=[name], kv_cache_spec=spec) for name, spec in specs.items()],
            kv_cache_tensors=[
                KVCacheTensor(
                    layers=[name],
                    size=blocks * spec.page_size_bytes,
                    layer_stride=0,
                    block_stride=spec.page_size_bytes,
                    offset=0,
                )
                for name, spec in specs.items()
            ],
        )

    @staticmethod
    def extra(buffers=1, independent=None):
        return {
            "backend": "mooncake",
            "use_layerwise": True,
            "layerwise_num_shared_buffers": buffers,
            "layerwise_independent_layers": [0, -1] if independent is None else independent,
        }

    def test_one_two_three_slots_keep_indexers_private_at_equal_page_size(self):
        for buffers in (1, 2, 3):
            with self.subTest(buffers=buffers):
                config, main, indexer = self.make_config()
                original_tensors = list(config.kv_cache_tensors)
                self.assertEqual(main.page_size_bytes, indexer.page_size_bytes)

                plan = build_main_only_kv_cache_plan(config, 8, self.extra(buffers))
                self.assertEqual(len(plan.layout.buffer_slots), buffers + 2)
                self.assertEqual(len(plan.tensors), buffers + 2 + 8)
                self.assertEqual(
                    plan.bytes_per_block, (buffers + 2) * main.page_size_bytes + 8 * indexer.page_size_bytes
                )
                self.assertEqual(sum(tensor.size for tensor in plan.tensors), config.num_blocks * plan.bytes_per_block)
                self.assertEqual(config.num_blocks, 4)
                self.assertEqual(config.kv_cache_tensors, original_tensors)

                # Exercise descriptors as an allocator consumer: shared Main
                # views observe the same sentinel; all Indexer backings differ.
                views = {}
                for slot, tensor in enumerate(plan.tensors):
                    backing = torch.full((tensor.size,), slot, dtype=torch.int8)
                    for name in tensor.layers:
                        views[name] = backing
                indexer_ptrs = {views[f"model.layers.{layer}.lookup"].data_ptr() for layer in range(8)}
                self.assertEqual(len(indexer_ptrs), 8)
                for slot in plan.layout.buffer_slots:
                    slot_views = [views[f"model.layers.{layer}.attn"] for layer in slot]
                    self.assertEqual(len({view.data_ptr() for view in slot_views}), 1)
                    self.assertTrue(all(torch.equal(view, slot_views[0]) for view in slot_views))
                    self.assertNotIn(slot_views[0].data_ptr(), indexer_ptrs)
                self.assertEqual(plan.layout.independent_layers, [0, 7])
                self.assertIsInstance(plan.layout.layer_cache_specs[0].indexer.spec, AscendSFAIndexerCacheSpec)

    def test_equal_page_sizes_with_different_dtype_do_not_alias(self):
        config, _, _ = self.make_config(layers=6)
        for layer in (1, 3, 5):
            group = config.kv_cache_groups[layer]
            group.kv_cache_spec = replace(group.kv_cache_spec, dtype=torch.float16, head_size=4)
        plan = build_main_only_kv_cache_plan(config, 6, self.extra(independent=[]))
        self.assertEqual(plan.layout.buffer_slots, ((0, 2, 4), (1, 3, 5)))

    def test_enough_slots_preserve_private_main_and_indexer_capacity(self):
        config, main, indexer = self.make_config()
        plan = build_main_only_kv_cache_plan(config, 8, self.extra(8))
        self.assertFalse(plan.layout.has_layer_reuse)
        self.assertTrue(all(len(tensor.layers) == 1 for tensor in plan.tensors))
        self.assertEqual(plan.bytes_per_block, 8 * (main.page_size_bytes + indexer.page_size_bytes))

    def test_indexer_quantization_and_replication_keep_full_private_capacity(self):
        config, _, _ = self.make_config()
        for group in config.kv_cache_groups[8:]:
            group.kv_cache_spec = replace(
                group.kv_cache_spec,
                scale_dim=1,
                scale_dtype=torch.float16,
                sfa_dcp_replicated_indexer_size=2,
                cache_sparse_li_c8=True,
            )
        # Preserve the generic descriptor contract accepted by the runner:
        # indexer scale/replication capacity is rebuilt from the physical spec.
        plan = build_main_only_kv_cache_plan(config, 8, self.extra())
        indexers = [tensor for tensor in plan.tensors if tensor.layers[0].endswith("lookup")]
        self.assertEqual(len(indexers), 8)
        self.assertTrue(all(tensor.size == 4 * 2 * 2 * (8 + 2) for tensor in indexers))

    def test_native_indexer_aliases_are_rejected(self):
        config, _, indexer = self.make_config()
        config.kv_cache_tensors[-2:] = [
            KVCacheTensor(
                layers=["model.layers.6.lookup", "model.layers.7.lookup"],
                size=4 * indexer.page_size_bytes,
                layer_stride=0,
                block_stride=indexer.page_size_bytes,
                offset=0,
            )
        ]
        with self.assertRaisesRegex(NotImplementedError, "native cross-layer Indexer"):
            build_main_only_kv_cache_plan(config, 8, self.extra())

    def test_missing_duplicate_out_of_bounds_and_unknown_stride_descriptors_fail(self):
        for fault in ("missing", "duplicate", "bounds", "stride"):
            with self.subTest(fault=fault):
                config, _, _ = self.make_config()
                if fault == "missing":
                    config.kv_cache_tensors.pop()
                elif fault == "duplicate":
                    config.kv_cache_tensors.append(config.kv_cache_tensors[0])
                elif fault == "bounds":
                    config.kv_cache_tensors[0] = replace(config.kv_cache_tensors[0], offset=1)
                else:
                    config.kv_cache_tensors[0] = replace(config.kv_cache_tensors[0], block_stride=1)
                with self.assertRaises((ValueError, NotImplementedError)):
                    build_main_only_kv_cache_plan(config, 8, self.extra())

    def test_missing_main_extra_components_and_special_layouts_fail(self):
        for fault in ("missing_main", "extra_main", "mtp", "compressed", "host", "block_size", "mamba"):
            with self.subTest(fault=fault):
                config, main, _ = self.make_config()
                specs = {group.layer_names[0]: group.kv_cache_spec for group in config.kv_cache_groups}
                if fault == "missing_main":
                    specs.pop("model.layers.0.attn")
                elif fault == "extra_main":
                    specs["model.layers.0.other"] = main
                elif fault == "mtp":
                    specs["model.mtp.0.attn"] = main
                elif fault == "compressed":
                    specs["model.layers.0.attn"] = replace(main, tokens_per_state=2)
                elif fault == "host":
                    specs["model.layers.0.attn"] = replace(main, store_on_host=True)
                elif fault == "block_size":
                    specs["model.layers.0.lookup"] = replace(specs["model.layers.0.lookup"], block_size=4)
                else:
                    specs["model.layers.0.state"] = MambaSpec(block_size=2, shapes=((1,),), dtypes=(torch.int8,))
                with self.assertRaises((ValueError, NotImplementedError)):
                    build_main_only_kv_cache_plan(self.config_from_specs(specs), 8, self.extra())

    def test_invalid_config_does_not_silently_truncate_shared_buffer_count(self):
        config, _, _ = self.make_config()
        for count in (True, 1.5, 0, -1, "1.5"):
            with self.subTest(count=count), self.assertRaises((TypeError, ValueError)):
                build_main_only_kv_cache_plan(config, 8, self.extra(count))

    def test_duplicate_manager_membership_is_rejected(self):
        config, _, _ = self.make_config()
        config.kv_cache_groups.append(config.kv_cache_groups[0])
        with self.assertRaisesRegex(ValueError, "exactly one manager group"):
            build_main_only_kv_cache_plan(config, 8, self.extra())

    def test_zero_block_profile_has_zero_allocation_and_known_page_budget(self):
        config, main, indexer = self.make_config(blocks=0)
        plan = build_main_only_kv_cache_plan(config, 8, self.extra())
        self.assertEqual(sum(tensor.size for tensor in plan.tensors), 0)
        self.assertEqual(plan.bytes_per_block, 3 * main.page_size_bytes + 8 * indexer.page_size_bytes)

    def test_profile_budget_counts_every_private_indexer(self):
        config, _, _ = self.make_config()
        specs = {group.layer_names[0]: group.kv_cache_spec for group in config.kv_cache_groups}
        plan = build_main_only_kv_cache_plan(config, 8, self.extra())
        layers, slots, factor = mooncake_layerwise.get_reuse_memory_info(specs, 8, self.extra())
        self.assertEqual((layers, slots), (8, 3))
        self.assertEqual(factor, sum(spec.page_size_bytes for spec in specs.values()) / plan.bytes_per_block)
        self.assertLess(factor, layers / slots)

    def vllm_config(self):
        return SimpleNamespace(
            kv_transfer_config=SimpleNamespace(
                kv_connector="AscendStoreConnector",
                kv_role="kv_producer",
                kv_connector_extra_config=self.extra(),
            ),
            model_config=SimpleNamespace(get_num_layers=lambda parallel: 8, enforce_eager=True),
            parallel_config=SimpleNamespace(pipeline_parallel_size=1),
            speculative_config=None,
            use_v2_model_runner=False,
        )

    def test_runner_uses_backend_plan_without_changing_block_capacity(self):
        config, _, _ = self.make_config()
        specs = {group.layer_names[0]: group.kv_cache_spec for group in config.kv_cache_groups}
        config.kv_cache_groups = [
            KVCacheGroupSpec(
                layer_names=list(specs),
                kv_cache_spec=UniformTypeKVCacheSpecs(block_size=2, kv_cache_specs=specs),
            )
        ]
        runtime = self.vllm_config()
        extra = runtime.kv_transfer_config.kv_connector_extra_config
        expected = mooncake_layerwise.build_cache_plan(config, runtime, extra)
        # Runtime opt-in stays disabled in production until barriers are ready.
        with patch.object(mooncake_layerwise, "extract_layout_config", return_value=extra):
            apply_layerwise_kv_cache_plan(config, runtime)
        self.assertEqual(config.kv_cache_tensors, list(expected.tensors))
        self.assertEqual(config.num_blocks, 4)

    def test_glm_mtp_keeps_a_private_main_and_indexer_without_moving_target_slots(self):
        config, main, indexer = self.make_config(layers=9)
        specs = {group.layer_names[0]: group.kv_cache_spec for group in config.kv_cache_groups}
        config.kv_cache_groups = [
            KVCacheGroupSpec(
                layer_names=list(specs),
                kv_cache_spec=UniformTypeKVCacheSpecs(block_size=2, kv_cache_specs=specs),
            )
        ]
        runtime = self.vllm_config()
        runtime.speculative_config = SimpleNamespace(method="mtp")
        runtime.model_config.hf_text_config = SimpleNamespace(model_type="glm_moe_dsa", num_nextn_predict_layers=1)
        extra = runtime.kv_transfer_config.kv_connector_extra_config
        runtime.cache_config = SimpleNamespace(enable_prefix_caching=True)
        with self.assertRaisesRegex(NotImplementedError, "requires prefix caching disabled"):
            mooncake_layerwise.build_cache_plan(config, runtime, extra)
        runtime.cache_config.enable_prefix_caching = False
        plan = mooncake_layerwise.build_cache_plan(config, runtime, extra)
        baseline, _, _ = self.make_config()
        baseline_plan = build_main_only_kv_cache_plan(baseline, 8, extra)

        self.assertEqual(plan.layout.buffer_slots, (*baseline_plan.layout.buffer_slots, (8,)))
        self.assertEqual(plan.layout.independent_layers, [0, 7, 8])
        self.assertEqual(plan.layout.prefetch_layer_map, baseline_plan.layout.prefetch_layer_map)
        self.assertIn(["model.layers.8.attn"], [tensor.layers for tensor in plan.tensors])
        self.assertIn(["model.layers.8.lookup"], [tensor.layers for tensor in plan.tensors])
        self.assertEqual(
            plan.bytes_per_block, baseline_plan.bytes_per_block + main.page_size_bytes + indexer.page_size_bytes
        )
        layers, slots, factor = mooncake_layerwise.get_reuse_memory_info(specs, 8, extra, vllm_config=runtime)
        self.assertEqual((layers, slots), (9, 4))
        self.assertEqual(factor, sum(spec.page_size_bytes for spec in specs.values()) / plan.bytes_per_block)

    def test_glm_mtp_store_keys_are_separate_from_target_only_keys(self):
        runtime = self.vllm_config()
        runtime.parallel_config.tensor_parallel_size = 16
        runtime.model_config.hf_text_config = SimpleNamespace(model_type="glm_moe_dsa", num_nextn_predict_layers=1)
        runtime.model_config.get_total_num_hidden_layers = lambda: 8
        runtime.model_config.get_layers_start_end_indices = lambda parallel: (0, 8)
        runtime.model_config.compute_hash = lambda: "glm"
        runtime.cache_config = SimpleNamespace(block_size=2, cache_dtype="auto", compute_hash=lambda: "cache")

        self.assertEqual(mooncake_layerwise.layerwise_topology_namespace(runtime), "")
        runtime.speculative_config = SimpleNamespace(method="mtp", compute_hash=lambda: "mtp")
        self.assertTrue(mooncake_layerwise.layerwise_topology_namespace(runtime))
        runtime.kv_transfer_config.kv_connector_extra_config.pop("layerwise_num_shared_buffers")
        self.assertEqual(mooncake_layerwise.layerwise_topology_namespace(runtime), "")

    def test_disabled_mooncake_reuse_keeps_original_descriptors(self):
        config, _, _ = self.make_config()
        original = config.kv_cache_tensors
        runtime = self.vllm_config()
        runtime.kv_transfer_config.kv_connector_extra_config.pop("layerwise_num_shared_buffers")
        apply_layerwise_kv_cache_plan(config, runtime)
        self.assertIs(config.kv_cache_tensors, original)

    def test_explicit_large_buffer_count_preserves_ordinary_decode_configuration(self):
        config, _, _ = self.make_config()
        runtime = self.vllm_config()
        runtime.kv_transfer_config.kv_role = "kv_consumer"
        runtime.speculative_config = SimpleNamespace(method="mtp")
        extra = runtime.kv_transfer_config.kv_connector_extra_config
        extra["layerwise_num_shared_buffers"] = 8
        original = config.kv_cache_tensors
        apply_layerwise_kv_cache_plan(config, runtime)
        self.assertIs(config.kv_cache_tensors, original)

    def test_mooncake_opt_in_requires_both_layerwise_and_shared_count(self):
        self.assertIsNone(mooncake_layerwise.extract_layout_config({"use_layerwise": True}))
        self.assertIsNone(mooncake_layerwise.extract_layout_config({"layerwise_num_shared_buffers": 3}))
        extra = self.extra()
        self.assertIs(mooncake_layerwise.extract_layout_config(extra), extra)

    def test_backend_rejects_decode_pp_speculative_and_unverified_runner(self):
        for fault in ("decode", "pp", "speculative", "v2", "cp", "graph"):
            with self.subTest(fault=fault):
                config, _, _ = self.make_config()
                runtime = self.vllm_config()
                if fault == "decode":
                    runtime.kv_transfer_config.kv_role = "kv_consumer"
                elif fault == "pp":
                    runtime.parallel_config.pipeline_parallel_size = 2
                elif fault == "speculative":
                    runtime.speculative_config = SimpleNamespace(method="mtp")
                elif fault == "v2":
                    runtime.use_v2_model_runner = True
                elif fault == "graph":
                    runtime.model_config.enforce_eager = False
                else:
                    runtime.parallel_config.prefill_context_parallel_size = 2
                with self.assertRaises((ValueError, NotImplementedError)):
                    mooncake_layerwise.build_cache_plan(config, runtime, self.extra())

    def test_backend_rejects_unverified_multiple_manager_groups(self):
        config, _, _ = self.make_config()
        with self.assertRaisesRegex(NotImplementedError, "single KV manager group"):
            mooncake_layerwise.build_cache_plan(config, self.vllm_config(), self.extra())

    def test_store_uses_the_same_main_layout_as_the_runner(self):
        from tests.ut.distributed.ascend_store.test_pool_worker import make_worker

        config, _, _ = self.make_config()
        plan = build_main_only_kv_cache_plan(config, 8, self.extra())
        with (
            patch.object(mooncake_layerwise, "extract_layout_config", return_value=self.extra()),
            patch.object(mooncake_layerwise, "build_cache_plan", return_value=plan) as build,
        ):
            worker = make_worker(
                self,
                num_layers=8,
                use_mla=True,
                use_layerwise=True,
                extra_config=self.extra(),
                kv_cache_config=config,
            )
        build.assert_called_once()
        self.assertTrue(worker.layerwise_offload)
        self.assertEqual(worker.prefetch_layer_map, plan.layout.prefetch_layer_map)
        self.assertEqual(worker.independent_layers, plan.layout.independent_layers)
