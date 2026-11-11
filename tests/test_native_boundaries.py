"""Real native metadata plus the storage/ownership boundaries of resize."""
from dataclasses import replace
import os
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch
from vllm.config import AttentionConfig, CacheConfig
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id
from vllm.v1.kv_cache_interface import FullAttentionSpec, MLAAttentionSpec, KVCacheGroupSpec

from dynamic_memory_plugin.adapter import EXTENSION, plan_cache
from dynamic_memory_plugin.compat import verify_runtime
from dynamic_memory_plugin.runtime_profile import plan_kv_cache
from dynamic_memory_plugin.storage_contract import validate_allocation
from dynamic_memory_plugin.withdrawal import WithdrawingBlockPool
from dynamic_memory_plugin.worker import DynamicMemoryWorker, _authorize_resize


class NativeBoundaryTests(unittest.TestCase):
    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized(), 'CPU qualification initialized CUDA')

    def test_exact_installed_native_seams(self):
        checked = verify_runtime()
        self.assertIn(checked['version'], ('0.32.0', '0.33.0'))
        self.assertGreater(checked['checked_files'], 0)

    def test_native_geometry_planning_and_unsupported_representation(self):
        config = NS(
            cache_config=CacheConfig(enable_prefix_caching=True),
            attention_config=AttentionConfig(),
            model_config=NS(max_model_len=64, enforce_eager=True, hf_config=NS()),
            parallel_config=NS(worker_extension_cls=EXTENSION, decode_context_parallel_size=1),
            scheduler_config=NS(async_scheduling=False, scheduler_cls=None, watermark=0.0),
            compilation_config=NS(cudagraph_mode=NS(name='NONE'), mode=0),
            device_config=NS(device_type='cuda'), use_v2_model_runner=False,
            max_in_flight_tokens=512)
        config.cache_config.kv_cache_layout = 'BLNHC'
        specs = [FullAttentionSpec(block_size=16, num_kv_heads=2, head_size=128, dtype=torch.bfloat16),
                 MLAAttentionSpec(block_size=16, num_kv_heads=1, head_size=576, dtype=torch.bfloat16)]
        for spec in specs:
            with self.subTest(spec=type(spec).__name__):
                old = plan_kv_cache(config, [KVCacheGroupSpec(['layer'], spec)], 32*spec.page_size_bytes)
                old.kv_cache_layout = 'BLNHC'
                with patch.dict(os.environ, VLLM_DYNAMIC_MEMORY_ALLOCATOR='vmm',
                                VLLM_DYNAMIC_MEMORY_CHUNK_BYTES=str(8*spec.page_size_bytes)):
                    new = plan_cache(config, old, 16*spec.page_size_bytes)
                self.assertEqual((old.num_blocks, new.num_blocks), (32, 16))
                self.assertEqual(validate_allocation(old, vmm=True)['packed_block_bytes'], spec.page_size_bytes)
                old.kv_cache_groups[0].kv_cache_spec = replace(spec, dtype=torch.float8_e4m3fn)
                with self.assertRaises(ValueError):
                    validate_allocation(old, vmm=True)

    def test_retiring_tail_waits_for_live_references(self):
        pool = WithdrawingBlockPool.adopt(BlockPool(10, True, 16))
        active = pool.get_new_blocks(6)
        pool.begin_withdrawal(5)
        self.assertFalse(pool.ready())
        self.assertEqual(pool.get_num_free_blocks(), 0)
        pool.free_blocks(active[4:])
        self.assertTrue(pool.ready())
        pool.commit_capacity(5)
        self.assertEqual(pool.blocks[1:], active[:4])
        self.assertTrue(all(block.ref_cnt == 1 for block in active[:4]))

    def test_retiring_prefix_cannot_be_hit_or_reinserted(self):
        pool = WithdrawingBlockPool.adopt(BlockPool(10, True, 16))
        active = pool.get_new_blocks(6)
        key = make_block_hash_with_group_id(b'prefix', 0)
        pool._insert_block_hash(key, active[-1], 16)
        pool.begin_withdrawal(5)
        self.assertIsNone(pool.cached_block_hash_to_block.get_one_block(key))
        pool._insert_block_hash(key, active[-1], 16)
        self.assertIsNone(pool.cached_block_hash_to_block.get_one_block(key))
        pool.cancel_withdrawal()
        pool.free_blocks(active)
        self.assertEqual(pool.get_num_free_blocks(), 9)

    def test_internal_worker_permit_is_required_and_single_use(self):
        worker = NS(model_runner=NS(kv_cache_config=NS(num_blocks=8, kv_cache_tensors=[NS(size=80)])))
        context = {'controller_id': 'engine', 'operation_id': 'op', 'sequence': 1}
        method = 'dynamic_memory_preserve'
        kwargs = dict(expected_old_bytes=80, expected_old_blocks=8, operation_context=context)
        call = getattr(DynamicMemoryWorker, method)
        with patch('dynamic_memory_plugin.worker.vmm_enabled', return_value=True), patch(
                'dynamic_memory_plugin.vmm_integration.resize_vmm', return_value={'outcome': 'applied'}) as physical:
            self.assertEqual(call(worker, 40, 4, **kwargs)['outcome'], 'rejected')
            with _authorize_resize(worker, method, 40, 4, **kwargs) as armed:
                arguments = dict(kwargs, operation_context=armed)
                self.assertEqual(call(worker, 40, 4, **arguments)['outcome'], 'applied')
                self.assertEqual(call(worker, 40, 4, **arguments)['outcome'], 'rejected')
            physical.assert_called_once()
