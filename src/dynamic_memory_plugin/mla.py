"""Narrow BF16 MLA admission and stable-view checks; no attention kernels."""


def validate_profile(config):
    import torch
    from vllm.v1.kv_cache_layout import KVCacheLayout
    from .adapter import vmm_enabled

    model, cache = config.model_config, config.cache_config
    backend = config.attention_config.backend
    if (not model.use_mla or model.dtype is not torch.bfloat16
            or getattr(backend, "name", backend) not in {None, "TRITON_MLA"}
            or cache.cache_dtype not in {"auto", "bfloat16"}
            or cache.get_resolved_kv_cache_layout() is not KVCacheLayout.BLNHC
            or not vmm_enabled()):
        raise ValueError("MLA requires BF16, TRITON_MLA and BLNHC VMM")


def validate_spec(spec):
    import torch
    from vllm.v1.kv_cache_interface import MLAAttentionSpec
    from vllm.v1.kv_cache_interface import KVQuantMode

    if (type(spec) is not MLAAttentionSpec or spec.dtype is not torch.bfloat16
            or spec.num_kv_heads != 1 or spec.head_size_v != 0
            or spec.head_size <= 0 or spec.block_size <= 0 or spec.block_size % 16
            or spec.kv_quant_mode != KVQuantMode.NONE
            or spec.cache_dtype_str not in {None, "auto", "bfloat16"}
            or spec.tokens_per_state != 1 or spec.num_head_slots not in (None, 1)
            or spec.state_content_bytes is not None or spec.page_size_padded is not None
            or spec.alignment is not None or spec.block_stride_alignment is not None
            or spec.model_version is not None or spec.is_index_group_leader
            or spec.sliding_window is not None or spec.attention_chunk_size is not None
            or spec.non_causal or spec.non_causal_multi_token_decode
            or spec.page_size_bytes != spec.block_size * spec.head_size * 2):
        raise ValueError("Only unpacked, causal BF16 MLAAttentionSpec storage is supported")


def validate_allocation(config, kv_config):
    """Native spec/allocation metadata is authoritative, not HF dimensions."""
    from .storage_contract import validate_allocation as validate_storage
    validate_spec(kv_config.kv_cache_groups[0].kv_cache_spec)
    validate_storage(kv_config, vmm=True)


def binding_signature(runner):
    """Check the real 3D aliases used by MLA, not only runner's 4D views."""
    signature = []
    context = runner.compilation_config.static_forward_context
    views = runner._dynamic_memory_vmm_views
    if (len(runner.kv_caches) != len(views)
            or {id(v) for v in runner.kv_caches} != {id(v) for v in views.values()}):
        raise RuntimeError("MLA runner cache bindings changed")
    for name, view in views.items():
        layer = context[name]
        bound = layer.kv_cache
        expected = view.squeeze(1)
        if (layer.attn_backend.get_name() != "TRITON_MLA"
                or bound.untyped_storage().data_ptr() != view.untyped_storage().data_ptr()
                or bound.data_ptr() != expected.data_ptr()
                or bound.storage_offset() != expected.storage_offset()
                or bound.shape != expected.shape or bound.stride() != expected.stride()
                or bound.dtype != view.dtype or bound.device != view.device):
            raise RuntimeError("MLA layer cache is not the admitted stable native alias")
        signature.append((name, id(view), id(bound), bound.data_ptr(),
                          tuple(bound.shape), bound.stride(), bound.dtype, bound.device))
    return signature
