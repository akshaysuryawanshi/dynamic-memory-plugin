"""Resize admission from pinned native storage, independent of model identity.

Model loading and kernel support remain vLLM's responsibility. This module
checks only the state interpretation/placement used by the resize operation.
"""


def validate_spec(spec, *, allow_mla=False):
    import torch
    from vllm.v1.kv_cache_interface import FullAttentionSpec, MLAAttentionSpec, KVQuantMode
    if type(spec) is MLAAttentionSpec and allow_mla:
        from .mla import validate_spec as validate_mla
        validate_mla(spec)
        return "uniform_mla"
    if type(spec) is not FullAttentionSpec:
        raise ValueError("resize.cache_kind: requires native FullAttentionSpec or BF16 MLA VMM storage")
    if (spec.dtype not in {torch.bfloat16, torch.float16}
            or spec.kv_quant_mode != KVQuantMode.NONE):
        raise ValueError("resize.cache_representation: requires unpacked BF16/FP16 KV")
    if spec.non_causal or spec.sliding_window is not None or spec.attention_chunk_size is not None:
        raise ValueError("resize.attention_semantics: causal full attention required; window/chunk/noncausal adapter absent")
    if (any(type(v) is not int or v <= 0 for v in
            (spec.block_size, spec.num_kv_heads, spec.head_size, spec.head_size_v))
            or spec.head_size_v != spec.head_size
            or spec.tokens_per_state != 1
            or spec.num_head_slots not in (None, spec.num_kv_heads)
            or spec.state_content_bytes is not None
            or spec.page_size_padded is not None
            or spec.block_stride_alignment is not None):
        raise ValueError("resize.cache_geometry: requires unpacked equal-width K/V token pages without padding")
    return "uniform_full_attention"


def validate_allocation(kv_config, *, vmm=False):
    """Validate complete native allocation before planning or touching CUDA.

    LBNHC remains the existing native allocation path; VMM requires stable
    block-major offsets (BLNHC). Group count and storage ownership are separate
    from the model's expert routing or projection dimensions.
    """
    from vllm.v1.kv_cache_interface import KVCacheGroupRole
    if len(kv_config.kv_cache_groups) != 1:
        raise ValueError("resize.cache_groups: Exactly one uniform cache group is supported")
    group = kv_config.kv_cache_groups[0]
    kind = validate_spec(group.kv_cache_spec, allow_mla=vmm)
    names = group.layer_names
    if (not names or len(set(names)) != len(names) or group.host_resident
            or group.is_eagle_group or group.role != KVCacheGroupRole.DEFAULT
            or kv_config.kv_tp_replicas != 1
            or kv_config.hisparse_host_num_blocks is not None
            or kv_config.hisparse_shared_host_pool):
        raise ValueError("resize.cache_ownership: unique device layers and ordinary single-worker ownership required")
    if len(kv_config.kv_cache_tensors) != 1:
        raise ValueError("resize.cache_allocation: requires one shared backing allocation")
    tensor = kv_config.kv_cache_tensors[0]
    page = group.kv_cache_spec.page_size_bytes
    blocks = kv_config.num_blocks
    layout = getattr(kv_config.kv_cache_layout, 'name', kv_config.kv_cache_layout)
    if layout not in {"BLNHC", "LBNHC"} or (vmm and layout != "BLNHC"):
        raise ValueError("resize.cache_layout: stable native BLNHC required for VMM; native path allows LBNHC")
    strides = (page, page * len(names)) if layout == "BLNHC" else (page * blocks, page)
    if (type(blocks) is not int or blocks <= 0 or tensor.offset != 0
            or tensor.host_resident or list(tensor.layers) != list(names)
            or (tensor.layer_stride, tensor.block_stride) != strides
            or tensor.size != blocks * len(names) * page):
        raise ValueError("resize.cache_allocation: backing differs from native uniform geometry")
    return {
        "contract": kind,
        "layout": layout,
        "spec_type": type(group.kv_cache_spec).__name__,
        "layers": len(names),
        "block_size": group.kv_cache_spec.block_size,
        "num_kv_heads": group.kv_cache_spec.num_kv_heads,
        "head_size": group.kv_cache_spec.head_size,
        "head_size_v": group.kv_cache_spec.head_size_v,
        "dtype": str(group.kv_cache_spec.dtype),
        "packed_block_bytes": page * len(names),
    }
