"""Validate ordinary native Triton cache bindings before preserved resize."""

from collections import Counter


def _binding_key(view):
    return (view.data_ptr(), tuple(view.shape), tuple(view.stride()), view.dtype,
            view.storage_offset())


def validate_bindings(runner, config):
    import torch
    from vllm.model_executor.layers.attention import Attention
    from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
    from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl
    from vllm.v1.kv_cache_interface import create_kv_cache_views
    from .storage_contract import validate_allocation

    validate_allocation(config)
    if getattr(runner, "_dynamic_memory_vmm_arena", None) is not None:
        raise ValueError("resize.triton: VMM has no composed storage adapter")
    group = config.kv_cache_groups[0]
    spec = group.kv_cache_spec
    names = group.layer_names
    context = runner.compilation_config.static_forward_context
    if (runner.shared_kv_cache_layers or runner._kernel_block_sizes != [spec.block_size]
            or {name for name, layer in context.items()
                if isinstance(layer, AttentionLayerBase)} != set(names)
            or len(runner.kv_caches) != len(names)):
        raise ValueError("resize.triton: complete unshared attention bindings required")
    tensor = config.kv_cache_tensors[0]
    references = create_kv_cache_views(
        torch.empty(tensor.size, dtype=torch.int8, device="meta"), spec,
        config.num_blocks, runner.cache_config.get_resolved_kv_cache_layout(), tensor,
        kernel_block_size=spec.block_size)
    stores = set()
    bindings = Counter(_binding_key(view) for view in runner.kv_caches)
    signature = lambda view: (view.shape, view.stride(), view.dtype,
                              view.storage_offset())
    for name, reference in zip(names, references):
        layer = context[name]
        impl, view = layer.impl, layer.kv_cache
        if (type(layer) is not Attention or type(impl) is not TritonAttentionImpl
                or layer.query_quant is not None
                or layer.kv_cache_dtype not in {"auto", "bfloat16", "float16"}
                or impl.kv_cache_dtype != layer.kv_cache_dtype
                or impl.num_kv_heads != spec.num_kv_heads
                or impl.head_size != spec.head_size
                or layer.sliding_window is not None
                or layer.head_size_v != spec.head_size_v):
            raise ValueError("resize.triton: native ordinary reader/writer contract differs")
        if (signature(view) != signature(reference)
                or view.untyped_storage().nbytes() != tensor.size):
            raise ValueError("resize.triton: live view differs from native allocation descriptor")
        stores.add(view.untyped_storage().data_ptr())
        if bindings[_binding_key(view)] != 1:
            raise ValueError("resize.triton: runner and attention bindings disagree")
    if len(stores) != 1:
        raise ValueError("resize.triton: layers must share one owned native allocation")
