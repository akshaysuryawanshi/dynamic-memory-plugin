"""Opt-in stable-view CUDA VMM adapter for the reviewed eager runner.

Only scheduler-admitted blocks may be touched. Views span reserved virtual
addresses, while the arena maps just the admitted prefix of physical chunks.
"""
import os
from dataclasses import dataclass
from threading import get_native_id
from functools import wraps
from time import monotonic_ns
from types import MethodType

from .adapter import (
    CacheCapacityPlan,
    backing_bytes,
    chunk_bytes,
    execution_headroom_bytes,
    plan_cache,
    validate_config,
)


@dataclass(frozen=True)
class _VmmAllocationPlan:
    layout: object
    spec: object
    reserved_config: object
    initial_bytes: int
    reserved_bytes: int
    chunk_bytes: int
    mla: bool


def _plan_vmm_allocation(runner, config, kernel_block_sizes):
    from vllm.v1.kv_cache_interface import MLAAttentionSpec
    from vllm.v1.kv_cache_layout import KVCacheLayout
    from .storage_contract import validate_allocation

    validate_config(runner.vllm_config)
    layout = runner.cache_config.get_resolved_kv_cache_layout()
    if layout is not KVCacheLayout.BLNHC:
        raise ValueError("VMM requires VLLM_KV_CACHE_LAYOUT=BLNHC")
    if hasattr(runner, "_dynamic_memory_vmm_arena"):
        raise RuntimeError("VMM cache already initialized")
    if (len(config.kv_cache_groups) != 1 or config.needs_kv_cache_zeroing
            or runner.shared_kv_cache_layers):
        raise ValueError(
            "VMM requires one cache group without shared layers or state zeroing")

    validate_allocation(config, vmm=True)
    spec = config.kv_cache_groups[0].kv_cache_spec
    if kernel_block_sizes != [spec.block_size]:
        raise ValueError("VMM does not support kernel block splitting")
    mla = type(spec) is MLAAttentionSpec
    expected_backend = "TRITON_MLA" if mla else "FLASH_ATTN"
    if any(group.backend.get_name() != expected_backend
           for groups in runner.attn_groups for group in groups):
        raise ValueError("VMM requires " + expected_backend)
    if mla:
        from .mla import validate_allocation as validate_mla_allocation
        validate_mla_allocation(runner.vllm_config, config)

    chunk = chunk_bytes()
    initial = backing_bytes(config)
    page = initial // config.num_blocks
    if chunk % page or initial % chunk:
        raise ValueError(
            "Initial KV budget and chunk must align to complete physical chunks")
    maximum = int(os.environ.get("VLLM_DYNAMIC_MEMORY_MAX_KV_BYTES", str(1024**3)))
    reserved_config = plan_cache(runner.vllm_config, config, maximum)
    if mla:
        validate_mla_allocation(runner.vllm_config, reserved_config)
    reserved = backing_bytes(reserved_config)
    if reserved < initial or reserved % chunk:
        raise ValueError("VMM ceiling must cover initial allocation in whole chunks")
    return _VmmAllocationPlan(
        layout=layout,
        spec=spec,
        reserved_config=reserved_config,
        initial_bytes=initial,
        reserved_bytes=reserved,
        chunk_bytes=chunk,
        mla=mla,
    )


def _is_consistent_vmm_storage(diagnostics, logical_bytes):
    """Prove that logical capacity still matches every owned VMM resource."""
    return (
        not diagnostics["closed"]
        and not diagnostics["poisoned"]
        and diagnostics["address_reserved"]
        and diagnostics["accounting_consistent"]
        and diagnostics["committed_mapped_bytes"] == logical_bytes
        and diagnostics["mapped_physical_bytes"] == logical_bytes
        and diagnostics["retained_physical_bytes"] == logical_bytes
        and logical_bytes <= diagnostics["reserved_bytes"]
        and logical_bytes % diagnostics["chunk_bytes"] == 0
    )


def _initialize_tensors(runner, config, kernel_block_sizes, kv_cache_allocation_context=None):
    import torch
    from vllm.v1.kv_cache_interface import create_kv_cache_views
    from vllm.v1.worker.utils import bind_kv_cache, map_kv_caches_to_kernel_blocks
    from .vmm import CudaDriver, VmmArena

    plan = _plan_vmm_allocation(runner, config, kernel_block_sizes)
    device_index = (
        runner.device.index
        if runner.device.index is not None
        else torch.cuda.current_device()
    )
    driver = CudaDriver(device_index)
    arena = VmmArena(driver, plan.reserved_bytes, plan.chunk_bytes)
    # Keep ownership before constructing views: initialization failure exits the
    # worker rather than releasing storage with outstanding asynchronous work.
    runner._dynamic_memory_vmm_arena = arena
    arena.resize(plan.initial_bytes)
    raw = arena.tensor()
    views = {}
    for tensor in plan.reserved_config.kv_cache_tensors:
        tensor_views = create_kv_cache_views(
            raw,
            plan.spec,
            plan.reserved_config.num_blocks,
            plan.layout,
            tensor,
            kernel_block_size=kernel_block_sizes[0],
        )
        views.update(zip(tensor.layers, tensor_views))
    attention_groups = [
        group for groups in runner.attn_groups for group in groups
    ]
    layer_views = map_kv_caches_to_kernel_blocks(views, attention_groups)
    if any(layer_views[name] is not view for name, view in views.items()):
        raise ValueError(
            "Dynamic-memory BF16 profile requires unchanged scheduler/kernel KV views")
    bind_kv_cache(
        views,
        runner.compilation_config.static_forward_context,
        runner.kv_caches,
        kv_cache_groups=config.kv_cache_groups,
        layer_kv_caches=layer_views,
    )
    runner._dynamic_memory_vmm_views = views
    runner._dynamic_memory_vmm_raw = raw
    if plan.mla:
        from .mla import binding_signature
        runner._dynamic_memory_mla_binding_signature = binding_signature(runner)
    return views


def install_allocator_hook():
    """One startup hook; no wrapper is added to inference execution methods."""
    from vllm.v1.worker import gpu_model_runner
    GPUModelRunner = gpu_model_runner.GPUModelRunner
    _install_copy_hook(gpu_model_runner)
    original = GPUModelRunner.initialize_kv_cache
    if getattr(original, "_dynamic_memory_vmm_hook", False):
        return

    @wraps(original)
    def initialize(self, config, is_profiling=False, kv_cache_allocation_context=None):
        if is_profiling:
            return original(self, config, is_profiling, kv_cache_allocation_context)
        if "initialize_kv_cache_tensors" in self.__dict__:
            raise RuntimeError("Another extension owns the KV allocation hook")
        self.initialize_kv_cache_tensors = MethodType(_initialize_tensors, self)
        try:
            return original(self, config, is_profiling, kv_cache_allocation_context)
        finally:
            del self.initialize_kv_cache_tensors

    initialize._dynamic_memory_vmm_hook = True
    GPUModelRunner.initialize_kv_cache = initialize


def _install_copy_hook(gpu_model_runner):
    """Give the upstream block copier a view of only physically mapped blocks."""
    original = gpu_model_runner.copy_kv_cache_blocks_inplace
    if getattr(original, "_dynamic_memory_vmm_hook", False):
        return

    @wraps(original)
    def copy(kv_caches, num_blocks, kv_cache_block_copies):
        copies = tuple(kv_cache_block_copies)
        if not copies:
            return original(kv_caches, num_blocks, copies)
        if type(num_blocks) is not int or num_blocks <= 0:
            raise ValueError("VMM block-copy capacity must be a positive integer")
        for item in copies:
            try:
                src, dst = item
            except (TypeError, ValueError) as exc:
                raise ValueError("VMM block copy must contain a source and destination") from exc
            if (type(src) is not int or type(dst) is not int
                    or not 0 <= src < num_blocks or not 0 <= dst < num_blocks):
                raise ValueError("VMM block copy references storage outside mapped capacity")
        bounded = []
        for cache in kv_caches:
            if cache.shape[0] < num_blocks:
                raise ValueError("KV view is smaller than the mapped VMM capacity")
            bounded.append(cache[:num_blocks])
        return original(bounded, num_blocks, copies)

    copy._dynamic_memory_vmm_hook = True
    gpu_model_runner.copy_kv_cache_blocks_inplace = copy


class _VmmResizeOperation:
    """One stable-view resize, with explicit validation and publication phases."""

    def __init__(self, worker, target_bytes, expected_blocks):
        self.worker = worker
        self.runner = worker.model_runner
        self.target_bytes = target_bytes
        self.expected_blocks = expected_blocks
        self.started = monotonic_ns()
        self.phase_start = self.started
        self.spans = []
        self.arena = None
        self.verification = None

    def record_phase(self, name):
        end = monotonic_ns()
        self.spans.append({"name": name,
                           "start_monotonic_ns": self.phase_start,
                           "end_monotonic_ns": end})
        self.phase_start = end

    def error(self, outcome, exc):
        result = {"outcome": outcome,
                  "error": f"{type(exc).__name__}: {exc}",
                  "phase_spans": self.spans}
        if self.arena is not None:
            result["vmm"] = self.arena.diagnostics()
        return result

    def plan(self):
        from .worker import _assert_shrink_references

        # Current VMM consistency is a fatal check and must precede every
        # rejectable configuration or target decision.
        try:
            self.arena = self.runner._dynamic_memory_vmm_arena
            old = self.runner.kv_cache_config
            old_bytes = backing_bytes(old)
            current = self.arena.diagnostics()
            if not _is_consistent_vmm_storage(current, old_bytes):
                raise RuntimeError(
                    "Existing VMM storage and worker capacity are not safely consistent")
        except Exception as exc:
            return self.error("failed", exc)

        try:
            validate_config(self.worker.vllm_config)
            new = plan_cache(
                self.worker.vllm_config, old, self.target_bytes)
            _assert_shrink_references(self.runner, new.num_blocks)
            new_bytes = backing_bytes(new)
            if (new.num_blocks != self.expected_blocks
                    or new_bytes > self.arena.reserved_bytes):
                raise ValueError(
                    "VMM worker/scheduler capacity disagreement or ceiling exceeded")
            chunk = chunk_bytes()
            if old_bytes % chunk or new_bytes % chunk:
                raise ValueError("VMM requires complete physical chunks")
            self.reserve = execution_headroom_bytes()
            if self.reserve < 0:
                raise ValueError("Headroom must be nonnegative")
            self.capacity = CacheCapacityPlan(old, new, old_bytes, new_bytes)
            self.verify_diagnostic = os.environ.get("DYNAMIC_MEMORY_VERIFY_KV_COPY") == "1"
            self.diagnostic_bytes = min(
                old_bytes, new_bytes) if self.verify_diagnostic else 0
            self.required = max(new_bytes - old_bytes, 0) + self.diagnostic_bytes
            self.record_phase("plan_validate")
            return None
        except Exception as exc:
            return self.error("rejected", exc)

    def inspect_bindings(self):
        from .worker import _snapshot_weight_identity

        try:
            self.mla_binding = getattr(
                self.runner, "_dynamic_memory_mla_binding_signature", None)
            self.binding_signature = None
            if self.mla_binding is not None:
                from .mla import binding_signature
                self.binding_signature = binding_signature
                if binding_signature(self.runner) != self.mla_binding:
                    raise RuntimeError("MLA binding changed before resize")
            self.weight_signature = _snapshot_weight_identity(self.runner)
            self.record_phase("inspect_weights_before")
            self.pointer_signature = [
                (view.data_ptr(), tuple(view.shape), view.stride())
                for view in self.runner._dynamic_memory_vmm_views.values()
            ]
            self.input_batch = self.runner.input_batch
            self.request_states = self.runner.requests
            return None
        except Exception as exc:
            return self.error("failed", exc)

    def preflight(self, torch):
        from .worker import _empty_cache_for_resize, _memory

        try:
            self.before = _memory(torch, self.worker.device)
            self.before["external_kv_bytes"] = self.capacity.old_bytes
            needs_trim = (self.required
                          and self.required + self.reserve
                          > self.before["driver_free_bytes"])
            if needs_trim:
                _empty_cache_for_resize(torch, self.runner, "preflight")
                self.before = _memory(torch, self.worker.device)
                self.before["external_kv_bytes"] = self.capacity.old_bytes
        except Exception as exc:
            self.arena.poisoned = True
            return self.error("failed", exc)
        if (self.required
                and self.required + self.reserve > self.before["driver_free_bytes"]):
            return self.error("rejected", ValueError(
                "Insufficient free VRAM for added chunks, diagnostic snapshot, and headroom"))
        self.record_phase("preflight")
        return None

    def snapshot_diagnostic(self, torch):
        if not self.verify_diagnostic:
            return None
        try:
            self.verification = self.runner._dynamic_memory_vmm_raw[
                :self.diagnostic_bytes].clone()
            torch.cuda.synchronize(self.worker.device)
            self.record_phase("snapshot_retained_diagnostic")
            return None
        except Exception as exc:
            self.verification = None
            return {"outcome": "failed",
                    "error": f"Diagnostic snapshot failed: {exc}",
                    "vmm": self.arena.diagnostics()}

    def resize_storage(self):
        try:
            self.arena.resize(self.capacity.new_bytes)
            name = ("map_zero_fence"
                    if self.capacity.new_bytes > self.capacity.old_bytes
                    else "fence_unmap")
            self.record_phase(name)
            return None
        except Exception as exc:
            safe = (not self.arena.poisoned
                    and self.arena.mapped_bytes == self.capacity.old_bytes)
            return {"outcome": "rejected" if safe else "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "old_storage_preserved": safe, "before": self.before,
                    "phase_spans": self.spans, "vmm": self.arena.diagnostics()}

    def verify_and_publish(self, torch):
        from .worker import _empty_cache_for_resize, _memory, _weight_identity_matches

        retained_verified = self.verification is not None
        try:
            if retained_verified:
                current = self.runner._dynamic_memory_vmm_raw[:self.diagnostic_bytes]
                if not torch.equal(self.verification, current):
                    raise RuntimeError(
                        "Retained KV contents changed during VMM resize")
                self.verification = None
                _empty_cache_for_resize(torch, self.runner, "diagnostic")
                self.record_phase("verify_retained_diagnostic")
            if (self.mla_binding is not None
                    and self.binding_signature(self.runner) != self.mla_binding):
                raise RuntimeError("MLA binding changed during resize")
            weights_unchanged = _weight_identity_matches(
                self.runner, self.weight_signature)
            self.record_phase("inspect_weights_after")
            pointers = [
                (view.data_ptr(), tuple(view.shape), view.stride())
                for view in self.runner._dynamic_memory_vmm_views.values()
            ]
            if not weights_unchanged or self.pointer_signature != pointers:
                raise RuntimeError("VMM changed model or KV view identity")
            if (self.runner.input_batch is not self.input_batch
                    or self.runner.requests is not self.request_states):
                raise RuntimeError("Request state changed during VMM resize")
            from .worker import _publish_worker_capacity
            _publish_worker_capacity(
                self.worker, self.capacity.new, self.capacity.new_bytes)
            self.record_phase("publish_worker")
            after = _memory(torch, self.worker.device)
            after["external_kv_bytes"] = self.capacity.new_bytes
        except Exception as exc:
            self.verification = None
            return {"outcome": "failed", "error": f"{type(exc).__name__}: {exc}",
                    "phase_spans": self.spans, "vmm": self.arena.diagnostics()}
        return self.result(retained_verified, after)

    def result(self, retained_verified, after):
        return {
            "outcome": "applied", "num_blocks": self.capacity.new.num_blocks,
            "allocated_kv_bytes": self.capacity.new_bytes, "weights_unchanged": True,
            "live_kv_preserved": True,
            "active_worker_requests": len(self.runner.requests), "copied_bytes": 0,
            "retained_kv_bitwise_verified": retained_verified,
            "diagnostic_snapshot_bytes": (
                self.diagnostic_bytes if retained_verified else 0),
            "kv_pointers_shapes_strides_unchanged": True,
            "input_batch_unchanged": True, "request_state_unchanged": True,
            "allocator": "vmm", "before": self.before, "after": after,
            "worker_pid": os.getpid(), "worker_tid": get_native_id(),
            "start_monotonic_ns": self.started, "phase_spans": self.spans,
            "vmm": self.arena.diagnostics(),
            "timings_seconds": {
                span["name"]: (
                    span["end_monotonic_ns"] - span["start_monotonic_ns"]) / 1e9
                for span in self.spans
            },
        }


def resize_vmm(worker, target_bytes, expected_blocks):
    import torch

    operation = _VmmResizeOperation(worker, target_bytes, expected_blocks)
    for phase in (operation.plan, operation.inspect_bindings):
        error = phase()
        if error is not None:
            return error
    error = operation.preflight(torch)
    if error is not None:
        return error
    error = operation.snapshot_diagnostic(torch)
    if error is not None:
        return error
    error = operation.resize_storage()
    if error is not None:
        return error
    return operation.verify_and_publish(torch)
