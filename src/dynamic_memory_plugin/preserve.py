"""Resize native-allocator KV while preserving live data at an eager boundary.

The native path temporarily requires both old and new backing allocations and
does not claim pointer stability. The VMM path is implemented separately.
"""
import gc
import os
from copy import deepcopy
from dataclasses import dataclass
from time import monotonic, monotonic_ns

from .adapter import (
    CacheCapacityPlan,
    backing_bytes,
    execution_headroom_bytes,
    plan_cache,
    validate_config,
)
from .worker import (
    _assert_shrink_references,
    _memory,
    _publish_worker_capacity,
    _weights,
)


def _error(outcome, exc, prefix=None):
    message = f"{type(exc).__name__}: {exc}"
    return {"outcome": outcome, "error": f"{prefix}: {message}" if prefix else message}


@dataclass(frozen=True)
class _PreserveFailure:
    error: str
    unsafe: bool


class _PreserveOperation:
    """One preserve attempt, including the GPU views needed for recovery."""

    def __init__(self, worker, target_bytes, expected_blocks):
        self.worker = worker
        self.runner = worker.model_runner
        self.target_bytes = target_bytes
        self.expected_blocks = expected_blocks
        self.begin = monotonic()
        self.start_ns = monotonic_ns()
        self.input_batch = self.runner.input_batch
        self.request_states = self.runner.requests
        self.active_requests = len(self.request_states)
        self.timings = {}
        self.phase = "allocation"
        self.new_views = {}
        self.changed_bindings = False

    def plan(self):
        try:
            validate_config(self.worker.vllm_config)
            backends = {
                group.backend.get_name()
                for groups in self.runner.attn_groups for group in groups
            }
            if backends not in ({"FLASH_ATTN"}, {"TRITON_ATTN"}):
                raise ValueError(
                    "Preserving resize requires ordinary FLASH_ATTN or TRITON_ATTN")
            old = deepcopy(self.runner.kv_cache_config)
            self.triton = backends == {"TRITON_ATTN"}
            if self.triton:
                from .triton import validate_bindings
                validate_bindings(self.runner, old)
            new = plan_cache(
                self.worker.vllm_config, old, self.target_bytes)
            _assert_shrink_references(self.runner, new.num_blocks)
            spec = old.kv_cache_groups[0].kv_cache_spec
            if (new.num_blocks != self.expected_blocks
                    or self.runner._kernel_block_sizes != [spec.block_size]):
                raise ValueError("Unsupported kernel block geometry or mismatched plan")
            if old.needs_kv_cache_zeroing or self.runner.shared_kv_cache_layers:
                raise ValueError(
                    "KV zeroing and shared-layer metadata require a later adapter")
            self.reserve = execution_headroom_bytes()
            if self.reserve < 0:
                raise ValueError("VLLM_DYNAMIC_MEMORY_HEADROOM_BYTES must be nonnegative")
            self.capacity = CacheCapacityPlan(
                old, new, backing_bytes(old), backing_bytes(new))
            self.names = old.kv_cache_groups[0].layer_names
            return None
        except Exception as exc:
            return _error("rejected", exc)

    def inspect(self):
        try:
            # Existing bindings are a safety invariant, not target validation.
            self.context = self.runner.compilation_config.static_forward_context
            self.old_views = {
                name: self.context[name].kv_cache for name in self.names
            }
            if any(view.ndim != 4 or view.shape[0] != self.capacity.old.num_blocks
                   for view in self.old_views.values()):
                raise ValueError("Unexpected live KV layer view")
            self.signature = _weights(self.runner)
            return None
        except Exception as exc:
            return _error("failed", exc, "Existing state inspection failed")

    def preflight(self, torch):
        try:
            # A CUDA exception here may be a preceding asynchronous fault.
            torch.cuda.synchronize(self.worker.device)
            torch.cuda.empty_cache()
            self.before = _memory(torch, self.worker.device)
        except Exception as exc:
            return _error("failed", exc, "CUDA preflight failed")
        if self.capacity.new_bytes + self.reserve > self.before["driver_free_bytes"]:
            return {"outcome": "rejected", "error":
                    "Preserving resize needs temporary VRAM for the entire new buffer"}
        self.timings["preflight"] = monotonic() - self.begin
        return None

    def allocate_and_copy(self, torch, allocate_kv_cache):
        phase = monotonic()
        self.new_views = allocate_kv_cache(
            self.capacity.new, self.worker.device,
            self.worker.cache_config.get_resolved_kv_cache_layout(),
            self.runner._kernel_block_sizes)
        self.phase = "allocation_fence"
        torch.cuda.synchronize(self.worker.device)
        self.timings["allocate_zero"] = monotonic() - phase

        phase = monotonic()
        retained = min(self.capacity.old.num_blocks, self.capacity.new.num_blocks)
        self.phase = "copy"
        for name in self.names:
            self.new_views[name][:retained].copy_(
                self.old_views[name][:retained])
        self.phase = "copy_fence"
        torch.cuda.synchronize(self.worker.device)
        self.timings["copy"] = monotonic() - phase
        copied_bytes = sum(
            view[:retained].numel() * view.element_size()
            for view in self.old_views.values())

        copy_verified = os.environ.get("DYNAMIC_MEMORY_VERIFY_KV_COPY") == "1"
        if copy_verified:
            phase = monotonic()
            self.phase = "verify_copy"
            for name in self.names:
                new_bits = self.new_views[name][:retained].view(torch.int16)
                old_bits = self.old_views[name][:retained].view(torch.int16)
                if not torch.equal(new_bits, old_bits):
                    raise RuntimeError(f"KV copy verification failed for {name}")
            self.timings["verify_copy"] = monotonic() - phase
        return copied_bytes, copy_verified

    def publish(self, bind_kv_cache):
        phase = monotonic()
        self.phase = "publish_bindings"
        self.changed_bindings = True
        self.runner.kv_caches.clear()
        bind_kv_cache(
            self.new_views, self.context, self.runner.kv_caches,
            kv_cache_groups=self.capacity.new.kv_cache_groups)
        _publish_worker_capacity(
            self.worker, self.capacity.new, self.capacity.new_bytes)
        if self.triton:
            from .triton import validate_bindings
            validate_bindings(self.runner, self.capacity.new)
        self.phase = "verify_identity"
        if (self.runner.input_batch is not self.input_batch
                or self.runner.requests is not self.request_states):
            raise RuntimeError("Request state changed during preserve operation")
        if _weights(self.runner) != self.signature:
            raise RuntimeError("Model parameters changed")
        self.timings["publish_worker"] = monotonic() - phase

    def recover(self, torch, bind_kv_cache, failure):
        try:
            if self.changed_bindings:
                self.runner.kv_caches.clear()
                bind_kv_cache(
                    self.old_views, self.context, self.runner.kv_caches,
                    kv_cache_groups=self.capacity.old.kv_cache_groups)
                _publish_worker_capacity(
                    self.worker, self.capacity.old, self.capacity.old_bytes)
            self.new_views.clear()
            torch.cuda.synchronize(self.worker.device)
            torch.cuda.empty_cache()
        except Exception as recovery:
            return {"outcome": "failed", "error": failure.error,
                    "failure_phase": self.phase,
                    "rollback_error": str(recovery), "cleanup_completed": False}
        if failure.unsafe:
            return {"outcome": "failed", "error": failure.error,
                    "failure_phase": self.phase, "cleanup_completed": True}
        try:
            if (self.runner.input_batch is not self.input_batch
                    or self.runner.requests is not self.request_states):
                raise RuntimeError("Request state identity changed; recovery unsafe")
            if _weights(self.runner) != self.signature:
                raise RuntimeError("Model parameter identity changed; recovery unsafe")
        except Exception as recovery:
            return {"outcome": "failed", "error": failure.error,
                    "rollback_error": f"{type(recovery).__name__}: {recovery}"}
        return {"outcome": "rejected", "error": failure.error,
                "old_storage_preserved": True}

    def execute(self, torch):
        from vllm.config import set_current_vllm_config
        from vllm.v1.worker.utils import allocate_kv_cache, bind_kv_cache

        failure = None
        try:
            with torch.inference_mode(), set_current_vllm_config(
                    self.worker.vllm_config):
                copied_bytes, copy_verified = self.allocate_and_copy(
                    torch, allocate_kv_cache)
                self.publish(bind_kv_cache)
        except Exception as exc:
            unsafe = (self.phase not in {"allocation", "publish_bindings"}
                      or (self.phase == "allocation"
                          and not isinstance(exc, torch.OutOfMemoryError)))
            failure = _PreserveFailure(
                error=f"{type(exc).__name__}: {exc}",
                unsafe=unsafe,
            )
        # The exception and extracted helper frames are gone before rollback.
        if failure is not None:
            return self.recover(torch, bind_kv_cache, failure)

        phase = monotonic()
        self.old_views.clear()
        gc.collect()
        torch.cuda.empty_cache()
        self.timings["release_old"] = monotonic() - phase
        return {"outcome": "applied", "num_blocks": self.capacity.new.num_blocks,
                "allocated_kv_bytes": self.capacity.new_bytes,
                "weights_unchanged": True, "live_kv_preserved": True,
                "active_worker_requests": self.active_requests,
                "copied_bytes": copied_bytes,
                "copy_bitwise_verified": copy_verified,
                "input_batch_unchanged": True, "before": self.before,
                "after": _memory(torch, self.worker.device),
                "timings_seconds": self.timings, "worker_pid": os.getpid(),
                "start_monotonic_ns": self.start_ns}


def replace_preserving(worker, target_bytes, expected_blocks):
    import torch

    operation = _PreserveOperation(worker, target_bytes, expected_blocks)
    for phase in (operation.plan, operation.inspect):
        error = phase()
        if error is not None:
            return error
    error = operation.preflight(torch)
    if error is not None:
        return error
    return operation.execute(torch)
