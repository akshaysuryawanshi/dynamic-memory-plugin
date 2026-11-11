"""Worker RPC extension for drain or synchronous iteration-boundary resizing."""

import gc
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass

from .adapter import (
    CacheCapacityPlan,
    backing_bytes,
    execution_headroom_bytes,
    plan_cache,
    validate_config,
    vmm_enabled,
)


def _weights(runner):
    return [(name, p.data_ptr(), tuple(p.shape), str(p.dtype))
            for name, p in runner.model.named_parameters()]


def _snapshot_weight_identity(runner):
    """Capture this operation's model bindings and parameter storage metadata.

    This is deliberately not a persistent cache. Each resize inspects the
    current module tree, including any replacement made between operations.
    Storage identity checks do not compare parameter values.
    """
    model = runner.model
    parameters = tuple((name, parameter, parameter.data_ptr(), parameter.shape, parameter.dtype)
                       for name, parameter in model.named_parameters())
    return model, parameters


def _weight_identity_matches(runner, snapshot):
    """Verify current bindings incrementally without a second signature list."""
    model, expected = snapshot
    if runner.model is not model:
        return False
    current = iter(model.named_parameters())
    for name, parameter, pointer, shape, dtype in expected:
        entry = next(current, None)
        if entry is None:
            return False
        current_name, current_parameter = entry
        if (current_name != name or current_parameter is not parameter
                or current_parameter.data_ptr() != pointer
                or current_parameter.shape != shape or current_parameter.dtype != dtype):
            return False
    return next(current, None) is None


def _memory(torch, device):
    free, total = torch.cuda.mem_get_info(device)
    return {"driver_free_bytes": free, "driver_total_bytes": total,
            "torch_allocated_bytes": torch.cuda.memory_allocated(device),
            "torch_reserved_bytes": torch.cuda.memory_reserved(device)}


def _empty_cache_for_resize(torch, runner, reason):
    """Count existing trim attempts so observation cannot hide allocator resets."""
    if reason not in {"preflight", "diagnostic"}:
        raise ValueError("Unknown VMM empty-cache reason")
    counters = getattr(runner, "_dynamic_memory_vmm_resize_counters", None)
    if counters is None:
        counters = {}
        runner._dynamic_memory_vmm_resize_counters = counters
    key = reason + "_empty_cache_attempts"
    counters[key] = counters.get(key, 0) + 1
    torch.cuda.empty_cache()


class ResizeAuthorizationError(ValueError):
    pass


@dataclass(frozen=True)
class _ResizeRequest:
    method: str
    target_bytes: int
    expected_blocks: int
    expected_old_bytes: int
    expected_old_blocks: int


@dataclass(frozen=True)
class _ResizeAuthorization:
    request: _ResizeRequest
    context: dict


@contextmanager
def _authorize_resize(worker, method, target_bytes, expected_blocks, *,
                      expected_old_bytes, expected_old_blocks, operation_context):
    """Arm one synchronous in-process call; not exposed on the worker class."""
    if "_dynamic_memory_resize_authorization" in vars(worker):
        raise RuntimeError("Overlapping worker resize authorization")
    request = _ResizeRequest(
        method=method, target_bytes=target_bytes, expected_blocks=expected_blocks,
        expected_old_bytes=expected_old_bytes, expected_old_blocks=expected_old_blocks)
    context = dict(operation_context)
    worker._dynamic_memory_resize_authorization = _ResizeAuthorization(request, context)
    try:
        yield context
    finally:
        vars(worker).pop("_dynamic_memory_resize_authorization", None)


def _consume_resize_authorization(worker, request, operation_context):
    """Consume the permit even when authorization or later validation fails."""
    permit = vars(worker).pop("_dynamic_memory_resize_authorization", None)
    if (permit is None or permit.context is not operation_context
            or permit.request != request):
        raise ResizeAuthorizationError("Worker resize requires engine-owned single-use authorization")


def _is_positive_integer(value):
    # bool is an int subclass, but is never a valid capacity.
    return type(value) is int and value > 0


def _validate_rpc_preconditions(worker, request, operation_context):
    """Check an authorized request without consuming or changing worker state."""
    if not (_is_positive_integer(request.target_bytes)
            and _is_positive_integer(request.expected_blocks)):
        raise ValueError("Worker resize target must contain positive integers")

    old = worker.model_runner.kv_cache_config
    valid_old_capacity = (
        _is_positive_integer(request.expected_old_bytes)
        and _is_positive_integer(request.expected_old_blocks))
    if (not valid_old_capacity
            or backing_bytes(old) != request.expected_old_bytes
            or old.num_blocks != request.expected_old_blocks):
        raise RuntimeError("Scheduler and worker old KV capacities disagree")

    return _validated_operation_identity(operation_context)


def _validated_operation_identity(context):
    """Return only the three validated identity fields, detached from the caller."""
    if not isinstance(context, dict):
        raise ValueError("Operation context must be a dictionary")
    controller_id = context.get("controller_id")
    operation_id = context.get("operation_id")
    sequence = context.get("sequence")
    valid_ids = all(
        isinstance(value, str) and 1 <= len(value) <= 128
        for value in (controller_id, operation_id))
    valid_sequence = type(sequence) is int and sequence > 0
    if not valid_ids or not valid_sequence:
        raise ValueError(
            "Operation context must contain bounded IDs and a positive sequence")
    return {"controller_id": controller_id, "operation_id": operation_id,
            "sequence": sequence}


class WorkerReferencesPending(ValueError):
    """Cached worker request tables still refer to the retiring tail."""


def _assert_shrink_references(runner, expected_blocks):
    if expected_blocks >= runner.kv_cache_config.num_blocks:
        return
    references_tail = any(
        not 0 <= block_id < expected_blocks
        for request in runner.requests.values()
        for group in request.block_ids
        for block_id in group
    )
    if references_tail:
        raise WorkerReferencesPending("Worker still references the shrinking KV tail")


def _publish_worker_capacity(worker, config, allocated_bytes):
    """Publish a planned capacity to the worker-owned cache fields."""
    worker.model_runner.kv_cache_config = config
    worker.cache_config.num_gpu_blocks = config.num_blocks
    worker.cache_config.kv_cache_memory_bytes = allocated_bytes


def _attach_operation_context(result, context):
    if context is not None:
        result["operation_context"] = dict(context)
        for span in result.get("phase_spans", ()):
            span.update(context)
    return result


def _validated_rpc_context(worker, method, target_bytes, expected_blocks,
                           expected_old_bytes, expected_old_blocks,
                           operation_context):
    try:
        request = _ResizeRequest(
            method=method, target_bytes=target_bytes, expected_blocks=expected_blocks,
            expected_old_bytes=expected_old_bytes, expected_old_blocks=expected_old_blocks)
        _consume_resize_authorization(worker, request, operation_context)
        context = _validate_rpc_preconditions(worker, request, operation_context)
        return context, None
    except ResizeAuthorizationError as exc:
        return None, {"outcome": "rejected", "error": str(exc)}
    except Exception as exc:
        return None, {"outcome": "failed",
                      "error": f"{type(exc).__name__}: {exc}"}


def _replace_allocation(worker, config):
    """Return scalar failure metadata, never a traceback owning cache tensors."""
    import torch
    from vllm.config import set_current_vllm_config
    phase = "cleanup"
    try:
        with torch.inference_mode(), set_current_vllm_config(worker.vllm_config):
            worker.model_runner._cleanup_profiling_kv_cache()
            # The upstream helper's last local layer view can survive until it
            # returns. Trim again here, after those local references are gone.
            gc.collect()
            phase = "trim"
            torch.cuda.empty_cache()
            phase = "allocation"
            worker.initialize_from_config(config)
            phase = "allocation_fence"
            torch.cuda.synchronize(worker.device)
        return None
    except Exception as exc:
        # A later successful fence cannot clear evidence of a previous async
        # CUDA error. Only the explicit allocation-capacity exception permits
        # rollback to establish a usable fresh cache.
        return {"error": f"{type(exc).__name__}: {exc}", "phase": phase,
                "unsafe": phase != "allocation" or not isinstance(exc, torch.OutOfMemoryError)}


class DynamicMemoryWorker:
    def dynamic_memory_preserve(self, target_bytes: int, expected_blocks: int, *,
                                expected_old_bytes: int, expected_old_blocks: int,
                                operation_context):
        context, error = _validated_rpc_context(
            self, "dynamic_memory_preserve", target_bytes, expected_blocks,
            expected_old_bytes, expected_old_blocks, operation_context)
        if error is not None:
            return error
        if vmm_enabled():
            from .vmm_integration import resize_vmm
            result = resize_vmm(self, target_bytes, expected_blocks)
        else:
            from .preserve import replace_preserving
            result = replace_preserving(self, target_bytes, expected_blocks)
        return _attach_operation_context(result, context)

    def dynamic_memory_replace(self, target_bytes: int, expected_blocks: int, *,
                               expected_old_bytes: int, expected_old_blocks: int,
                               operation_context):
        context, error = _validated_rpc_context(
            self, "dynamic_memory_replace", target_bytes, expected_blocks,
            expected_old_bytes, expected_old_blocks, operation_context)
        if error is not None:
            return error
        result = _replace(self, target_bytes, expected_blocks)
        return _attach_operation_context(result, context)


class _NativeReplaceOperation:
    """One drained native-allocator replacement and its fresh-cache rollback."""

    def __init__(self, worker, target_bytes, expected_blocks):
        self.worker = worker
        self.runner = worker.model_runner
        self.target_bytes = target_bytes
        self.expected_blocks = expected_blocks

    def plan(self):
        try:
            validate_config(self.worker.vllm_config)
            if self.runner.input_batch.num_reqs or self.runner.requests:
                raise RuntimeError("Worker request state has not drained")
            old = deepcopy(self.runner.kv_cache_config)
            new = plan_cache(
                self.worker.vllm_config, old, self.target_bytes)
            if new.num_blocks != self.expected_blocks:
                raise RuntimeError("Worker and scheduler resize plans disagree")
            self.reserve = execution_headroom_bytes()
            if self.reserve < 0:
                raise ValueError(
                    "VLLM_DYNAMIC_MEMORY_HEADROOM_BYTES must be nonnegative")
            self.capacity = CacheCapacityPlan(
                old, new, backing_bytes(old), backing_bytes(new))
            return None
        except Exception as exc:
            return {"outcome": "rejected",
                    "error": f"{type(exc).__name__}: {exc}"}

    def preflight(self, torch):
        try:
            torch.cuda.synchronize(self.worker.device)
            gc.collect()
            torch.cuda.empty_cache()
            self.before = _memory(torch, self.worker.device)
        except Exception as exc:
            return {"outcome": "failed",
                    "error": f"CUDA preflight failed: {type(exc).__name__}: {exc}"}
        growth = self.capacity.new_bytes - self.capacity.old_bytes
        if growth > self.before["driver_free_bytes"] - self.reserve:
            return {"outcome": "rejected",
                    "error": "Insufficient free VRAM after execution headroom reserve"}
        return None

    def inspect_weights(self):
        try:
            self.signature = _weights(self.runner)
            return None
        except Exception as exc:
            return {"outcome": "failed",
                    "error": f"Weight inspection failed: {type(exc).__name__}: {exc}"}

    def execute(self, torch):
        failure = _replace_allocation(self.worker, self.capacity.new)
        if failure is None and _weights(self.runner) != self.signature:
            failure = {
                "error": "Model parameter pointers changed during KV resize",
                "phase": "verify_identity", "unsafe": True,
            }
        if failure is None:
            self.worker.cache_config.kv_cache_memory_bytes = self.capacity.new_bytes
            return {"outcome": "applied", "num_blocks": self.capacity.new.num_blocks,
                    "allocated_kv_bytes": self.capacity.new_bytes,
                    "weights_unchanged": True, "before": self.before,
                    "after": _memory(torch, self.worker.device)}

        rollback_failure = _replace_allocation(self.worker, self.capacity.old)
        rollback_error = (
            rollback_failure["error"] if rollback_failure is not None else None)
        if rollback_error is None and _weights(self.runner) != self.signature:
            rollback_error = "Model parameter identity changed; recovery unsafe"
        if failure["unsafe"] or rollback_error is not None:
            return {"outcome": "failed", "error": failure["error"],
                    "failure_phase": failure["phase"],
                    "rollback_error": rollback_error,
                    "rollback_completed": rollback_error is None,
                    "before": self.before}
        self.worker.cache_config.kv_cache_memory_bytes = self.capacity.old_bytes
        return {"outcome": "rolled_back", "error": failure["error"],
                "allocated_kv_bytes": self.capacity.old_bytes,
                "num_blocks": self.capacity.old.num_blocks, "before": self.before,
                "after": _memory(torch, self.worker.device)}


def _replace(self, target_bytes: int, expected_blocks: int):
    import torch

    if vmm_enabled():
        if self.model_runner.input_batch.num_reqs or self.model_runner.requests:
            return {"outcome": "rejected",
                    "error": "Worker request state has not drained"}
        from .vmm_integration import resize_vmm
        return resize_vmm(self, target_bytes, expected_blocks)

    operation = _NativeReplaceOperation(self, target_bytes, expected_blocks)
    for phase in (operation.plan, lambda: operation.preflight(torch),
                  operation.inspect_weights):
        error = phase()
        if error is not None:
            return error
    return operation.execute(torch)
