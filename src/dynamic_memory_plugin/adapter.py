"""Pinned scheduler/worker adapter for reviewed vLLM 0.32.0 and 0.33.0."""

import math
import os
from copy import deepcopy
from dataclasses import dataclass
from .control import ControlError
from .config_issues import ConfigIssue, ConfigIssueCode as IssueCode

EXTENSION = "dynamic_memory_plugin.worker.DynamicMemoryWorker"


@dataclass(frozen=True)
class CacheCapacityPlan:
    """The old and new worker cache capacities for one resize."""

    old: object
    new: object
    old_bytes: int
    new_bytes: int


def execution_headroom_bytes():
    return int(os.environ.get(
        "VLLM_DYNAMIC_MEMORY_HEADROOM_BYTES", str(256 * 1024**2)))


def vmm_enabled():
    value = os.environ.get("VLLM_DYNAMIC_MEMORY_ALLOCATOR", "native")
    if value not in {"native", "vmm"}:
        raise ValueError("VLLM_DYNAMIC_MEMORY_ALLOCATOR must be native or vmm")
    return value == "vmm"


def chunk_bytes():
    value = int(os.environ.get("VLLM_DYNAMIC_MEMORY_CHUNK_BYTES", str(14 * 1024**2)))
    if value <= 0:
        raise ValueError("VMM chunk bytes must be positive")
    return value


def shrink_policy():
    value = os.environ.get("VLLM_DYNAMIC_MEMORY_SHRINK_POLICY", "wait_for_capacity")
    if value not in {"wait_for_capacity", "immediate"}:
        raise ValueError(
            "VLLM_DYNAMIC_MEMORY_SHRINK_POLICY must be wait_for_capacity or immediate"
        )
    return value


def _execution_issues(config):
    issues = []
    try:
        shrink_policy()
    except ValueError as exc:
        issues.append(ConfigIssue(IssueCode.INVALID_SHRINK_POLICY, detail=str(exc)))
    parallel = config.parallel_config
    if getattr(parallel, "enable_fault_tolerance", False):
        issues.append(ConfigIssue(IssueCode.FAULT_TOLERANCE))
    executor = getattr(parallel, "distributed_executor_backend", "uni")
    if executor != "uni":
        issues.append(ConfigIssue(IssueCode.EXECUTOR, actual=executor, expected="uni"))
    for name in ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size",
                 "decode_context_parallel_size", "prefill_context_parallel_size"):
        size = getattr(parallel, name, 1)
        if size != 1:
            issues.append(ConfigIssue(
                IssueCode.PARALLEL_SIZE, field=name, actual=size, expected=1))
    if parallel.worker_extension_cls != EXTENSION:
        issues.append(ConfigIssue(IssueCode.WORKER_EXTENSION, expected=EXTENSION))
    if config.use_v2_model_runner:
        issues.append(ConfigIssue(IssueCode.MODEL_RUNNER))
    model = config.model_config
    compilation = config.compilation_config
    if not model.enforce_eager or compilation.cudagraph_mode.name != "NONE":
        issues.append(ConfigIssue(IssueCode.EAGER_EXECUTION))
    if getattr(compilation.mode, "value", compilation.mode) != 0:
        issues.append(ConfigIssue(IssueCode.COMPILATION))
    return issues


def _model_issues(config):
    # Native loading owns model validity; resize owns its live storage contract.
    issues = []
    model = config.model_config
    parallel = config.parallel_config
    if getattr(model.hf_config, "is_encoder_decoder", False):
        issues.append(ConfigIssue(IssueCode.ENCODER_DECODER))
    for name in ("enable_expert_parallel", "enable_eplb"):
        if getattr(parallel, name, False):
            issues.append(ConfigIssue(IssueCode.EXPERT_PARALLELISM, field=name))
    if getattr(model, "use_mla", False):
        from .mla import validate_profile
        try:
            validate_profile(config)
        except ValueError as exc:
            issues.append(ConfigIssue(IssueCode.MLA_PROFILE, detail=str(exc)))
    return issues


def _allocator_issues(config):
    issues = []
    model = config.model_config
    cache = config.cache_config
    if vmm_enabled():
        from vllm.v1.kv_cache_layout import KVCacheLayout
        if cache.get_resolved_kv_cache_layout() is not KVCacheLayout.BLNHC:
            issues.append(ConfigIssue(IssueCode.VMM_LAYOUT))
        backend = config.attention_config.backend
        name = getattr(backend, "name", backend)
        expected = "TRITON_MLA" if getattr(model, "use_mla", False) else "FLASH_ATTN"
        if name is not None and name != expected:
            issues.append(ConfigIssue(
                IssueCode.VMM_BACKEND, actual=name, expected=expected))
    if (getattr(model, "enable_sleep_mode", False)
            or getattr(model, "enable_cumem_allocator", False)):
        issues.append(ConfigIssue(IssueCode.ALLOCATOR_POOLS))
    return issues


def _serving_feature_issues(config):
    issues = []
    scheduler = config.scheduler_config
    if scheduler.async_scheduling or getattr(config.parallel_config, "use_ubatching", False):
        issues.append(ConfigIssue(IssueCode.ASYNC_SCHEDULING))
    if scheduler.scheduler_cls is not None:
        issues.append(ConfigIssue(IssueCode.CUSTOM_SCHEDULER))
    for name in ("speculative_config", "kv_transfer_config", "ec_transfer_config", "lora_config"):
        if getattr(config, name, None) is not None:
            issues.append(ConfigIssue(IssueCode.UNSUPPORTED_FEATURE, field=name))
    return issues


def _cache_issues(config):
    issues = []
    cache = config.cache_config
    if cache.num_gpu_blocks_override is not None:
        issues.append(ConfigIssue(
            IssueCode.BLOCK_OVERRIDE, actual=cache.num_gpu_blocks_override))
    if cache.cache_dtype not in {"auto", "bfloat16", "float16"}:
        issues.append(ConfigIssue(
            IssueCode.UNSUPPORTED_CACHE_DTYPE, actual=cache.cache_dtype))
    if getattr(cache, "kv_sharing_fast_prefill", False):
        issues.append(ConfigIssue(IssueCode.KV_SHARING))
    events = getattr(config, "kv_events_config", None)
    if events and events.enable_kv_cache_events:
        issues.append(ConfigIssue(IssueCode.KV_EVENTS))
    observability = getattr(config, "observability_config", None)
    if getattr(observability, "kv_cache_metrics", False):
        issues.append(ConfigIssue(IssueCode.KV_METRICS))
    return issues


def _mutation_environment_issues(config):
    issues = _allocator_issues(config)
    issues += _serving_feature_issues(config)
    issues += _cache_issues(config)
    offload = getattr(config, "offload_config", None)
    if offload and (offload.uva.cpu_offload_gb or offload.prefetch.offload_group_size):
        issues.append(ConfigIssue(IssueCode.WEIGHT_OFFLOAD))
    if config.device_config.device_type != "cuda":
        issues.append(ConfigIssue(
            IssueCode.DEVICE, actual=config.device_config.device_type, expected="cuda"))
    return issues


def configuration_issues(config):
    """Collect supported checks in order; malformed settings still raise.

    Codes and details are available to in-process callers. The existing
    runtime/API boundary continues to use validate_config's ValueError message.
    """
    from .safety import refuse_dev_mode
    refuse_dev_mode()
    return tuple(_execution_issues(config) + _model_issues(config)
                 + _mutation_environment_issues(config))


def validate_config(config):
    issues = configuration_issues(config)
    if issues:
        raise ValueError(
            "Unsupported configuration: " + "; ".join(issue.message for issue in issues))


def backing_bytes(kv_config):
    sizes = {tensor.size for tensor in kv_config.kv_cache_tensors}
    if len(sizes) != 1:
        raise ValueError("Expected one shared KV backing allocation")
    return sizes.pop()



def validate_cache_spec(spec):
    from .storage_contract import validate_spec
    return validate_spec(spec, allow_mla=vmm_enabled())


def cache_geometry(kv_config):
    """Expose actual native scheduler metadata for qualification/diagnostics."""
    if len(kv_config.kv_cache_groups) != 1:
        raise ValueError('Exactly one native KV group is required')
    group = kv_config.kv_cache_groups[0]
    spec = group.kv_cache_spec
    kind = validate_cache_spec(spec)
    layout = kv_config.kv_cache_layout
    return {
        "contract": kind,
        "layout": getattr(layout, "name", layout),
        "layers": len(group.layer_names),
        "spec_type": type(spec).__name__,
        "block_size": spec.block_size,
        "num_kv_heads": spec.num_kv_heads,
        "head_size": spec.head_size,
        "head_size_v": spec.head_size_v,
        "dtype": str(spec.dtype).removeprefix("torch."),
        "layer_page_bytes": spec.page_size_bytes,
        "packed_block_bytes": backing_bytes(kv_config) // kv_config.num_blocks,
    }

def plan_cache(config, old, target_bytes):
    from .runtime_profile import plan_kv_cache

    if len(old.kv_cache_groups) != 1:
        raise ValueError("Exactly one uniform FullAttentionSpec group is supported")
    validate_cache_spec(old.kv_cache_groups[0].kv_cache_spec)
    from .storage_contract import validate_allocation
    validate_allocation(old, vmm=vmm_enabled())
    if vmm_enabled():
        target_bytes = target_bytes // chunk_bytes() * chunk_bytes()
    planned = plan_kv_cache(config, deepcopy(old.kv_cache_groups), target_bytes)
    planned.kv_cache_layout = old.kv_cache_layout
    spec = old.kv_cache_groups[0].kv_cache_spec
    usable = planned.num_blocks - 1  # block zero is permanently reserved
    required = math.ceil(config.model_config.max_model_len / spec.block_size)
    if usable < required:
        raise ControlError("invalid_request", f"Target has {usable} usable blocks; max_model_len needs {required}")
    validate_allocation(planned, vmm=vmm_enabled())
    if backing_bytes(planned) > target_bytes:
        raise ValueError("Planner exceeded requested bytes")
    return planned


class EngineBackend:
    def __init__(self, engine):
        self.engine = engine

    def fence(self, error):
        from .lifecycle import fence_engine
        fence_engine(self.engine, error)

    def capacity(self):
        from .withdrawal import referenced_block_count
        cfg = self.engine.scheduler.kv_cache_config
        pool = self.engine.scheduler.kv_cache_manager.block_pool
        target = getattr(pool, "retire_target", None)
        pending_target = getattr(self, "preserve_target_blocks", None)
        referenced = referenced_block_count(pool)
        return {
            "allocated_kv_bytes": backing_bytes(cfg),
            "num_blocks": cfg.num_blocks,
            "kv_cache_geometry": cache_geometry(cfg),
            "allocator": "vmm" if vmm_enabled() else "native",
            "shrink_policy": shrink_policy(),
            "resize_granularity_bytes": chunk_bytes() if vmm_enabled() else None,
            "free_blocks": pool.get_num_free_blocks(),
            "paused": self.engine.is_scheduler_paused(),
            "retiring_blocks": (
                0 if target is None else pool.num_gpu_blocks - target),
            "retiring_referenced_blocks": (
                0 if target is None
                else sum(block.ref_cnt > 0 for block in pool.blocks[target:])),
            "withdrawal_stage": getattr(self, "withdrawal_stage", None),
            "pending_target_blocks": pending_target,
            "pending_shrink_blocks": (
                None if pending_target is None
                else pool.num_gpu_blocks - pending_target),
            "referenced_blocks": referenced,
            "referenced_blocks_over_target": (
                None if pending_target is None
                else max(0, referenced - pending_target)),
        }

    def prepare(self, target_bytes, strategy="drain"):
        import os
        from vllm.v1.core.kv_cache_manager import KVCacheManager
        e, s = self.engine, self.engine.scheduler
        validate_config(e.vllm_config)
        if e.is_scheduler_paused() or e.model_executor.is_sleeping:
            raise ControlError("operation_conflict", "Engine already paused/asleep; resize must own its pause")
        maximum = int(os.environ.get("VLLM_DYNAMIC_MEMORY_MAX_KV_BYTES", str(1024**3)))
        if target_bytes > maximum:
            raise ControlError("invalid_request", f"Target exceeds configured prototype ceiling {maximum}")
        old = deepcopy(s.kv_cache_config)
        new = plan_cache(e.vllm_config, old, target_bytes)
        def manager(cfg):
            return KVCacheManager(
                kv_cache_config=cfg, max_model_len=s.max_model_len,
                max_in_flight_tokens=e.vllm_config.max_in_flight_tokens,
                enable_caching=s.cache_config.enable_prefix_caching,
                use_eagle=s.use_eagle_block_drop, num_prefill_lookahead=s.num_prefill_lookahead,
                log_stats=s.log_stats, enable_kv_cache_events=False,
                dcp_world_size=1, pcp_world_size=1, scheduler_block_size=s.block_size,
                hash_block_size=s.hash_block_size, metrics_collector=s.kv_metrics_collector,
                watermark=s.scheduler_config.watermark,
                enable_mamba_shared_prefix_checkpoint=False,
            )
        plan = {"old": old, "new": new, "old_bytes": backing_bytes(old),
                "new_bytes": backing_bytes(new), "requested_bytes": target_bytes}
        if strategy == "drain" and new.num_blocks != old.num_blocks:
            plan.update(old_manager=manager(old), new_manager=manager(new))
        return plan

    def pause(self):
        return self.engine.pause_scheduler(mode="wait", clear_cache=False)

    def scheduling_paused(self):
        # PAUSED_NEW still permits already-running requests to step. Recovery
        # is confirmed only when every scheduling path is stopped.
        from vllm.v1.core.sched.interface import PauseState
        return self.engine.scheduler.pause_state == PauseState.PAUSED_ALL

    def ensure_paused(self):
        """Stop scheduling best-effort and return any nonfatal pause error.

        Multiprocess ``keep`` mode sets PAUSED_ALL synchronously, even when its
        returned Future is still waiting for output delivery. Confirmation here
        is therefore about scheduling, not completion of that output flush.
        """
        error = None
        try:
            self.engine.pause_scheduler(mode="keep", clear_cache=False)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        try:
            paused = self.scheduling_paused()
        except Exception as exc:
            raise RuntimeError(f"could not inspect scheduler pause state: {exc}") from exc
        if not paused:
            detail = "" if error is None else f" after {error}"
            raise RuntimeError("scheduling pause could not be confirmed" + detail)
        return error

    def assert_drained(self):
        e, s = self.engine, self.engine.scheduler
        if not e.is_scheduler_paused() or s.running or e.batch_queue:
            raise RuntimeError("Pause has not drained active work")
        if s.has_finished_requests() or s.deferred_frees:
            raise RuntimeError("Worker cleanup or deferred block frees still pending")
        # has_requests() hides waiting requests during PAUSED_NEW. Check actual
        # block references; keep waiting queues and the scheduler object intact.
        if any(b.ref_cnt for b in s.kv_cache_manager.block_pool.blocks if not b.is_null):
            raise RuntimeError("Referenced KV remains after drain; cannot discard it")
        if any(r.num_computed_tokens for r in s.requests.values()):
            raise RuntimeError("Waiting requests still depend on computed KV")

    def replace(self, plan):
        return self._dispatch_resize("dynamic_memory_replace", plan)

    def _dispatch_resize(self, method, plan):
        from .worker import _authorize_resize
        executor = self.engine.model_executor
        worker = self._local_worker()
        args = (plan["requested_bytes"], plan["new"].num_blocks)
        kwargs = {"expected_old_bytes": plan["old_bytes"],
                  "expected_old_blocks": plan["old"].num_blocks}
        # The authorization is a Python object identity, never a public RPC.
        # UniProc dispatch is synchronous and preserves that identity. A JSON
        # copy of the same operation IDs cannot authorize worker mutation.
        with _authorize_resize(worker, method, *args, **kwargs,
                               operation_context=plan["operation_context"]) as context:
            result = executor.collective_rpc(method, args=args,
                                            kwargs={**kwargs, "operation_context": context})
        if len(result) != 1:
            raise RuntimeError("Expected exactly one worker result")
        return result[0]

    def _local_worker(self):
        from vllm.v1.executor.uniproc_executor import UniProcExecutor
        executor = self.engine.model_executor
        if type(executor) is not UniProcExecutor:
            raise RuntimeError("Resize authorization requires the pinned UniProcExecutor")
        return executor.driver_worker.worker

    def publish(self, plan, rollback):
        from vllm.v1.core.kv_cache_utils import update_kv_cache_capacity
        s = self.engine.scheduler
        cfg = plan["old"] if rollback else plan["new"]
        s.kv_cache_manager = plan["old_manager"] if rollback else plan["new_manager"]
        s.kv_cache_config = cfg
        s.cache_config.num_gpu_blocks = cfg.num_blocks
        s.cache_config.kv_cache_memory_bytes = backing_bytes(cfg)
        update_kv_cache_capacity(self.engine.vllm_config, cfg)
        s.prev_step_scheduled_req_ids.clear()

    def resume(self):
        from vllm.v1.engine import EngineCoreRequestType
        self.engine.resume_scheduler()
        # A drain callback runs inside _process_input_queue's idle branch.
        # That branch next blocks on get(), even if resume made queued
        # requests runnable. Wake it explicitly; do not require another HTTP
        # request or a status poll to restart inference.
        self.engine.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, None))

    def begin_preserving(self, plan, callback):
        from .withdrawal import WithdrawingBlockPool
        e = self.engine
        if e.batch_queue:
            raise ValueError("Outstanding batches are unsupported for preserving resize")
        pool = e.scheduler.kv_cache_manager.block_pool
        target = plan["new"].num_blocks
        self.preserve_target_blocks = target if target < pool.num_gpu_blocks else None
        self.withdrawal_stage = (
            "waiting_for_capacity" if self.preserve_target_blocks is not None else None
        )
        if self.preserve_target_blocks is None:
            WithdrawingBlockPool.adopt(pool)
        elif shrink_policy() == "immediate":
            pool = WithdrawingBlockPool.adopt(pool)
            pool.begin_withdrawal(target)
            self.withdrawal_stage = "withdrawing_tail"
        # The synchronous engine's step returns after its model result. Check
        # withdrawal progress here, keeping the normal scheduler serving work.
        self.original_step = e.step_fn
        def step():
            result = self.original_step()
            callback()
            from .lifecycle import check_fence
            check_fence(e)
            return result
        self.wrapped_step = step
        e.step_fn = step

    def preserving_ready(self):
        from .withdrawal import WithdrawingBlockPool, referenced_block_count
        e = self.engine
        # Scheduler completion frees blocks before the worker consumes its
        # finished_req_ids on the next step. Keep serving until that cleanup
        # step has removed stale worker block tables, even if the tail is free.
        # has_requests() includes finished requests, so this also advances when
        # the last user finishes; no new HTTP traffic or polling is needed.
        if e.batch_queue or e.scheduler.deferred_frees or e.scheduler.has_finished_requests():
            return False
        pool = e.scheduler.kv_cache_manager.block_pool
        if getattr(self, "withdrawal_stage", None) == "waiting_for_capacity":
            target = self.preserve_target_blocks
            if referenced_block_count(pool) > target:
                return False
            pool = WithdrawingBlockPool.adopt(pool)
            pool.begin_withdrawal(target)
            self.withdrawal_stage = "withdrawing_tail"
        if not pool.ready():
            return False
        if self.preserve_target_blocks is not None:
            self._assert_tail_unreferenced(self.preserve_target_blocks)
            # Preempted requests also retain old worker tables until resumed
            # or finished. Inspect Python metadata only, without an RPC/CUDA
            # query. Wait for its normal refresh, preserving the deadline and
            # cancellation semantics instead of falsely rejecting the resize.
            from .worker import _assert_shrink_references, WorkerReferencesPending
            try:
                _assert_shrink_references(self._local_worker().model_runner,
                                          self.preserve_target_blocks)
            except WorkerReferencesPending:
                return False
        return True

    def replace_preserving(self, plan):
        self._assert_shrink_references(plan)
        return self._dispatch_resize("dynamic_memory_preserve", plan)

    def _assert_shrink_references(self, plan):
        target = plan["new"].num_blocks
        if target >= plan["old"].num_blocks:
            return
        self._assert_tail_unreferenced(target)

    def _assert_tail_unreferenced(self, target):
        manager = self.engine.scheduler.kv_cache_manager
        for group in manager.coordinator.single_type_managers:
            references = (block for blocks in group.req_to_blocks.values() for block in blocks)
            pending = (block for pair in group._pending_cow_copies for block in pair)
            partial = (block for _, block in group._partial_hit_reqs.values())
            from itertools import chain
            if any(not 0 <= block.block_id < target for block in chain(references, pending, partial)):
                raise RuntimeError("Scheduler still references the shrinking KV tail")

    def publish_preserving(self, plan):
        from vllm.v1.core.kv_cache_utils import update_kv_cache_capacity
        self._assert_shrink_references(plan)
        s = self.engine.scheduler
        manager, cfg = s.kv_cache_manager, plan["new"]
        manager.block_pool.commit_capacity(cfg.num_blocks)
        manager.kv_cache_config = manager.coordinator.kv_cache_config = s.kv_cache_config = cfg
        manager.watermark_blocks = int(s.scheduler_config.watermark * cfg.num_blocks)
        s.cache_config.num_gpu_blocks = cfg.num_blocks
        s.cache_config.kv_cache_memory_bytes = backing_bytes(cfg)
        update_kv_cache_capacity(self.engine.vllm_config, cfg)

    def end_preserving(self, cancel=False, failed=False):
        if hasattr(self, "wrapped_step") and self.engine.step_fn is self.wrapped_step:
            self.engine.step_fn = self.original_step
        if failed:
            return self.ensure_paused()
        else:
            pool = self.engine.scheduler.kv_cache_manager.block_pool
            if hasattr(pool, "release_adapter"):
                if cancel:
                    pool.cancel_withdrawal()
                pool.release_adapter()
            if hasattr(self, "preserve_target_blocks"):
                del self.preserve_target_blocks
            if hasattr(self, "withdrawal_stage"):
                del self.withdrawal_stage
