"""Structured configuration findings and their human-readable presentation."""

from dataclasses import dataclass
from enum import StrEnum


class ConfigIssueCode(StrEnum):
    INVALID_SHRINK_POLICY = "invalid_shrink_policy"
    FAULT_TOLERANCE = "fault_tolerance"
    EXECUTOR = "executor"
    PARALLEL_SIZE = "parallel_size"
    WORKER_EXTENSION = "worker_extension"
    MODEL_RUNNER = "model_runner"
    EAGER_EXECUTION = "eager_execution"
    COMPILATION = "compilation"
    ENCODER_DECODER = "encoder_decoder"
    EXPERT_PARALLELISM = "expert_parallelism"
    MLA_PROFILE = "mla_profile"
    VMM_LAYOUT = "vmm_layout"
    VMM_BACKEND = "vmm_backend"
    ALLOCATOR_POOLS = "allocator_pools"
    ASYNC_SCHEDULING = "async_scheduling"
    CUSTOM_SCHEDULER = "custom_scheduler"
    UNSUPPORTED_FEATURE = "unsupported_feature"
    BLOCK_OVERRIDE = "block_override"
    UNSUPPORTED_CACHE_DTYPE = "unsupported_cache_dtype"
    KV_SHARING = "kv_sharing"
    KV_EVENTS = "kv_events"
    KV_METRICS = "kv_metrics"
    WEIGHT_OFFLOAD = "weight_offload"
    DEVICE = "device"


@dataclass(frozen=True)
class ConfigIssue:
    code: ConfigIssueCode
    field: str = ""
    actual: object = None
    expected: object = None
    detail: str = ""

    @property
    def message(self):
        return _MESSAGES[self.code].format(
            field=self.field, expected=self.expected, detail=self.detail)


# Wording is presentation, never the identifier used to classify a finding.
_MESSAGES = {
    ConfigIssueCode.INVALID_SHRINK_POLICY: "{detail}",
    ConfigIssueCode.FAULT_TOLERANCE: "enable_fault_tolerance conflicts with fatal resize fencing",
    ConfigIssueCode.EXECUTOR: "dynamic memory requires the in-process UniProcExecutor",
    ConfigIssueCode.PARALLEL_SIZE: "{field} must be 1",
    ConfigIssueCode.WORKER_EXTENSION: "set --worker-extension-cls {expected}",
    ConfigIssueCode.MODEL_RUNNER: "V1 model runner required (VLLM_USE_V2_MODEL_RUNNER=0)",
    ConfigIssueCode.EAGER_EXECUTION: "first prototype requires --enforce-eager",
    ConfigIssueCode.COMPILATION: "compilation mode must be NONE",
    ConfigIssueCode.ENCODER_DECODER: "encoder/decoder state needs a resize adapter",
    ConfigIssueCode.EXPERT_PARALLELISM: "{field} changes ownership outside the single-worker contract",
    ConfigIssueCode.MLA_PROFILE: "{detail}",
    ConfigIssueCode.VMM_LAYOUT: "VMM requires BLNHC block-major storage",
    ConfigIssueCode.VMM_BACKEND: "VMM storage adapter requires {expected}",
    ConfigIssueCode.ALLOCATOR_POOLS: "sleep/CuMem pools require a later adapter",
    ConfigIssueCode.ASYNC_SCHEDULING: "disable asynchronous scheduling and ubatching",
    ConfigIssueCode.CUSTOM_SCHEDULER: "custom schedulers are not supported",
    ConfigIssueCode.UNSUPPORTED_FEATURE: "{field} is not supported yet",
    ConfigIssueCode.BLOCK_OVERRIDE: "num_gpu_blocks_override conflicts with byte targets",
    ConfigIssueCode.UNSUPPORTED_CACHE_DTYPE: "unquantized KV cache required initially",
    ConfigIssueCode.KV_SHARING: "KV sharing fast prefill unsupported",
    ConfigIssueCode.KV_EVENTS: "KV events need eviction notifications before support",
    ConfigIssueCode.KV_METRICS: "KV residency metrics need withdrawal-aware accounting",
    ConfigIssueCode.WEIGHT_OFFLOAD: "weight offloading unsupported",
    ConfigIssueCode.DEVICE: "CUDA required for physical resizing",
}
