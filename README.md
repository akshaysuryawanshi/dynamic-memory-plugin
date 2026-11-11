# Dynamic Memory Plugin

Resize a running vLLM engine's KV cache without reloading the model.

- **Drain:** finish active requests, resize, then resume queued requests.
- **Preserve:** retain ongoing generations while resizing at an engine iteration
  boundary. Shrinking waits for retiring blocks to become unreferenced.

## Compatibility

| vLLM | PyTorch | cuda-bindings | Installation extra |
| --- | --- | --- | --- |
| 0.32.0 | 2.13.0 | 13.4.4 | `runtime032` |
| 0.33.0 | 2.14.0 | 13.4.4 | `runtime033` |

Requires Linux, Python 3.11 and one NVIDIA CUDA GPU. Use the V1 model runner,
eager execution, synchronous scheduling and an unquantized KV cache. Startup
checks exact dependency versions and fingerprints of the vLLM internals used.

## Install

```bash
git clone https://github.com/akshaysuryawanshi/dynamic-memory-plugin.git
cd dynamic-memory-plugin
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install '.[runtime032]'
```

Use `'.[runtime033]'` in a separate environment for vLLM 0.33.0.

## Start vLLM

```bash
VLLM_PLUGINS=dynamic_memory,dynamic_memory_api \
VLLM_USE_V2_MODEL_RUNNER=0 \
VLLM_USE_FLASHINFER_SAMPLER=0 \
VLLM_DYNAMIC_MEMORY_ALLOCATOR=native \
VLLM_DYNAMIC_MEMORY_MAX_KV_BYTES=2147483648 \
vllm serve Qwen/Qwen3-0.6B \
  --served-model-name qwen3-memory \
  --worker-extension-cls dynamic_memory_plugin.worker.DynamicMemoryWorker \
  --host 127.0.0.1 --port 8000 \
  --no-async-scheduling --no-enable-cumem-allocator \
  --no-enable-prefix-caching --attention-backend FLASH_ATTN \
  --dtype bfloat16 --block-size 16 --max-model-len 2048 \
  --max-num-seqs 4 --max-num-batched-tokens 512 \
  --kv-cache-memory-bytes 536870912 \
  --enforce-eager --compilation-config '{"mode": 0}'
```

This starts with a 512 MiB KV-cache target and allows resize targets up to 2 GiB.
Targets describe KV-cache bytes, not total GPU usage, and round to allocation
boundaries. The strategy is selected per resize request.

## Resize

```bash
# Inspect capacity and the latest operation.
curl -sS http://127.0.0.1:8000/v1/dynamic-memory/status

# Shrink to 256 MiB, preserving ongoing generations.
curl -sS http://127.0.0.1:8000/v1/dynamic-memory/resize \
  -H 'Content-Type: application/json' \
  -d '{"target_bytes":268435456,"strategy":"preserve","withdrawal_timeout_seconds":30}'

# After completion, grow to 512 MiB using drain.
curl -sS http://127.0.0.1:8000/v1/dynamic-memory/resize \
  -H 'Content-Type: application/json' \
  -d '{"target_bytes":536870912,"strategy":"drain"}'
```

Poll status until `operation.state` is `completed` or `unchanged` before another
resize. One mutation runs at a time. A failure with `restart_required: true`
requires restarting the engine. Withdrawal timeout applies before mutation;
it cannot interrupt a CUDA call.

## Run a workload

In another terminal, activate `.venv` and run:

```bash
vllm bench serve --backend openai --base-url http://127.0.0.1:8000 \
  --endpoint /v1/completions --model Qwen/Qwen3-0.6B \
  --served-model-name qwen3-memory --dataset-name random \
  --random-input-len 128 --random-output-len 256 --random-range-ratio 0.0 \
  --num-prompts 64 --num-warmups 1 --max-concurrency 4 --request-rate inf \
  --ignore-eos --seed 42 --percentile-metrics ttft,itl,e2el \
  --metric-percentiles 50,95,99 --save-result --result-dir results
```

## Allocators and runtime patch

The default **native** allocator replaces the cache allocation. Preserve copies
retained KV blocks and temporarily needs room for both old and new buffers.

The optional **VMM** (Virtual Memory Management) allocator maps and unmaps physical
chunks behind stable virtual addresses. For the Qwen example, set
`VLLM_DYNAMIC_MEMORY_ALLOCATOR=vmm`, `VLLM_KV_CACHE_LAYOUT=BLNHC`,
`VLLM_DYNAMIC_MEMORY_CHUNK_BYTES=14680064`,
`VLLM_DYNAMIC_MEMORY_MAX_KV_BYTES=1056964608`, and use
`--kv-cache-memory-bytes 528482304`. These values align to 14 MiB chunks.

The plugin uses a small **in-process patch (monkey patch)** to add engine utility
methods and lifecycle guards, and to check resize progress between iterations.
VMM also hooks cache allocation and block copying. Hooks are enabled by the
plugin settings above; no installed vLLM source files need editing. These
internal interfaces are why compatibility is pinned to the versions listed.

## Limits and tests

This is an experimental single-worker plugin. Unsupported configurations are
rejected, including distributed execution, CUDA graphs, asynchronous scheduling,
LoRA, speculative decoding and sleep mode. Preserve can wait for live references
and briefly pauses generation during the physical resize.

The resize endpoints have no plugin authentication or authorization. Keep them
on a trusted interface. Operation state is in memory; there is no durable
journal or retry history. Inspect status after a lost response before retrying.

```bash
python -m unittest discover -s tests -v
```

Local GPU checks cover native drain/preserve on 0.32.0 and VMM preserve on 0.33.0
with Qwen3-0.6B. This is not coverage of every model and configuration.

## License

[MIT](LICENSE).
