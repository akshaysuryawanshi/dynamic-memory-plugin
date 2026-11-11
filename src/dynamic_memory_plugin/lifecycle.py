"""Fail closed through vLLM's existing EngineDead propagation.

The engine latch is authoritative. We never gracefully drain a process whose KV
mapping or scheduler publication is uncertain. Supervision lives outside vLLM.
"""

from functools import wraps


class DynamicMemoryFatalError(RuntimeError):
    pass


def check_fence(engine):
    error = getattr(engine, "_dynamic_memory_fatal", None)
    if error is not None:
        raise DynamicMemoryFatalError(f"Dynamic KV state is uncertain; restart required: {error}")


def fence_engine(engine, error):
    if getattr(engine, "_dynamic_memory_fatal", None) is not None:
        return
    engine._dynamic_memory_fatal = str(error)[:4096] or "unknown resize failure"
    # This is not a graceful shutdown request: shutdown draining may execute
    # another model step. Wake the event loop so its guard raises fatally.
    engine.process_input_queue_block = False
    queue = getattr(engine, "input_queue", None)
    if queue is not None:
        from vllm.v1.engine import EngineCoreRequestType
        queue.put_nowait((EngineCoreRequestType.WAKEUP, None))


def guarded(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        check_fence(self)
        result = method(self, *args, **kwargs)
        check_fence(self)
        return result
    call._dynamic_memory_lifecycle_guard = True
    return call


def install_lifecycle_guards(engine_class=None):
    if engine_class is None:
        from vllm.v1.engine.core import EngineCoreProc
        engine_class = EngineCoreProc
    # A utility exception alone is swallowed by _invoke_utility_method. These
    # outer guards run after its receipt is queued and escape to the busy-loop
    # fatal path, which sends ENGINE_CORE_DEAD and fails existing streams.
    for name in ("_handle_client_request", "_process_engine_step"):
        method = getattr(engine_class, name)
        if not getattr(method, "_dynamic_memory_lifecycle_guard", False):
            setattr(engine_class, name, guarded(method))
