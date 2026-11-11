"""Install namespaced EngineCore utilities through vLLM's general plugin."""

import os

from .compat import runtime_info, verify_runtime
from .control import ControlError


def _with_runtime(result):
    result["runtime"] = runtime_info()
    return result


def _controller(engine):
    if not hasattr(engine, "_dynamic_memory_controller"):
        from .adapter import EngineBackend, validate_config
        from .controller import ResizeController
        validate_config(engine.vllm_config)
        engine._dynamic_memory_controller = ResizeController(EngineBackend(engine))
    return engine._dynamic_memory_controller


def dynamic_memory_status(self):
    return _with_runtime(_controller(self).status())


def dynamic_memory_resize(self, target_bytes: int, strategy: str = "drain",
                          withdrawal_timeout_seconds: float | None = None):
    try:
        return _with_runtime(_controller(self).resize(
            target_bytes, strategy, withdrawal_timeout_seconds))
    except ControlError as exc:
        return {"control_error": exc.as_dict()}


def register():
    # Installed general plugins otherwise autoload. This package changes no
    # engine unless its own plugin name is explicitly selected.
    allowed = {name.strip() for name in os.environ.get("VLLM_PLUGINS", "").split(",")}
    if "dynamic_memory" not in allowed:
        return
    from .safety import refuse_dev_mode
    refuse_dev_mode()
    verify_runtime()
    runtime_info()
    from vllm.v1.engine.core import EngineCore
    methods = {
        "dynamic_memory_status": dynamic_memory_status,
        "dynamic_memory_resize": dynamic_memory_resize,
    }
    for name, method in methods.items():
        existing = getattr(EngineCore, name, None)
        if existing is not None and existing is not method:
            raise RuntimeError(f"EngineCore already defines {name}; refusing overwrite")
    for name, method in methods.items():
        setattr(EngineCore, name, method)
    from .lifecycle import install_lifecycle_guards
    install_lifecycle_guards()
    from .adapter import vmm_enabled
    if vmm_enabled():
        from .vmm_integration import install_allocator_hook
        install_allocator_hook()
