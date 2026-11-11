"""Minimal unauthenticated HTTP surface for a trusted local vLLM server."""

import asyncio
import os
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict, Field

from .control import MAX_TARGET_BYTES


_UTILITY_LIMIT = 4
_STARTUP_TIMEOUT_SECONDS = 30.0
_CONTROL_ERROR_STATUS = {"invalid_request": 422, "operation_conflict": 409}
_RESIZE_STATUS = {
    "draining": 202,
    "withdrawing": 202,
    "resizing": 202,
    "rejected": 409,
    "rolled_back": 409,
    "timed_out": 409,
    "failed_paused": 503,
    "failed_unsafe": 503,
}


class _DynamicMemoryRoute(APIRoute):
    """Keep structured plugin errors independent of global OpenAI handlers."""

    def get_route_handler(self):
        handler = super().get_route_handler()

        async def dispatch(request):
            try:
                return await handler(request)
            except HTTPException as exc:
                return JSONResponse(
                    {"detail": exc.detail},
                    status_code=exc.status_code,
                    headers=exc.headers,
                )
            except RequestValidationError as exc:
                errors = [
                    {key: value for key, value in error.items()
                     if key in {"type", "loc", "msg"}}
                    for error in exc.errors()
                ]
                return JSONResponse({"detail": errors}, status_code=422)

        return dispatch


class _UtilityTasks:
    """Own native utility calls after HTTP cancellation or response loss."""

    def __init__(self):
        self.tasks = set()
        self.unavailable = False

    def _done(self, task):
        try:
            # Retrieve every exception after a disconnected waiter leaves.
            if task.cancelled() or task.exception() is not None:
                self.unavailable = True
        except BaseException:
            self.unavailable = True
        finally:
            self.tasks.discard(task)

    async def call(self, engine, method, *args):
        if self.unavailable:
            raise HTTPException(
                503, "Dynamic-memory utility transport failed; restart the API process")
        if len(self.tasks) >= _UTILITY_LIMIT:
            raise HTTPException(
                429, "Too many outstanding dynamic-memory utility calls")

        async def invoke():
            result = await engine.engine_core.call_utility_async(method, *args)
            _validate_result(result, method)
            return result

        # Shielding preserves ownership if the HTTP request is cancelled. There
        # is intentionally no HTTP mutation timeout and no cancellation claim.
        task = asyncio.create_task(invoke())
        self.tasks.add(task)
        task.add_done_callback(self._done)
        return await asyncio.shield(task)


def _utility_tasks(state):
    registry = getattr(state, "dynamic_memory_utility_tasks", None)
    if registry is None:
        registry = state.dynamic_memory_utility_tasks = _UtilityTasks()
    return registry


def _validate_result(result, method):
    if not isinstance(result, dict):
        raise ValueError("Engine utility result must be a dictionary")
    if "control_error" in result:
        error = result["control_error"]
        if (not isinstance(error, dict)
                or set(error) != {"code", "message"}
                or error.get("code") not in _CONTROL_ERROR_STATUS
                or not isinstance(error.get("message"), str)
                or len(error["message"]) > 4096):
            raise ValueError("Malformed engine control error")
    elif method == "dynamic_memory_resize":
        operation = result.get("operation")
        states = {
            "draining", "withdrawing", "resizing", "completed", "unchanged",
            "rejected", "rolled_back", "failed_paused", "failed_unsafe", "timed_out",
        }
        if (not isinstance(operation, dict)
                or operation.get("state") not in states):
            raise ValueError("Malformed engine resize result")


def _validate_initial_status(initial):
    _validate_result(initial, "dynamic_memory_status")
    capacity = initial.get("capacity")
    if (initial.get("implementation") != "dynamic-memory-poc-v1"
            or initial.get("restart_required") is not False
            or not isinstance(initial.get("controller_id"), str)
            or not isinstance(capacity, dict)
            or capacity.get("physical_allocation_state") != "known"
            or capacity.get("logical_allocation_state") != "known"
            or type(capacity.get("logical_kv_bytes")) is not int
            or capacity["logical_kv_bytes"] <= 0
            or capacity.get("allocated_kv_bytes") != capacity["logical_kv_bytes"]
            or type(capacity.get("num_blocks")) is not int
            or capacity["num_blocks"] <= 0):
        raise RuntimeError("Dynamic memory did not initialize with known KV storage")


def _client(request):
    engine = getattr(request.app.state, "dynamic_memory_client", None)
    if engine is None:
        raise HTTPException(503, "Dynamic-memory engine is not initialized")
    return engine


async def _call_utility(registry, engine, method, *args):
    if registry.unavailable:
        raise HTTPException(
            503, "Dynamic-memory utility transport failed; restart the API process")
    try:
        await engine.check_health()
    except Exception as exc:
        registry.unavailable = True
        raise HTTPException(503, "Engine is unavailable") from exc
    try:
        result = await registry.call(engine, method, *args)
    except HTTPException:
        raise
    except asyncio.CancelledError:
        # The shielded utility task remains owned by the registry.
        raise
    except Exception as exc:
        registry.unavailable = True
        raise HTTPException(
            503,
            "Engine utility response was lost; read status after recovery and do not retry blindly",
        ) from exc
    if "control_error" in result:
        error = result["control_error"]
        raise HTTPException(_CONTROL_ERROR_STATUS[error["code"]], error)
    return result


class ResizeBody(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    target_bytes: int = Field(gt=0, le=MAX_TARGET_BYTES)
    strategy: Literal["drain", "preserve"] = "drain"
    withdrawal_timeout_seconds: float | None = Field(
        default=None, gt=0, allow_inf_nan=False)


class DynamicMemoryEndpoints:
    name = "dynamic_memory_api"
    required_tasks = ("generate",)

    def attach_router(self, app):
        router = APIRouter(
            prefix="/v1/dynamic-memory", route_class=_DynamicMemoryRoute)
        registry = _utility_tasks(app.state)

        @router.get("/status")
        async def status(request: Request):
            return await _call_utility(
                registry, _client(request), "dynamic_memory_status")

        @router.post("/resize")
        async def resize(body: ResizeBody, request: Request):
            result = await _call_utility(
                registry,
                _client(request),
                "dynamic_memory_resize",
                body.target_bytes,
                body.strategy,
                body.withdrawal_timeout_seconds,
            )
            state = result["operation"]["state"]
            return JSONResponse(result, status_code=_RESIZE_STATUS.get(state, 200))

        app.include_router(router)

    async def init_state(self, engine_client, state, args):
        from .safety import refuse_dev_mode
        from .compat import verify_runtime

        refuse_dev_mode()
        verify_runtime()
        allowed = {
            name.strip() for name in os.environ.get("VLLM_PLUGINS", "").split(",")}
        if "dynamic_memory" not in allowed:
            raise RuntimeError(
                "Enable both dynamic_memory and dynamic_memory_api plugins")
        if not hasattr(getattr(engine_client, "engine_core", None), "call_utility_async"):
            raise RuntimeError("Requires the asynchronous multiprocess EngineClient")
        if getattr(args, "api_server_count", 1) not in (None, 1):
            raise RuntimeError("Prototype supports one API server process")

        state.dynamic_memory_client = None
        initial = await asyncio.wait_for(
            engine_client.engine_core.call_utility_async("dynamic_memory_status"),
            timeout=_STARTUP_TIMEOUT_SECONDS,
        )
        # Startup status is read-only; cancelling it cannot interrupt mutation.
        _validate_initial_status(initial)
        state.dynamic_memory_client = engine_client
