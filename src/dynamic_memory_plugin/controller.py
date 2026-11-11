"""Engine-loop-owned resize state machine, independently testable without vLLM."""

from concurrent.futures import Future
from copy import deepcopy
import math
import os
from threading import get_native_id
from time import monotonic, monotonic_ns, time_ns
from uuid import uuid4

from .control import ControlError


class ResizeController:
    """Own one mutation at a time on the engine thread.

    The backend prepares CPU metadata before touching storage. Unknown worker,
    publication, or rollback outcomes fence the engine and retain the last safe
    capacity observation instead of guessing current storage.
    """

    ACTIVE_STATES = frozenset({"draining", "withdrawing", "resizing"})
    FAILURE_STATES = frozenset({"failed_paused", "failed_unsafe"})

    def __init__(self, backend):
        self.backend = backend
        self.controller_id = str(uuid4())
        self.operation_sequence = 0
        self.fatal_error = None
        self.last_safe_capacity = deepcopy(backend.capacity())
        self.operation = None
        self.plan = None
        self.started = None
        self.deadline = None

    def status(self):
        # A deadline never interrupts an iteration or a CUDA call. It can only
        # stop withdrawal before physical mutation at an engine-loop boundary.
        if (self.operation is not None
                and self.operation["state"] == "withdrawing"
                and self.deadline is not None
                and monotonic() >= self.deadline):
            self._cancel_withdrawal("timed_out")
        return self._snapshot()

    def _snapshot(self):
        uncertain = self.fatal_error is not None or (
            self.operation is not None
            and self.operation["state"] in self.FAILURE_STATES)
        return {
            "implementation": "dynamic-memory-poc-v1",
            "experimental": True,
            "controller_id": self.controller_id,
            "capacity": self._snapshot_capacity(uncertain),
            "operation": deepcopy(self.operation),
            "restart_required": uncertain,
            "scheduling_pause_confirmed": (
                self.operation["state"] == "failed_paused" if uncertain else None
            ),
            "withdrawal_timeout_semantics": {
                "enforced_at": "engine iteration and status utility boundaries",
                "interrupts_active_iteration_or_cuda_call": False,
            },
        }

    def _snapshot_capacity(self, uncertain):
        if uncertain:
            capacity = deepcopy(self.last_safe_capacity)
        else:
            capacity = self.backend.capacity()
            self.last_safe_capacity = deepcopy(capacity)
        observed_bytes = capacity.get("allocated_kv_bytes")
        observed_blocks = capacity.get("num_blocks")
        capacity["logical_kv_bytes"] = observed_bytes
        capacity["logical_allocation_state"] = "unknown" if uncertain else "known"
        capacity["physical_allocation_state"] = "unknown" if uncertain else "known"
        if uncertain:
            capacity["last_observed_config"] = {
                "logical_kv_bytes": observed_bytes,
                "num_blocks": observed_blocks,
            }
            capacity["logical_kv_bytes"] = None
            capacity["allocated_kv_bytes"] = None
            capacity["num_blocks"] = None
            capacity["paused"] = (
                True if self.operation["state"] == "failed_paused" else None)
            for key in (
                "free_blocks", "retiring_blocks", "retiring_referenced_blocks",
                "withdrawal_stage", "pending_target_blocks", "pending_shrink_blocks",
                "referenced_blocks", "referenced_blocks_over_target",
            ):
                if key in capacity:
                    capacity[key] = None
        return capacity

    def resize(self, target_bytes: int, strategy: str = "drain",
               withdrawal_timeout_seconds: float | None = None):
        self._validate_request(target_bytes, strategy, withdrawal_timeout_seconds)
        if self.fatal_error is not None:
            raise ControlError(
                "operation_conflict", "Resize state is uncertain; restart the engine")
        if self.operation and self.operation["state"] in self.ACTIVE_STATES:
            raise ControlError("operation_conflict", "A resize is already in progress")

        self._initialize_resize(target_bytes, strategy, withdrawal_timeout_seconds)
        if self.plan["new_bytes"] == self.plan["old_bytes"]:
            self._finish("unchanged", True)
        elif strategy == "preserve":
            self._start_preserving()
        else:
            self._start_draining()
        return self.status()

    @staticmethod
    def _validate_request(target_bytes, strategy, withdrawal_timeout_seconds):
        if type(target_bytes) is not int or target_bytes <= 0:
            raise ControlError("invalid_request", "target_bytes must be a positive integer")
        if strategy not in {"drain", "preserve"}:
            raise ControlError("invalid_request", "strategy must be drain or preserve")
        if withdrawal_timeout_seconds is not None:
            if (type(withdrawal_timeout_seconds) not in (float, int)
                    or not math.isfinite(withdrawal_timeout_seconds)
                    or withdrawal_timeout_seconds <= 0
                    or strategy != "preserve"):
                raise ControlError(
                    "invalid_request",
                    "withdrawal_timeout_seconds must be finite, positive, and use preserve",
                )

    def _initialize_resize(self, target_bytes, strategy, withdrawal_timeout_seconds):
        prepare_start = monotonic_ns()
        plan = self.backend.prepare(target_bytes, strategy)
        self.operation_sequence += 1
        operation_id = str(uuid4())
        self.plan = plan
        self.started = monotonic()
        self.deadline = (
            None if withdrawal_timeout_seconds is None
            else self.started + withdrawal_timeout_seconds)
        self.operation = {
            "operation_id": operation_id,
            "sequence": self.operation_sequence,
            "requested_bytes": target_bytes,
            "old_bytes": plan["old_bytes"],
            "effective_bytes": plan["old_bytes"],
            "planned_bytes": plan["new_bytes"],
            "state": "draining",
            "error": None,
            "safe_to_resume": False,
            "restart_required": False,
            "strategy": strategy,
            "engine_pid": os.getpid(),
            "engine_tid": get_native_id(),
            "start_monotonic_ns": monotonic_ns(),
            "start_unix_ns": time_ns(),
            "prepare_start_monotonic_ns": prepare_start,
            "withdrawal_timeout_seconds": withdrawal_timeout_seconds,
        }
        plan["operation_context"] = {
            "controller_id": self.controller_id,
            "operation_id": operation_id,
            "sequence": self.operation_sequence,
        }

    def _start_preserving(self):
        self.operation["state"] = "withdrawing"
        try:
            self.backend.begin_preserving(self.plan, self._progress_preserving)
        except Exception as exc:
            try:
                self.backend.end_preserving(cancel=True)
                self._finish("rejected", True, str(exc))
            except Exception as cleanup:
                self._fail(f"{exc}; cleanup failed: {cleanup}", preserving=True)
            return
        self._progress_preserving()

    def _start_draining(self):
        try:
            future = self.backend.pause()
            if future is None:
                self._after_drain(None)
            else:
                future.add_done_callback(self._after_drain)
        except Exception as exc:
            self._fail(str(exc))

    def _cancel_withdrawal(self, state):
        try:
            self.backend.end_preserving(cancel=True)
            self._finish(state, True)
        except Exception as exc:
            self._fail(str(exc), preserving=True)

    def _fail(self, error, preserving=False):
        self.fatal_error = str(error)[:4096]
        fence = getattr(self.backend, "fence", None)
        if fence is not None:
            try:
                fence(self.fatal_error)
            except Exception as exc:
                error = f"{error}; engine fence failed: {exc}"
        try:
            secondary = (
                self.backend.end_preserving(failed=True)
                if preserving else self.backend.ensure_paused())
            if secondary:
                error = f"{error}; recovery pause reported: {secondary}"
        except Exception as cleanup:
            error = f"{error}; failed to stop scheduling: {cleanup}"
        try:
            paused = self.backend.scheduling_paused() is True
        except Exception as check:
            paused = False
            error = f"{error}; failed to confirm scheduling pause: {check}"
        self._finish("failed_paused" if paused else "failed_unsafe", False, error)

    def _progress_preserving(self):
        if self.operation["state"] != "withdrawing":
            return
        if self.deadline is not None and monotonic() >= self.deadline:
            self._cancel_withdrawal("timed_out")
            return
        try:
            if not self.backend.preserving_ready():
                return
            self.operation["withdraw_seconds"] = monotonic() - self.started
            self.operation["state"] = "resizing"
            result = self.backend.replace_preserving(self.plan)
            self.operation["worker"] = result
            if result["outcome"] == "applied":
                self.backend.publish_preserving(self.plan)
                self.operation["effective_bytes"] = self.plan["new_bytes"]
                self.backend.end_preserving()
                self._finish("completed", True)
            elif result["outcome"] == "rejected":
                self.backend.end_preserving(cancel=True)
                self._finish("rejected", True, result.get("error"))
            else:
                raise RuntimeError(result.get("error", "Unknown worker outcome"))
        except Exception as exc:
            self._fail(str(exc), preserving=True)

    def _after_drain(self, future: Future | None):
        try:
            if future is not None:
                future.result()
            self.backend.assert_drained()
        except Exception as exc:
            self._fail(str(exc))
            return
        self.operation["state"] = "resizing"
        self.operation["drain_seconds"] = monotonic() - self.started
        try:
            result = self.backend.replace(self.plan)
            self.operation["worker"] = result
            if result["outcome"] == "applied":
                self.backend.publish(self.plan, rollback=False)
                self.operation["effective_bytes"] = self.plan["new_bytes"]
                state = "completed"
            elif result["outcome"] == "rolled_back":
                self.backend.publish(self.plan, rollback=True)
                state = "rolled_back"
            elif result["outcome"] == "rejected":
                state = "rejected"
            else:
                self._fail(result.get("error", "Worker outcome unknown"))
                return
            self.backend.resume()
            self._finish(state, True, result.get("error"))
        except Exception as exc:
            self._fail(str(exc))

    def _finish(self, state, safe, error=None):
        self.operation.update(
            state=state,
            safe_to_resume=safe,
            error=error,
            restart_required=state in self.FAILURE_STATES,
            elapsed_seconds=monotonic() - self.started,
            end_monotonic_ns=monotonic_ns(),
        )
        self.deadline = None
        if state not in self.FAILURE_STATES:
            self.plan = None
