"""Chunked CUDA virtual memory, independent of vLLM and its scheduler.

The caller owns synchronization with readers and the logical block pool: only
withdrawn, unused tail chunks may be shrunk. CUDA graphs are not supported by
this prototype. A tensor describes the entire reserved address range, including
unmapped addresses; never read/zero/copy that whole tensor indiscriminately.

No CUDA work happens at import. The small driver adapter reuses cuda.bindings
and the caller's existing PyTorch CUDA context; it creates no CUDA context.
"""

from dataclasses import dataclass
from math import lcm
import warnings


def allocation_quantum(block_bytes: int, granularity: int) -> int:
    """Smallest independently mappable unit containing only complete blocks."""
    if type(block_bytes) is not int or block_bytes <= 0:
        raise ValueError("block_bytes must be a positive integer")
    if type(granularity) is not int or granularity <= 0:
        raise ValueError("granularity must be a positive integer")
    return lcm(block_bytes, granularity)


def floor_budget(budget: int, quantum: int) -> int:
    if type(budget) is not int or budget < 0 or type(quantum) is not int or quantum <= 0:
        raise ValueError("budget must be nonnegative and quantum positive")
    return budget // quantum * quantum


class VmmError(RuntimeError):
    """A rejected transaction, or an arena requiring an engine fail-stop.

    ``poisoned=False`` guarantees the original committed mapping survived.
    ``committed_bytes`` is the current logical mapping boundary, not a promise
    that CUDA is usable after an asynchronous or recovery error.
    """

    def __init__(self, message, *, poisoned, committed_bytes):
        super().__init__(message)
        self.poisoned = poisoned
        self.committed_bytes = committed_bytes


class CudaCallError(RuntimeError):
    def __init__(self, operation, result):
        super().__init__(f"{operation} failed: {result}")
        self.code = int(result)
        # CUDA calls may surface a preceding asynchronous fault. Only known
        # argument/capacity/unsupported errors permit a clean rollback claim.
        self.fatal = self.code not in {1, 2, 800, 801}


@dataclass
class _Chunk:
    offset: int
    handle: object
    mapped: bool = False


class CudaDriver:
    """Minimal checked CUDA Python adapter, constructed on the worker thread."""

    def __init__(self, device_index: int):
        from cuda.bindings import driver

        self.api = driver
        self.device_index = int(device_index)
        self.context = self._call("cuCtxGetCurrent")
        if not int(self.context):
            raise RuntimeError("Initialize PyTorch CUDA on this thread before creating VMM storage")
        self.assert_context()
        supported = self._call(
            "cuDeviceGetAttribute",
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
            self.device_index,
        )
        if not supported:
            raise RuntimeError("CUDA virtual memory management is unavailable on this device")
        self.prop = driver.CUmemAllocationProp()
        self.prop.type = driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        self.prop.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        self.prop.location.id = self.device_index
        self.access = driver.CUmemAccessDesc()
        self.access.location.type = self.prop.location.type
        self.access.location.id = self.device_index
        self.access.flags = driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        self.granularity = int(self._call(
            "cuMemGetAllocationGranularity", self.prop,
            driver.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM,
        ))

    def _call(self, name, *args):
        result = getattr(self.api, name)(*args)
        if int(result[0]):
            raise CudaCallError(name, result[0])
        if len(result) == 1:
            return None
        return result[1] if len(result) == 2 else result[1:]

    def assert_context(self):
        current = self._call("cuCtxGetCurrent")
        if int(current) != int(self.context) or int(self._call("cuCtxGetDevice")) != self.device_index:
            raise RuntimeError("VMM must be accessed in its original CUDA context and device")

    def reserve(self, size):
        return int(self._call("cuMemAddressReserve", size, self.granularity, 0, 0))

    def free_address(self, address, size):
        self._call("cuMemAddressFree", address, size)

    def create(self, size):
        return self._call("cuMemCreate", size, self.prop, 0)

    def map(self, address, size, handle):
        self._call("cuMemMap", address, size, 0, handle, 0)

    def allow_access(self, address, size):
        self._call("cuMemSetAccess", address, size, [self.access], 1)

    def zero(self, address, size):
        self._call("cuMemsetD8", address, 0, size)

    def synchronize(self):
        self._call("cuCtxSynchronize")

    def unmap(self, address, size):
        self._call("cuMemUnmap", address, size)

    def release(self, handle):
        self._call("cuMemRelease", handle)


class _ArrayOwner:
    """PyTorch's CUDA-array-interface storage deleter retains this provider."""

    def __init__(self, arena):
        self.arena = arena
        arena._exports += 1
        self.__cuda_array_interface__ = {
            "shape": (arena.reserved_bytes,), "strides": None, "typestr": "|i1",
            "data": (arena.address, False), "version": 3,
        }

    def __del__(self):
        self.arena._exports -= 1


class VmmArena:
    """Stable virtual reservation backed by independently allocated chunks.

    All sizes are exact chunk multiples. ``resize`` either commits completely
    or raises VmmError. A clean rejection preserves old mappings and contents;
    a poisoned error requires the caller to stop GPU scheduling. The allocator
    tracks every handle and map separately, including partial failures.

    Not thread safe: all mutations and explicit close belong to one worker.
    Tensor views retain this arena through their storage owner. ``close`` is
    forbidden while any exported storage is alive.
    """

    def __init__(self, driver, reserve_bytes: int, chunk_bytes: int):
        if (type(chunk_bytes) is not int or chunk_bytes <= 0
                or chunk_bytes % driver.granularity):
            raise ValueError("chunk_bytes must be a positive multiple of CUDA allocation granularity")
        if type(reserve_bytes) is not int or reserve_bytes <= 0 or reserve_bytes % chunk_bytes:
            raise ValueError("reserve_bytes must be a positive multiple of chunk_bytes")
        self.driver = driver
        self.chunk_bytes = chunk_bytes
        self.reserved_bytes = reserve_bytes
        self.mapped_bytes = 0
        self._chunks = {}
        self._exports = 0
        self.poisoned = False
        self.closed = True
        self.address = 0
        driver.assert_context()
        self.address = driver.reserve(reserve_bytes)
        self.closed = False

    @property
    def retained_physical_bytes(self):
        """Includes any unreleased handles after a failed cleanup."""
        return sum(chunk.handle is not None for chunk in self._chunks.values()) * self.chunk_bytes

    def diagnostics(self):
        """Observe owned resources without CUDA calls or synchronization.

        ``mapped_bytes`` is the committed boundary retained for compatibility.
        After a poisoned partial failure it can differ from actual mappings or
        retained handles. Equal accounting is not proof of CUDA-context health.
        """
        handles = sum(chunk.handle is not None for chunk in self._chunks.values())
        mapped = sum(chunk.mapped for chunk in self._chunks.values())
        physical = handles * self.chunk_bytes
        mapped_physical = mapped * self.chunk_bytes
        return {
            "mapped_bytes": self.mapped_bytes,
            "committed_mapped_bytes": self.mapped_bytes,
            "mapped_physical_bytes": mapped_physical,
            "reserved_bytes": self.reserved_bytes,
            "active_reserved_bytes": 0 if self.closed else self.reserved_bytes,
            "retained_physical_bytes": physical,
            "chunk_bytes": self.chunk_bytes,
            "granularity_bytes": self.driver.granularity,
            "tracked_chunk_count": len(self._chunks),
            "handle_count": handles,
            "mapped_chunk_count": mapped,
            "unmapped_handle_count": sum(
                chunk.handle is not None and not chunk.mapped for chunk in self._chunks.values()),
            "exported_storage_count": self._exports,
            "address_reserved": not self.closed,
            "closed": self.closed,
            "poisoned": self.poisoned,
            "accounting_consistent": (
                len(self._chunks) == handles == mapped
                and self.mapped_bytes == physical == mapped_physical
                and (not self.closed or not self.address)
            ),
        }

    def _error(self, operation, error, cleanup=()):
        detail = f"VMM {operation} failed: {type(error).__name__}: {error}"
        if cleanup:
            detail += "; recovery: " + "; ".join(str(item) for item in cleanup)
        return VmmError(detail, poisoned=self.poisoned, committed_bytes=self.mapped_bytes)

    def _discard(self, chunks):
        failures = []
        for chunk in reversed(chunks):
            try:
                if chunk.mapped:
                    self.driver.unmap(self.address + chunk.offset, self.chunk_bytes)
                    chunk.mapped = False
                if chunk.handle is not None:
                    self.driver.release(chunk.handle)
                    # Retrying cleanup must not release the same native handle
                    # twice, even if subsequent Python bookkeeping fails.
                    chunk.handle = None
                self._chunks.pop(chunk.offset, None)
            except Exception as exc:
                # Never release an address reservation or forget a live handle
                # after partial cleanup. A later explicit close may retry it.
                failures.append(exc)
        return failures

    def resize(self, target_bytes: int):
        if self.closed or self.poisoned:
            raise VmmError("VMM arena is closed or poisoned", poisoned=self.poisoned,
                           committed_bytes=self.mapped_bytes)
        if (type(target_bytes) is not int or target_bytes < 0
                or target_bytes > self.reserved_bytes or target_bytes % self.chunk_bytes):
            raise ValueError("target_bytes must be a chunk multiple within the reservation")
        try:
            self.driver.assert_context()
        except Exception as exc:
            # Resuming the worker in an unexpected/faulted CUDA context is not
            # a safe rejected resize operation, even before mappings change.
            self.poisoned = True
            raise self._error("context validation", exc) from exc
        if target_bytes == self.mapped_bytes:
            return
        if target_bytes > self.mapped_bytes:
            self._grow(target_bytes)
        else:
            self._shrink(target_bytes)

    def _grow(self, target_bytes):
        added = []
        synchronizing = False
        write_attempted = False
        try:
            # Allocate and register every Python ownership record before any
            # native allocation. A host MemoryError must not orphan a CUDA
            # handle between cuMemCreate and bookkeeping it in this arena.
            added = [_Chunk(offset, None)
                     for offset in range(self.mapped_bytes, target_bytes, self.chunk_bytes)]
            self._chunks.update((chunk.offset, chunk) for chunk in added)
            for chunk in added:
                chunk.handle = self.driver.create(self.chunk_bytes)
                self.driver.map(self.address + chunk.offset, self.chunk_bytes, chunk.handle)
                chunk.mapped = True
                self.driver.allow_access(self.address + chunk.offset, self.chunk_bytes)
                write_attempted = True
                self.driver.zero(self.address + chunk.offset, self.chunk_bytes)
            synchronizing = True
            self.driver.synchronize()
        except Exception as exc:
            cleanup = []
            can_unmap = not synchronizing
            if write_attempted and not synchronizing:
                try:
                    # A later allocation can fail after earlier memset work
                    # was submitted. Drain those writes before unmapping their
                    # addresses, even though no new capacity was published.
                    self.driver.synchronize()
                except Exception as sync_error:
                    cleanup.append(sync_error)
                    can_unmap = False
            if can_unmap:
                cleanup.extend(self._discard(added))
            # Synchronization failures can indicate an asynchronous CUDA fault;
            # old storage contents must not be claimed safe in that case.
            self.poisoned = synchronizing or getattr(exc, "fatal", False) or bool(cleanup)
            raise self._error("grow", exc, cleanup) from exc
        self.mapped_bytes = target_bytes

    def _shrink(self, target_bytes):
        try:
            self.driver.synchronize()
        except Exception as exc:
            self.poisoned = True
            raise self._error("shrink synchronization", exc) from exc
        retiring = [self._chunks[offset]
                    for offset in range(target_bytes, self.mapped_bytes, self.chunk_bytes)]
        try:
            for chunk in retiring:
                self.driver.unmap(self.address + chunk.offset, self.chunk_bytes)
                chunk.mapped = False
        except Exception as exc:
            recovery = []
            for chunk in retiring:
                if chunk.mapped:
                    continue
                try:
                    self.driver.map(self.address + chunk.offset, self.chunk_bytes, chunk.handle)
                    chunk.mapped = True
                    self.driver.allow_access(self.address + chunk.offset, self.chunk_bytes)
                except Exception as remap_error:
                    recovery.append(remap_error)
            self.poisoned = getattr(exc, "fatal", False) or bool(recovery)
            raise self._error("shrink", exc, recovery) from exc
        # All unmaps succeeded while every old handle was still retained, so
        # any preceding failure was reversible without allocation or data copy.
        self.mapped_bytes = target_bytes
        cleanup = self._discard(retiring)
        if cleanup:
            self.poisoned = True
            raise self._error("release retired physical memory", cleanup[0], cleanup[1:])

    def tensor(self):
        """Export a zero-copy int8 tensor; its full reservation is NOT mapped."""
        if self.closed or self.poisoned or not self.mapped_bytes:
            raise RuntimeError("Map at least one chunk before exporting a healthy VMM arena")
        import torch
        self.driver.assert_context()
        return torch.as_tensor(_ArrayOwner(self), dtype=torch.int8,
                               device=f"cuda:{self.driver.device_index}")

    def close(self):
        if self.closed:
            return
        if self._exports:
            raise RuntimeError("Drop all VMM tensor/storage views before closing the arena")
        self.driver.assert_context()
        try:
            self.driver.synchronize()
        except Exception as exc:
            self.poisoned = True
            raise self._error("close synchronization", exc) from exc
        failures = self._discard(list(self._chunks.values()))
        if failures:
            self.poisoned = True
            raise self._error("close cleanup", failures[0], failures[1:])
        # Physical resources are gone even if releasing the virtual address
        # reservation fails. Keep that reservation owned for an explicit retry.
        self.mapped_bytes = 0
        try:
            self.driver.free_address(self.address, self.reserved_bytes)
        except Exception as exc:
            self.poisoned = True
            raise self._error("free address reservation", exc) from exc
        self.address = 0
        self.closed = True

    def __del__(self):
        if not getattr(self, "closed", True):
            try:
                self.close()
            except Exception as exc:
                # A failed CUDA context cannot safely be repaired in a GC
                # callback. Keep failures visible without aborting Python.
                warnings.warn(f"VMM cleanup deferred to CUDA process teardown: {exc}",
                              ResourceWarning)
