"""Refuse unreviewed runtime internals; inspect source without importing CUDA."""

import hashlib
import json
from importlib import metadata
from importlib.machinery import PathFinder, SourceFileLoader
from pathlib import Path
from functools import cache
import os
import sys


def import_source(relative: str) -> Path:
    """Resolve standard Python imports without executing parent packages.

    Distribution metadata can describe a different tree from PYTHONPATH. Also
    reject modules already loaded from a different location: checking only the
    next import's source would incorrectly bless the running interpreter.
    """
    parts = Path(relative).with_suffix("").parts
    search = None
    for index in range(len(parts)):
        name = ".".join(parts[:index + 1])
        spec = PathFinder.find_spec(name, search)
        if spec is None or not isinstance(spec.loader, SourceFileLoader) or not spec.origin:
            raise RuntimeError(f"Unreviewed import source: {name}")
        origin = Path(spec.origin).resolve(strict=True)
        loaded = sys.modules.get(name)
        if name in sys.modules:
            loaded_file = getattr(loaded, "__file__", None)
            loaded_spec = getattr(loaded, "__spec__", None)
            if (not loaded_file or not getattr(loaded_spec, "origin", None)
                    or Path(loaded_file).resolve(strict=True) != origin
                    or Path(loaded_spec.origin).resolve(strict=True) != origin):
                raise RuntimeError(f"Loaded module differs from import source: {name}")
        search = spec.submodule_search_locations
        if index < len(parts) - 1:
            if search is None:
                raise RuntimeError(f"Unreviewed package source: {name}")
            if loaded is not None:
                actual = getattr(loaded, "__path__", ())
                if [Path(p).resolve(strict=True) for p in actual] != [
                        Path(p).resolve(strict=True) for p in search]:
                    raise RuntimeError(f"Loaded package differs from import source: {name}")
    return origin


@cache
def runtime_info():
    """Capture the installed package identity when the plugin registers."""
    root = Path(__file__).parent
    names = ("VLLM_DYNAMIC_MEMORY_ALLOCATOR", "VLLM_DYNAMIC_MEMORY_CHUNK_BYTES", "VLLM_DYNAMIC_MEMORY_MAX_KV_BYTES",
             "VLLM_DYNAMIC_MEMORY_SHRINK_POLICY",
             "VLLM_DYNAMIC_MEMORY_HEADROOM_BYTES", "VLLM_KV_CACHE_LAYOUT", "VLLM_BATCH_INVARIANT", "DYNAMIC_MEMORY_VERIFY_KV_COPY")
    return {
        "package_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted(root.iterdir()) if p.suffix in {".py", ".json"}},
        "versions": {name: metadata.version(name) for name in ("vllm", "torch", "cuda-bindings")},
        "upstream": verify_runtime(),
        "settings": {name: os.environ.get(name) for name in names},
    }


def verify_runtime() -> dict:
    dist = metadata.distribution("vllm")
    from .runtime_profile import manifest_path
    expected = json.loads(manifest_path(dist.version).read_text())
    if dist.version != expected["version"]:
        raise RuntimeError(f"Requires reviewed vLLM {expected['version']}; found {dist.version}")
    for name, version in expected["dependencies"].items():
        actual = metadata.version(name)
        if actual != version:
            raise RuntimeError(f"Requires reviewed {name} {version}; found {actual}")
    mismatches = []
    sources = {}
    for relative, digest in expected["sha256"].items():
        path = import_source(relative)
        sources[relative] = str(path)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            mismatches.append(relative)
    if mismatches:
        raise RuntimeError("Unreviewed vLLM internals: " + ", ".join(mismatches))
    return {"version": dist.version, "checked_files": len(expected["sha256"]),
            "source_paths": sources, "source_sha256": expected["sha256"]}
