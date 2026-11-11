"""Exact selection for the two reviewed native vLLM runtimes."""

from importlib import metadata
import json
from pathlib import Path


_MANIFESTS = {
    "0.32.0": "compatibility.json",
    "0.33.0": "compatibility-0.33.0.json",
}


def select(version=None):
    """Return one exact reviewed profile; version ranges are never inferred."""
    installed = metadata.version("vllm") if version is None else version
    name = _MANIFESTS.get(installed)
    if name is None:
        raise RuntimeError(f"Unreviewed vLLM runtime: {installed}")
    profile = json.loads(Path(__file__).with_name(name).read_text())
    if profile.get("version") != installed:
        raise RuntimeError(f"Compatibility manifest does not match vLLM {installed}")
    return profile


def manifest_path(version=None):
    profile = select(version)
    return Path(__file__).with_name(_MANIFESTS[profile["version"]])


def plan_kv_cache(config, groups, available_memory):
    """Dispatch to the exact native planner location for the installed runtime."""
    if select()["version"] == "0.33.0":
        from vllm.v1.core.kv_cache_planning import get_kv_cache_config_from_groups
    else:
        from vllm.v1.core.kv_cache_utils import get_kv_cache_config_from_groups
    return get_kv_cache_config_from_groups(config, groups, available_memory)
