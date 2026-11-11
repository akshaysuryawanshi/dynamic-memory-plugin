"""Startup exclusions shared by engine and endpoint plugins."""


def refuse_dev_mode():
    from vllm import envs

    if envs.VLLM_SERVER_DEV_MODE:
        raise ValueError("Dynamic memory refuses VLLM_SERVER_DEV_MODE: direct worker RPC bypasses resize coordination")
