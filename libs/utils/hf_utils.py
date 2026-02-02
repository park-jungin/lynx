import sys
from typing import Optional

import torch


def maybe_patch_torch_autograd_graph_for_deepspeed() -> None:
    """
    Work around a DeepSpeed ZeRO stage 1/2 crash with some PyTorch versions where
    `torch.autograd.graph._get_grad_fn_or_grad_acc` is invoked under no-grad.
    """

    try:
        import torch.autograd.graph as autograd_graph
    except Exception:
        return

    original = getattr(autograd_graph, "_get_grad_fn_or_grad_acc", None)
    if not callable(original):
        return
    if getattr(original, "_lynx_enable_grad_patch", False):
        return

    def patched(t: torch.Tensor):
        with torch.enable_grad():
            return original(t)

    patched._lynx_enable_grad_patch = True  # type: ignore[attr-defined]
    autograd_graph._get_grad_fn_or_grad_acc = patched  # type: ignore[attr-defined]

    if "deepspeed.runtime.utils" in sys.modules:
        try:
            import deepspeed.runtime.utils as ds_utils  # type: ignore[import-not-found]

            if getattr(ds_utils, "_get_grad_fn_or_grad_acc", None) is original:
                ds_utils._get_grad_fn_or_grad_acc = patched  # type: ignore[attr-defined]
        except Exception:
            pass


def resolve_local_model_path(model_name_or_path: str, *, cache_dir: Optional[str], local_files_only: bool) -> str:
    """
    Resolve a HF model id to a local on-disk directory when running with `local_files_only=True`.

    If `model_name_or_path` is already a local path, return it unchanged.
    """

    import os

    if os.path.exists(model_name_or_path):
        return model_name_or_path
    if not local_files_only:
        return model_name_or_path

    try:
        from transformers.utils.hub import cached_file

        config_path = cached_file(
            model_name_or_path,
            "config.json",
            cache_dir=cache_dir,
            local_files_only=True,
            _raise_exceptions_for_missing_entries=False,
        )
    except Exception:
        return model_name_or_path

    if config_path is None:
        return model_name_or_path
    return os.path.dirname(config_path)

