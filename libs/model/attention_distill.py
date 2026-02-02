from __future__ import annotations

import math
from typing import Any, List

import torch
import torch.nn as nn


def _iter_model_candidates(model: Any) -> List[Any]:
    model = getattr(model, "module", model)
    candidates: List[Any] = []

    def add(obj: Any) -> None:
        if obj is not None:
            candidates.append(obj)

    add(model)
    add(getattr(model, "model", None))
    base_model = getattr(model, "base_model", None)
    add(base_model)
    add(getattr(base_model, "model", None) if base_model is not None else None)

    getter = getattr(model, "get_base_model", None)
    if callable(getter):
        try:
            bm = getter()
        except Exception:
            bm = None
        add(bm)
        add(getattr(bm, "model", None) if bm is not None else None)

    seen: set[int] = set()
    uniq: List[Any] = []
    for obj in candidates:
        oid = id(obj)
        if oid in seen:
            continue
        seen.add(oid)
        uniq.append(obj)
    return uniq


def get_vision_tower(model: Any) -> Any:
    for cand in _iter_model_candidates(model):
        if hasattr(cand, "vision_tower"):
            return getattr(cand, "vision_tower")
    raise AttributeError("Could not locate `vision_tower` on the provided model instance.")


def get_siglip_encoder_layers(vision_tower: Any) -> List[Any]:
    candidates: List[Any] = [vision_tower]
    inner = getattr(vision_tower, "vision_tower", None)
    if inner is not None:
        candidates.append(inner)

    for cand in candidates:
        if hasattr(cand, "vision_model") and hasattr(cand.vision_model, "encoder"):
            encoder = cand.vision_model.encoder
            layers = getattr(encoder, "layers", None)
            if isinstance(layers, (list, nn.ModuleList)):
                return list(layers)
        encoder = getattr(cand, "encoder", None)
        if encoder is not None:
            layers = getattr(encoder, "layers", None)
            if isinstance(layers, (list, nn.ModuleList)):
                return list(layers)
        layers = getattr(cand, "layers", None)
        if isinstance(layers, (list, nn.ModuleList)):
            return list(layers)

    raise AttributeError("Could not locate SigLip encoder layers under `vision_tower`.")


def llava_onevision_spatial_pool_query_indices(seq_len: int, *, device: torch.device) -> torch.Tensor:
    """
    Indices for the spatially pooled token grid used by LLaVA-OneVision (nearest-neighbor proxy).

    - If tokens are a square grid (S = G*G), we downsample to ceil(G/2) by taking every other token.
    - If tokens include a CLS token (S = 1 + G*G), we downsample patch tokens only (skip CLS).
    """

    if seq_len <= 0:
        raise ValueError("seq_len must be > 0")

    grid = int(round(math.sqrt(seq_len)))
    offset = 0
    if grid * grid != seq_len:
        grid2 = int(round(math.sqrt(seq_len - 1))) if seq_len > 1 else 0
        if grid2 > 0 and grid2 * grid2 == seq_len - 1:
            offset = 1
            grid = grid2
        else:
            raise ValueError(f"Cannot infer 2D patch grid from seq_len={seq_len}.")

    coords = torch.arange(0, grid, step=2, device=device, dtype=torch.long)
    idx = (coords[:, None] * grid + coords[None, :]).reshape(-1) + offset
    return idx

