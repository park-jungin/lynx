from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

from lynx_utils import encode_llava_onevision_video_tokens


IGNORE_INDEX = -100


def temporal_mean_pool_video_tokens(tokens: torch.Tensor, *, frames: int) -> torch.Tensor:
    if tokens.ndim != 3:
        raise ValueError(f"Expected tokens of shape (B, V, D), got {tuple(tokens.shape)}")
    if frames <= 0:
        raise ValueError("frames must be > 0")

    newline = tokens[:, -1:, :]
    flat = tokens[:, :-1, :]
    if flat.shape[1] % frames != 0:
        raise ValueError(f"Token length {flat.shape[1]} is not divisible by frames={frames}")
    seq_len = flat.shape[1] // frames
    pooled = flat.reshape(tokens.shape[0], frames, seq_len, tokens.shape[2]).mean(dim=1)
    return torch.cat([pooled, newline], dim=1)  # [B, NxN + 1, D]


def spatial_mean_pool_video_tokens(tokens: torch.Tensor, *, frames: int) -> torch.Tensor:
    if tokens.ndim != 3:
        raise ValueError(f"Expected tokens of shape (B, V, D), got {tuple(tokens.shape)}")
    if frames <= 0:
        raise ValueError("frames must be > 0")

    newline = tokens[:, -1:, :]
    flat = tokens[:, :-1, :]
    if flat.shape[1] % frames != 0:
        raise ValueError(f"Token length {flat.shape[1]} is not divisible by frames={frames}")
    seq_len = flat.shape[1] // frames
    pooled = flat.reshape(tokens.shape[0], frames, seq_len, tokens.shape[2]).mean(dim=2)
    return torch.cat([pooled, newline], dim=1)  # [B, num_frames + 1, D]


def spatiotemporal_mean_pool_video_tokens(tokens: torch.Tensor, *, frames: int) -> torch.Tensor:
    if tokens.ndim != 3:
        raise ValueError(f"Expected tokens of shape (B, V, D), got {tuple(tokens.shape)}")
    if frames <= 0:
        raise ValueError("frames must be > 0")

    newline = tokens[:, -1:, :]
    flat = tokens[:, :-1, :]
    if flat.shape[1] % frames != 0:
        raise ValueError(f"Token length {flat.shape[1]} is not divisible by frames={frames}")
    seq_len = flat.shape[1] // frames
    t_pooled = flat.reshape(tokens.shape[0], frames, seq_len, tokens.shape[2]).mean(dim=1)
    s_pooled = flat.reshape(tokens.shape[0], frames, seq_len, tokens.shape[2]).mean(dim=2)
    return torch.cat([t_pooled, s_pooled, newline], dim=1)  # [B, num_frames + NxN + 1, D]


@contextmanager
def _maybe_disable_adapter(model: Any):
    peft_model = getattr(model, "module", model)
    disable = getattr(peft_model, "disable_adapter", None)
    if callable(disable):
        with peft_model.disable_adapter():
            yield
        return
    yield


class LynXOnevisionWrapper(nn.Module):
    """
    Minimal wrapper to inject LynX-style dual streams (video + modality) into a LLaVA-OneVision model.

    - Video tokens are extracted with adapters disabled (base vision encoder).
    - Modality tokens (e.g., audio pseudo-images or depth pseudo-images) are extracted with adapters enabled (vision LoRA).
    - Both token streams are concatenated and injected at a single placeholder position in `input_ids`.

    This wrapper is optional; training scripts may also build `inputs_embeds` directly.
    """

    def __init__(
        self,
        base_model: nn.Module,
        *,
        placeholder_token_id: int,
        pool_video_tokens: bool = True,
        frame_chunk_size: int = 4,
        vision_adapter_name: Optional[str] = None,
        llm_adapter_name: Optional[str] = None,
        detach_modality_tokens: bool = False,
        merge_modality_frame_pairs: bool = False,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.placeholder_token_id = int(placeholder_token_id)
        self.pool_video_tokens = bool(pool_video_tokens)
        self.frame_chunk_size = int(frame_chunk_size)
        self.vision_adapter_name = vision_adapter_name
        self.llm_adapter_name = llm_adapter_name
        self.detach_modality_tokens = bool(detach_modality_tokens)
        self.merge_modality_frame_pairs = bool(merge_modality_frame_pairs)

    def get_input_embeddings(self) -> nn.Module:
        return self.base_model.get_input_embeddings()

    def _encode_streams(
        self,
        *,
        pixel_values_videos: torch.Tensor,
        audio_pixel_values_videos: Optional[torch.Tensor] = None,
        depth_pixel_values_videos: Optional[torch.Tensor] = None,
        fast_pixel_values_videos: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        provided = [
            audio_pixel_values_videos is not None,
            depth_pixel_values_videos is not None,
            fast_pixel_values_videos is not None,
        ]
        if sum(provided) != 1:
            raise ValueError("Provide exactly one of: audio_pixel_values_videos, depth_pixel_values_videos, fast_pixel_values_videos.")

        modality_pixel_values_videos = (
            fast_pixel_values_videos
            if fast_pixel_values_videos is not None
            else (depth_pixel_values_videos if depth_pixel_values_videos is not None else audio_pixel_values_videos)
        )
        if modality_pixel_values_videos is None:
            raise ValueError("Missing modality stream pixel values.")

        core_model = getattr(self.base_model, "module", self.base_model)
        device = next(core_model.parameters()).device
        dtype = next(core_model.parameters()).dtype

        pixel_values_videos = pixel_values_videos.to(device=device, dtype=dtype)
        modality_pixel_values_videos = modality_pixel_values_videos.to(device=device, dtype=dtype)

        if self.detach_modality_tokens:
            with torch.no_grad():
                with _maybe_disable_adapter(core_model):
                    z_v = encode_llava_onevision_video_tokens(
                        core_model,
                        pixel_values_videos=pixel_values_videos,
                        vision_feature_layer=core_model.config.vision_feature_layer,
                        vision_feature_select_strategy=core_model.config.vision_feature_select_strategy,
                        frame_chunk_size=self.frame_chunk_size,
                    )
        else:
            with _maybe_disable_adapter(core_model):
                z_v = encode_llava_onevision_video_tokens(
                    core_model,
                    pixel_values_videos=pixel_values_videos,
                    vision_feature_layer=core_model.config.vision_feature_layer,
                    vision_feature_select_strategy=core_model.config.vision_feature_select_strategy,
                    frame_chunk_size=self.frame_chunk_size,
                )

        peft_model = getattr(core_model, "module", core_model)
        if self.vision_adapter_name is not None:
            setter = getattr(peft_model, "set_adapter", None)
            if not callable(setter):
                raise AttributeError("Expected PEFT model to expose `set_adapter` for multi-adapter LynX.")
            peft_model.set_adapter(self.vision_adapter_name)

        if self.detach_modality_tokens:
            with torch.no_grad():
                z_a = encode_llava_onevision_video_tokens(
                    core_model,
                    pixel_values_videos=modality_pixel_values_videos,
                    vision_feature_layer=core_model.config.vision_feature_layer,
                    vision_feature_select_strategy=core_model.config.vision_feature_select_strategy,
                    frame_chunk_size=self.frame_chunk_size,
                )
        else:
            z_a = encode_llava_onevision_video_tokens(
                core_model,
                pixel_values_videos=modality_pixel_values_videos,
                vision_feature_layer=core_model.config.vision_feature_layer,
                vision_feature_select_strategy=core_model.config.vision_feature_select_strategy,
                frame_chunk_size=self.frame_chunk_size,
            )

        if self.llm_adapter_name is not None and self.vision_adapter_name is not None:
            peft_model.set_adapter(self.llm_adapter_name)

        modality_frames = int(modality_pixel_values_videos.shape[1])
        if self.merge_modality_frame_pairs:
            if modality_frames % 2 != 0:
                raise ValueError(f"merge_modality_frame_pairs requires an even number of modality frames, got {modality_frames}.")
            newline = z_a[:, -1:, :]
            flat = z_a[:, :-1, :]
            if flat.shape[1] % modality_frames != 0:
                raise ValueError(f"Token length {flat.shape[1]} is not divisible by modality_frames={modality_frames}")
            seq_len = flat.shape[1] // modality_frames
            tokens = flat.reshape(flat.shape[0], modality_frames, seq_len, flat.shape[2])
            tokens = (tokens[:, 0::2, :, :] + tokens[:, 1::2, :, :]) * 0.5
            modality_frames = modality_frames // 2
            z_a = torch.cat([tokens.reshape(tokens.shape[0], modality_frames * seq_len, tokens.shape[3]), newline], dim=1)

        if self.pool_video_tokens:
            # z_v = spatiotemporal_mean_pool_video_tokens(z_v, frames=int(pixel_values_videos.shape[1]))
            z_a = spatial_mean_pool_video_tokens(z_a, frames=modality_frames)
            # print(z_v.shape, z_a.shape)

        if self.detach_modality_tokens:
            z_v = z_v.detach()
            z_a = z_a.detach()
        return z_v, z_a

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values_videos: torch.Tensor,
        audio_pixel_values_videos: Optional[torch.Tensor] = None,
        depth_pixel_values_videos: Optional[torch.Tensor] = None,
        fast_pixel_values_videos: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        **kwargs: Any,
    ) -> Any:
        if input_ids.ndim != 2:
            raise ValueError(f"Expected input_ids of shape (B, L), got {tuple(input_ids.shape)}")
        batch_size = input_ids.shape[0]

        z_v, z_a = self._encode_streams(
            pixel_values_videos=pixel_values_videos,
            audio_pixel_values_videos=audio_pixel_values_videos,
            depth_pixel_values_videos=depth_pixel_values_videos,
            fast_pixel_values_videos=fast_pixel_values_videos,
        )
        z = torch.cat([z_v, z_a], dim=1)

        embed = self.base_model.get_input_embeddings()
        new_embeds: list[torch.Tensor] = []
        new_labels: list[torch.Tensor] = []
        for i in range(batch_size):
            ids = input_ids[i]
            if attention_mask is not None:
                valid = int(attention_mask[i].to(device=ids.device).sum().item())
                ids = ids[:valid]
                lab = labels[i][:valid] if labels is not None else None
            else:
                lab = labels[i] if labels is not None else None
            pos = (ids == self.placeholder_token_id).nonzero()
            if pos.numel() != 1:
                raise ValueError(f"Expected exactly 1 placeholder token id={self.placeholder_token_id} per sample, got {pos.numel()}")
            ph = int(pos.item())

            prefix_ids = ids[:ph]
            suffix_ids = ids[ph + 1 :]
            prefix_emb = embed(prefix_ids).to(dtype=z.dtype)
            suffix_emb = embed(suffix_ids).to(dtype=z.dtype)
            new_embeds.append(torch.cat([prefix_emb, z[i], suffix_emb], dim=0))

            if lab is not None:
                prefix_lab = lab[:ph]
                suffix_lab = lab[ph + 1 :]
                ignore = torch.full((z.shape[1],), IGNORE_INDEX, dtype=lab.dtype, device=lab.device)
                new_labels.append(torch.cat([prefix_lab, ignore, suffix_lab], dim=0))

        max_len = max(x.shape[0] for x in new_embeds) if new_embeds else 0
        hidden = new_embeds[0].shape[-1] if new_embeds else z.shape[-1]
        device = new_embeds[0].device if new_embeds else z.device
        dtype = new_embeds[0].dtype if new_embeds else z.dtype

        inputs_embeds = torch.zeros((batch_size, max_len, hidden), device=device, dtype=dtype)
        attention_mask = torch.zeros((batch_size, max_len), device=device, dtype=torch.long)
        position_ids = torch.zeros((batch_size, max_len), device=device, dtype=torch.long)
        labels_padded: Optional[torch.Tensor] = None
        if labels is not None:
            labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, device=labels.device, dtype=labels.dtype)

        for i, emb in enumerate(new_embeds):
            seq_len = emb.shape[0]
            inputs_embeds[i, :seq_len] = emb
            attention_mask[i, :seq_len] = 1
            position_ids[i, :seq_len] = torch.arange(seq_len, device=device)
            if labels_padded is not None:
                labels_padded[i, :seq_len] = new_labels[i]

        return self.base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels_padded,
            use_cache=use_cache,
            **kwargs,
        )

    def save_pretrained(self, save_directory: str, **kwargs: Any) -> None:
        saver = getattr(self.base_model, "save_pretrained", None)
        if callable(saver):
            saver(save_directory, **kwargs)
            return
        raise AttributeError("Underlying base_model does not support save_pretrained().")
