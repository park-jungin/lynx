import gc
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor, HfArgumentParser, LlavaOnevisionForConditionalGeneration, TrainingArguments, set_seed
from transformers.trainer_utils import IntervalStrategy
from tqdm.auto import tqdm

from libs.dataset.lynx_alignment import (
    AudioFromVideoDataset,
    DepthFromFramesDataset,
    FastVideoFromVideoDataset,
    LynX3DDataCollator,
    LynXDataCollator,
    LynXFastVideoCollator,
    VideoFromVideoCollator,
    VideoFromVideoDataset,
    chunk_starts,
    iter_scene_dirs,
    pad_to_length,
)
from libs.model.attention_distill import get_siglip_encoder_layers, get_vision_tower
from libs.utils.hf_utils import maybe_patch_torch_autograd_graph_for_deepspeed, resolve_local_model_path
from libs.utils.lynx_3d import list_image_paths, sample_uniform
from libs.utils.lynx_trainer import LynXAVQATrainer
from lynx_utils import (
    build_dummy_prompt_token_ids,
    encode_llava_onevision_video_tokens,
    iter_video_paths_from_dir,
    iter_video_paths_from_mapping,
    load_video_frames_and_audio,
    resolve_attn_implementation,
    running_mean_var_finalize,
    running_mean_var_init,
    running_mean_var_update,
)


@dataclass
class LynXModelArguments:
    model_name_or_path: str = field(
        default="llava-hf/llava-onevision-qwen2-7b-ov-hf",
        metadata={"help": "HF model id or local path for LLaVA-OneVision."},
    )
    cache_dir: Optional[str] = field(default='/mnt/hdd1/')
    local_files_only: bool = field(default=False)
    use_flash_attn: bool = field(default=True)


@dataclass
class LynXDataArguments:
    reference_video_root: str = field(default="./data/video_instruction_tuning/avqa/videos")
    reference_video_mapping: Optional[str] = field(default="./data/video_instruction_tuning/avqa/avqa_from_vid_to_video_name.json")
    reference_max_videos: int = field(default=50000)
    reference_num_frames: int = field(default=20)
    reference_stats_path: Optional[str] = field(default="./checkpoints/LynX_7b/audio/avqa/train/reference_stats.pt")

    train_video_root: str = field(default="./data/video_instruction_tuning/avqa/videos")
    train_video_mapping: Optional[str] = field(default="./data/video_instruction_tuning/avqa/avqa_from_vid_to_video_name.json")
    train_annotation_file: Optional[str] = field(default="./data/video_instruction_tuning/avqa/train_qa.json")
    train_max_videos: Optional[int] = field(default=None)

    video_backend: str = field(default="torchvision")
    vision_frame_chunk_size: int = field(default=4)

    audio_target_sr: int = field(default=16000)
    audio_seconds: float = field(default=8.0)
    mel_n_mels: int = field(default=128)
    mel_n_fft: int = field(default=1024)
    mel_hop_length: int = field(default=256)
    mel_win_length: int = field(default=1024)

    audio_repeat_frames: Optional[int] = field(default=None)


@dataclass
class LynXEgoExoDataArguments:
    reference_video_root: str = field(default="./data/video_instruction_tuning/egoexo")
    reference_video_mapping: Optional[str] = field(default="./data/video_instruction_tuning/egoexo/from_take_id_to_video.json")
    reference_max_videos: int = field(default=50000)
    reference_num_frames: int = field(default=20)
    reference_stats_path: Optional[str] = field(default=None)

    train_video_root: str = field(default="./data/video_instruction_tuning/egoexo")
    train_video_mapping: Optional[str] = field(default="./data/video_instruction_tuning/egoexo/from_take_id_to_video.json")
    train_max_videos: Optional[int] = field(default=None)
    train_num_frames: Optional[int] = field(default=None)

    video_backend: str = field(default="torchvision")
    vision_frame_chunk_size: int = field(default=4)


@dataclass
class LynXFastVideoDataArguments:
    reference_video_roots: str = field(
        default="",
        metadata={"help": "Comma-separated list of roots (each root points to academic_source/ or liwei_youtube_videos/)."},
    )
    reference_video_mappings: str = field(
        default="",
        metadata={"help": "Comma-separated list of mapping jsons aligned with reference_video_roots."},
    )
    reference_max_videos: int = field(default=50000)
    reference_num_frames: int = field(default=20)
    reference_stats_path: Optional[str] = field(default=None)
    video_backend: str = field(default="torchvision")
    vision_frame_chunk_size: int = field(default=4)

    train_video_roots: str = field(default="")
    train_video_mappings: str = field(default="")
    train_max_videos: Optional[int] = field(default=None)

    fast_num_frames: Optional[int] = field(
        default=None, metadata={"help": "Frames sampled from each video before tiling. Defaults to 4 * reference_num_frames."}
    )
    fast_frame_multiplier: int = field(default=4)
    fast_tile_size: int = field(default=4, metadata={"help": "T frames per tiled composite (must be a perfect square)."})


@dataclass
class LynX3DDataArguments:
    reference_frames_root: str = field(default="./data/video_instruction_tuning/3d/frames_square")
    reference_max_videos: int = field(default=50000)
    reference_num_frames: int = field(default=20)
    reference_stats_path: Optional[str] = field(default=None)
    reference_use_all_frames: bool = field(default=False)
    reference_chunk_stride: Optional[int] = field(
        default=None,
        metadata={"help": "Stride (in frames) when reference_use_all_frames=True. Defaults to reference_num_frames."},
    )

    train_frames_root: str = field(default="./data/video_instruction_tuning/3d/frames_square")
    train_annotation_file: Optional[str] = field(default=None)
    train_max_videos: Optional[int] = field(default=None)

    color_subdir: str = field(default="color")
    depth_subdir: str = field(default="depth")

    vision_frame_chunk_size: int = field(default=4)

    depth_num_frames: int = field(default=20)
    depth_repeat_frames: Optional[int] = field(default=None)
    depth_clip_min_mm: float = field(default=200.0)
    depth_clip_max_mm: float = field(default=10000.0)
    depth_encoding: str = field(
        default="turbo",
        metadata={"help": "Depth preprocessing: gray, turbo, normals, turbo+normals."},
    )
    depth_intrinsics_filename: str = field(default="intrinsic_depth.txt")
    depth_auto_scale_intrinsics: bool = field(default=True)
    depth_use_all_frames: bool = field(default=False)
    depth_chunk_stride: Optional[int] = field(
        default=None,
        metadata={"help": "Stride (in frames) when depth_use_all_frames=True. Defaults to depth_num_frames."},
    )


@dataclass
class LynXLossArguments:
    dummy_prompt: str = field(default="<image>\tHint: Please answer the question and provide the final answer at the end.")
    dummy_placeholder_token: str = field(default="<image>")
    lambda_stat: float = field(default=0.01)
    lambda_attn: float = field(default=0.0)
    lambda_distill: float = field(default=1.0)
    distill_scope: Optional[str] = field(
        default=None,
        metadata={"help": "Attention distillation scope: 'vision' (vision tower only) or 'all' (vision + LLM)."},
    )

    use_traj_loss: bool = field(default=False)
    lambda_traj: float = field(default=0.0)
    traj_layers: str = field(default="all")  # "last" or "all"
    traj_ref_tokens: int = field(default=4096)


@dataclass
class LynXAVQAEvalArguments:
    enable_avqa_eval: bool = field(default=False)
    avqa_annotation_file: str = field(default="./data/video_instruction_tuning/avqa/val_qa.json")
    avqa_video_root: str = field(default="./data/video_instruction_tuning/avqa/videos")
    avqa_video_mapping: Optional[str] = field(default="./data/video_instruction_tuning/avqa/avqa_from_vid_to_video_name.json")
    avqa_num_frames: int = field(default=20)
    avqa_max_samples: Optional[int] = field(default=1500)
    avqa_eval_before_train: bool = field(default=False)
    avqa_eval_batch_size: int = field(default=1, metadata={"help": "Micro-batch size per GPU for AVQA evaluation."})

    avqa_pool_video_tokens: bool = field(default=False)
    avqa_frame_chunk_size: int = field(default=4)

    avqa_save_predictions: bool = field(default=True)
    avqa_predictions_dir: Optional[str] = field(default=None)
    avqa_save_metrics: bool = field(default=True)
    avqa_metrics_path: Optional[str] = field(default=None)


@dataclass
class LynXNoEvalArguments:
    enable_avqa_eval: bool = field(default=False)


@dataclass
class VisionLoraArguments:
    enable_vision_lora: bool = field(default=True)
    vision_lora_r: int = field(default=64)
    vision_lora_alpha: int = field(default=128)
    vision_lora_dropout: float = field(default=0.05)
    vision_lora_bias: str = field(default="none")


def _split_csv(value: Optional[str]) -> List[str]:
    if value is None:
        return []
    items = [v.strip() for v in str(value).split(",")]
    return [v for v in items if v]


def _iter_video_paths_from_mappings(mapping_paths: Sequence[str], video_roots: Sequence[str], *, max_items: Optional[int]) -> List[str]:
    if len(mapping_paths) != len(video_roots):
        raise ValueError(f"Expected mapping_paths and video_roots to have the same length, got {len(mapping_paths)} vs {len(video_roots)}")

    out: List[str] = []
    for mp, vr in zip(mapping_paths, video_roots):
        with open(mp, "r") as f:
            mapping = json.load(f)
        if not isinstance(mapping, dict):
            raise TypeError(f"Expected a dict in {mp}, got {type(mapping)}")

        for _, rel in mapping.items():
            out.append(os.path.join(vr, str(rel)))
            if max_items is not None and len(out) >= int(max_items):
                return out
    return out

def _load_rgb_frames_from_paths(frame_paths: Sequence[str]) -> List[Image.Image]:
    frames: List[Image.Image] = []
    for path in frame_paths:
        with Image.open(path) as im:
            frames.append(im.convert("RGB").copy())
    return frames

@torch.no_grad()
def _precompute_reference_stats_from_scene_dirs(
    *,
    model: Any,
    video_processor: Any,
    scene_dirs: Sequence[str],
    color_subdir: str,
    num_frames: int,
    dummy_prefix_ids: torch.Tensor,
    dummy_suffix_ids: torch.Tensor,
    vision_feature_layer: int,
    vision_feature_select_strategy: str,
    frame_chunk_size: int,
    disable_tqdm: bool,
    compute_vision_kv: bool,
    compute_llm_kv: bool,
    return_partial: bool = False,
    use_all_frames: bool = False,
    chunk_stride: Optional[int] = None,
) -> Dict[str, Any]:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    hidden_size = model.config.text_config.hidden_size
    running = running_mean_var_init(hidden_size, device=device)

    num_layers = model.config.text_config.num_hidden_layers
    ref_k_sum: Optional[torch.Tensor] = None
    ref_v_sum: Optional[torch.Tensor] = None

    vision_tower = get_vision_tower(model)
    vision_layers = get_siglip_encoder_layers(vision_tower)
    num_vision_layers = len(vision_layers)
    vision_k_sum: Optional[torch.Tensor] = None
    vision_v_sum: Optional[torch.Tensor] = None
    vision_seen_frames = 0
    vision_handles: List[Any] = []

    def make_vision_hook(layer_idx: int):
        def hook(module: Any, module_inputs: Tuple[Any, ...], module_kwargs: Dict[str, Any], module_outputs: Any) -> None:
            nonlocal vision_k_sum, vision_v_sum

            hidden_states = module_inputs[0] if len(module_inputs) > 0 else module_kwargs.get("hidden_states")
            if hidden_states is None:
                raise RuntimeError("Failed to capture vision attention hidden_states for reference KV cache (missing kwargs support).")
            k = module.k_proj(hidden_states)
            v = module.v_proj(hidden_states)
            batch_frames, seq_len, _ = k.shape

            num_heads = int(getattr(module, "num_heads", 0) or 0)
            head_dim = int(getattr(module, "head_dim", 0) or 0)
            if num_heads <= 0 or head_dim <= 0:
                raise ValueError("Could not infer SigLip attention num_heads/head_dim for reference KV cache.")

            k = k.view(batch_frames, seq_len, num_heads, head_dim).transpose(1, 2)
            v = v.view(batch_frames, seq_len, num_heads, head_dim).transpose(1, 2)

            if vision_k_sum is None or vision_v_sum is None:
                vision_k_sum = torch.zeros((num_vision_layers, num_heads, seq_len, head_dim), device=device, dtype=torch.float32)
                vision_v_sum = torch.zeros((num_vision_layers, num_heads, seq_len, head_dim), device=device, dtype=torch.float32)

            vision_k_sum[layer_idx].add_(k.to(dtype=torch.float32).sum(dim=0))
            vision_v_sum[layer_idx].add_(v.to(dtype=torch.float32).sum(dim=0))

        return hook

    if compute_vision_kv:
        for layer_idx, layer in enumerate(vision_layers):
            attn = getattr(layer, "self_attn", None)
            if attn is None:
                raise AttributeError(f"Vision encoder layer {layer_idx} does not expose `self_attn` for distillation.")
            hook_fn = make_vision_hook(layer_idx)
            try:
                vision_handles.append(attn.register_forward_hook(hook_fn, with_kwargs=True))
            except TypeError:
                def hook_no_kwargs(module: Any, module_inputs: Tuple[Any, ...], module_outputs: Any, *, _hook=hook_fn) -> None:
                    _hook(module, module_inputs, {}, module_outputs)

                vision_handles.append(attn.register_forward_hook(hook_no_kwargs))

    seen = 0
    pbar = tqdm(
        scene_dirs,
        desc="Precompute reference",
        total=len(scene_dirs),
        disable=disable_tqdm,
        leave=False,
        miniters=1,
        mininterval=0.0,
        dynamic_ncols=True,
    )
    try:
        for scene_dir in pbar:
            if not os.path.isdir(scene_dir):
                continue

            color_dir = os.path.join(scene_dir, color_subdir)
            color_paths = list_image_paths(color_dir, exts=(".jpg", ".jpeg", ".png"))
            if not color_paths:
                continue

            if use_all_frames:
                stride = int(chunk_stride) if chunk_stride is not None else int(num_frames)
                chunk_paths_list = [
                    pad_to_length(color_paths[start : start + num_frames], length=num_frames)
                    for start in chunk_starts(len(color_paths), chunk_size=num_frames, stride=stride)
                ]
            else:
                chunk_paths_list = [sample_uniform(color_paths, num_frames)]

            for chunk_paths in chunk_paths_list:
                frames = _load_rgb_frames_from_paths(chunk_paths)
                if not frames:
                    continue

                pv = video_processor([frames], return_tensors="pt")["pixel_values_videos"].to(device=device, dtype=dtype)
                z_v = encode_llava_onevision_video_tokens(
                    model,
                    pixel_values_videos=pv,
                    vision_feature_layer=vision_feature_layer,
                    vision_feature_select_strategy=vision_feature_select_strategy,
                    frame_chunk_size=frame_chunk_size,
                )
                running_mean_var_update(running, z_v.flatten(0, 1))

                if compute_llm_kv:
                    embed = model.get_input_embeddings()
                    prefix_emb = embed(dummy_prefix_ids.to(device)).unsqueeze(0)
                    suffix_emb = embed(dummy_suffix_ids.to(device)).unsqueeze(0)
                    inputs_embeds = torch.cat([prefix_emb, z_v, suffix_emb], dim=1)

                    seq_len = inputs_embeds.shape[1]
                    attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)
                    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

                    out = model(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        use_cache=True,
                        num_logits_to_keep=1,
                        return_dict=True,
                    )

                    v_start = int(prefix_emb.shape[1])
                    v_end = v_start + int(z_v.shape[1])
                    pkv = out.past_key_values
                    if pkv is None:
                        raise RuntimeError("Model did not return past_key_values with use_cache=True.")

                    if ref_k_sum is None or ref_v_sum is None:
                        k0, _v0 = pkv[0]
                        num_kv_heads = int(k0.shape[1])
                        head_dim = int(k0.shape[3])
                        ref_k_sum = torch.zeros((num_layers, num_kv_heads, v_end - v_start, head_dim), device=device, dtype=torch.float32)
                        ref_v_sum = torch.zeros((num_layers, num_kv_heads, v_end - v_start, head_dim), device=device, dtype=torch.float32)

                    for l in range(num_layers):
                        k_l, v_l = pkv[l]
                        ref_k_sum[l].add_(k_l[0, :, v_start:v_end, :].to(dtype=torch.float32))
                        ref_v_sum[l].add_(v_l[0, :, v_start:v_end, :].to(dtype=torch.float32))

                if compute_vision_kv:
                    vision_seen_frames += int(pv.shape[1])

                seen += 1
                pbar.set_postfix({"seen": seen})
    finally:
        for h in vision_handles:
            try:
                h.remove()
            except Exception:
                pass

    if seen == 0:
        if return_partial:
            return {
                "token_count": 0,
                "token_mean": running.mean.detach().cpu(),
                "token_m2": running.m2.detach().cpu(),
                "num_ref_videos": 0,
            }
        raise RuntimeError("Reference stats precompute found no frames on disk. Check reference paths.")

    if return_partial:
        payload: Dict[str, Any] = {
            "token_count": int(running.count),
            "token_mean": running.mean.detach().cpu(),
            "token_m2": running.m2.detach().cpu(),
            "num_ref_videos": int(seen),
        }

        if compute_llm_kv:
            if ref_k_sum is None or ref_v_sum is None:
                raise RuntimeError("Reference KV cache precompute did not record any samples.")
            payload["ref_k_sum"] = ref_k_sum.detach().cpu()
            payload["ref_v_sum"] = ref_v_sum.detach().cpu()

        if compute_vision_kv:
            if vision_seen_frames == 0 or vision_k_sum is None or vision_v_sum is None:
                raise RuntimeError("Vision reference KV cache precompute did not record any frames.")
            payload["vision_seen_frames"] = int(vision_seen_frames)
            payload["vision_k_sum"] = vision_k_sum.detach().cpu()
            payload["vision_v_sum"] = vision_v_sum.detach().cpu()

        return payload

    mean, var = running_mean_var_finalize(running)
    payload: Dict[str, Any] = {
        "token_mean": mean.cpu(),
        "token_var": var.cpu(),
        "num_ref_videos": seen,
    }

    if compute_llm_kv:
        if ref_k_sum is None or ref_v_sum is None:
            raise RuntimeError("Reference KV cache precompute did not record any samples.")
        payload["ref_k"] = (ref_k_sum / float(seen)).to(dtype=torch.float16).cpu()
        payload["ref_v"] = (ref_v_sum / float(seen)).to(dtype=torch.float16).cpu()

    if compute_vision_kv:
        if vision_seen_frames == 0 or vision_k_sum is None or vision_v_sum is None:
            raise RuntimeError("Vision reference KV cache precompute did not record any frames.")
        payload["vision_ref_k"] = (vision_k_sum / float(vision_seen_frames)).to(dtype=torch.float16).cpu()
        payload["vision_ref_v"] = (vision_v_sum / float(vision_seen_frames)).to(dtype=torch.float16).cpu()

    return payload


def _maybe_barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def _wait_for_file(path: str, *, poll_s: float = 5.0, timeout_s: float = 0.0, min_mtime: Optional[float] = None) -> None:
    start = time.time()
    while True:
        if os.path.exists(path):
            if min_mtime is None:
                return
            try:
                if os.path.getmtime(path) >= float(min_mtime):
                    return
            except OSError:
                pass
        if timeout_s and (time.time() - start) > timeout_s:
            raise TimeoutError(f"Timed out waiting for file: {path}")
        time.sleep(poll_s)


def _infer_rank_and_world_size(training_args: TrainingArguments) -> Tuple[int, int]:
    world_size_raw = os.environ.get("WORLD_SIZE", "1") or "1"
    try:
        world_size = int(world_size_raw)
    except ValueError:
        world_size = 1
    world_size = max(1, world_size)

    rank_env = os.environ.get("RANK")
    if rank_env is not None:
        try:
            rank = int(rank_env)
        except ValueError:
            rank = 0
    else:
        rank = int(getattr(training_args, "local_rank", -1))
        if rank < 0:
            rank = 0

    if rank < 0:
        rank = 0
    if rank >= world_size:
        rank = world_size - 1
    return rank, world_size


def _merge_mean_m2(
    *,
    count_a: int,
    mean_a: torch.Tensor,
    m2_a: torch.Tensor,
    count_b: int,
    mean_b: torch.Tensor,
    m2_b: torch.Tensor,
) -> Tuple[int, torch.Tensor, torch.Tensor]:
    if count_a == 0:
        return count_b, mean_b, m2_b
    if count_b == 0:
        return count_a, mean_a, m2_a

    total = int(count_a + count_b)
    delta = mean_b - mean_a
    mean = mean_a + delta * (float(count_b) / float(total))
    m2 = m2_a + m2_b + (delta**2) * (float(count_a) * float(count_b) / float(total))
    return total, mean, m2


def _aggregate_reference_partials(
    partial_paths: Sequence[str],
    *,
    compute_llm_kv: bool,
    compute_vision_kv: bool,
) -> Dict[str, Any]:
    total_seen_videos = 0

    token_count = 0
    token_mean: Optional[torch.Tensor] = None
    token_m2: Optional[torch.Tensor] = None

    ref_k_sum: Optional[torch.Tensor] = None
    ref_v_sum: Optional[torch.Tensor] = None

    vision_seen_frames = 0
    vision_k_sum: Optional[torch.Tensor] = None
    vision_v_sum: Optional[torch.Tensor] = None

    for path in partial_paths:
        part = torch.load(path, map_location="cpu")
        total_seen_videos += int(part.get("num_ref_videos") or 0)

        count_b = int(part.get("token_count") or 0)
        if count_b > 0:
            mean_b = part["token_mean"].to(dtype=torch.float64)
            m2_b = part["token_m2"].to(dtype=torch.float64)
            if token_mean is None or token_m2 is None or token_count == 0:
                token_count = count_b
                token_mean = mean_b
                token_m2 = m2_b
            else:
                token_count, token_mean, token_m2 = _merge_mean_m2(
                    count_a=token_count,
                    mean_a=token_mean,
                    m2_a=token_m2,
                    count_b=count_b,
                    mean_b=mean_b,
                    m2_b=m2_b,
                )

        if compute_llm_kv:
            k = part.get("ref_k_sum")
            v = part.get("ref_v_sum")
            if k is not None:
                if ref_k_sum is None:
                    ref_k_sum = k.to(dtype=torch.float32)
                else:
                    ref_k_sum.add_(k.to(dtype=torch.float32))
            if v is not None:
                if ref_v_sum is None:
                    ref_v_sum = v.to(dtype=torch.float32)
                else:
                    ref_v_sum.add_(v.to(dtype=torch.float32))

        if compute_vision_kv:
            vision_seen_frames += int(part.get("vision_seen_frames") or 0)
            k = part.get("vision_k_sum")
            v = part.get("vision_v_sum")
            if k is not None:
                if vision_k_sum is None:
                    vision_k_sum = k.to(dtype=torch.float32)
                else:
                    vision_k_sum.add_(k.to(dtype=torch.float32))
            if v is not None:
                if vision_v_sum is None:
                    vision_v_sum = v.to(dtype=torch.float32)
                else:
                    vision_v_sum.add_(v.to(dtype=torch.float32))

    if token_mean is None or token_m2 is None or token_count == 0:
        raise RuntimeError("Reference partial aggregation found no token statistics. Check reference paths.")
    if total_seen_videos <= 0:
        raise RuntimeError("Reference partial aggregation found no videos. Check reference paths.")

    mean = token_mean.to(dtype=torch.float32)
    var = (token_m2 / float(token_count)).to(dtype=torch.float32)

    payload: Dict[str, Any] = {
        "token_mean": mean.cpu(),
        "token_var": var.cpu(),
        "num_ref_videos": int(total_seen_videos),
    }

    if compute_llm_kv:
        if ref_k_sum is None or ref_v_sum is None:
            raise RuntimeError("Reference partial aggregation missing LLM KV sums.")
        payload["ref_k"] = (ref_k_sum / float(total_seen_videos)).to(dtype=torch.float16).cpu()
        payload["ref_v"] = (ref_v_sum / float(total_seen_videos)).to(dtype=torch.float16).cpu()

    if compute_vision_kv:
        if vision_seen_frames <= 0 or vision_k_sum is None or vision_v_sum is None:
            raise RuntimeError("Reference partial aggregation missing vision KV sums.")
        payload["vision_ref_k"] = (vision_k_sum / float(vision_seen_frames)).to(dtype=torch.float16).cpu()
        payload["vision_ref_v"] = (vision_v_sum / float(vision_seen_frames)).to(dtype=torch.float16).cpu()

    return payload


@torch.no_grad()
def _precompute_reference_stats(
    *,
    model: Any,
    video_processor: Any,
    video_paths: Sequence[str],
    num_frames: int,
    dummy_prefix_ids: torch.Tensor,
    dummy_suffix_ids: torch.Tensor,
    compute_traj: bool,
    traj_layers: str,
    traj_ref_tokens: int,
    video_backend: str,
    vision_feature_layer: int,
    vision_feature_select_strategy: str,
    frame_chunk_size: int,
    disable_tqdm: bool,
    compute_vision_kv: bool,
    compute_llm_kv: bool,
    return_partial: bool = False,
) -> Dict[str, Any]:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    hidden_size = model.config.text_config.hidden_size
    running = running_mean_var_init(hidden_size, device=device)

    num_layers = model.config.text_config.num_hidden_layers
    ref_k_sum: Optional[torch.Tensor] = None
    ref_v_sum: Optional[torch.Tensor] = None

    vision_tower = get_vision_tower(model)
    vision_layers = get_siglip_encoder_layers(vision_tower)
    num_vision_layers = len(vision_layers)
    vision_k_sum: Optional[torch.Tensor] = None
    vision_v_sum: Optional[torch.Tensor] = None
    vision_seen_frames = 0
    vision_handles: List[Any] = []

    def make_vision_hook(layer_idx: int):
        def hook(module: Any, module_inputs: Tuple[Any, ...], module_kwargs: Dict[str, Any], module_outputs: Any) -> None:
            nonlocal vision_k_sum, vision_v_sum

            hidden_states = module_inputs[0] if len(module_inputs) > 0 else module_kwargs.get("hidden_states")
            if hidden_states is None:
                raise RuntimeError("Failed to capture vision attention hidden_states for reference KV cache (missing kwargs support).")
            k = module.k_proj(hidden_states)
            v = module.v_proj(hidden_states)
            batch_frames, seq_len, _ = k.shape

            num_heads = int(getattr(module, "num_heads", 0) or 0)
            head_dim = int(getattr(module, "head_dim", 0) or 0)
            if num_heads <= 0 or head_dim <= 0:
                raise ValueError("Could not infer SigLip attention num_heads/head_dim for reference KV cache.")

            k = k.view(batch_frames, seq_len, num_heads, head_dim).transpose(1, 2)
            v = v.view(batch_frames, seq_len, num_heads, head_dim).transpose(1, 2)

            if vision_k_sum is None or vision_v_sum is None:
                vision_k_sum = torch.zeros((num_vision_layers, num_heads, seq_len, head_dim), device=device, dtype=torch.float32)
                vision_v_sum = torch.zeros((num_vision_layers, num_heads, seq_len, head_dim), device=device, dtype=torch.float32)

            vision_k_sum[layer_idx].add_(k.to(dtype=torch.float32).sum(dim=0))
            vision_v_sum[layer_idx].add_(v.to(dtype=torch.float32).sum(dim=0))

        return hook

    if compute_vision_kv:
        for layer_idx, layer in enumerate(vision_layers):
            attn = getattr(layer, "self_attn", None)
            if attn is None:
                raise AttributeError(f"Vision encoder layer {layer_idx} does not expose `self_attn` for distillation.")
            hook_fn = make_vision_hook(layer_idx)
            try:
                vision_handles.append(attn.register_forward_hook(hook_fn, with_kwargs=True))
            except TypeError:
                def hook_no_kwargs(module: Any, module_inputs: Tuple[Any, ...], module_outputs: Any, *, _hook=hook_fn) -> None:
                    _hook(module, module_inputs, {}, module_outputs)

                vision_handles.append(attn.register_forward_hook(hook_no_kwargs))

    seen = 0
    pbar = tqdm(
        video_paths,
        desc="Precompute reference",
        total=len(video_paths),
        disable=disable_tqdm,
        leave=False,
        miniters=1,
        mininterval=0.0,
        dynamic_ncols=True,
    )
    try:
        for vp in pbar:
            if not os.path.exists(vp):
                continue

            try:
                vframes, _, _ = load_video_frames_and_audio(vp, num_frames=num_frames, video_backend=video_backend, decode_audio=False)
            except Exception:
                continue
            frames = [vframes[i].permute(1, 2, 0).numpy() for i in range(vframes.shape[0])]
            pv = video_processor([frames], return_tensors="pt")["pixel_values_videos"].to(device=device, dtype=dtype)

            z_v = encode_llava_onevision_video_tokens(
                model,
                pixel_values_videos=pv,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
                frame_chunk_size=frame_chunk_size,
            )
            running_mean_var_update(running, z_v.flatten(0, 1))

            if compute_llm_kv:
                embed = model.get_input_embeddings()
                prefix_emb = embed(dummy_prefix_ids.to(device)).unsqueeze(0)
                suffix_emb = embed(dummy_suffix_ids.to(device)).unsqueeze(0)
                inputs_embeds = torch.cat([prefix_emb, z_v, suffix_emb], dim=1)

                seq_len = inputs_embeds.shape[1]
                attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)
                position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

                out = model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=True,
                    num_logits_to_keep=1,
                    return_dict=True,
                )

                v_start = int(prefix_emb.shape[1])
                v_end = v_start + int(z_v.shape[1])
                pkv = out.past_key_values
                if pkv is None:
                    raise RuntimeError("Model did not return past_key_values with use_cache=True.")

                if ref_k_sum is None or ref_v_sum is None:
                    k0, _v0 = pkv[0]
                    num_kv_heads = int(k0.shape[1])
                    head_dim = int(k0.shape[3])
                    ref_k_sum = torch.zeros((num_layers, num_kv_heads, v_end - v_start, head_dim), device=device, dtype=torch.float32)
                    ref_v_sum = torch.zeros((num_layers, num_kv_heads, v_end - v_start, head_dim), device=device, dtype=torch.float32)

                for l in range(num_layers):
                    k_l, v_l = pkv[l]
                    ref_k_sum[l].add_(k_l[0, :, v_start:v_end, :].to(dtype=torch.float32))
                    ref_v_sum[l].add_(v_l[0, :, v_start:v_end, :].to(dtype=torch.float32))

            if compute_vision_kv:
                vision_seen_frames += int(pv.shape[1])

            seen += 1
            pbar.set_postfix({"seen": seen})
    finally:
        for h in vision_handles:
            try:
                h.remove()
            except Exception:
                pass

    if seen == 0:
        if return_partial:
            return {
                "token_count": 0,
                "token_mean": running.mean.detach().cpu(),
                "token_m2": running.m2.detach().cpu(),
                "num_ref_videos": 0,
            }
        raise RuntimeError("Reference stats precompute found no videos on disk. Check reference paths.")

    if return_partial:
        payload: Dict[str, Any] = {
            "token_count": int(running.count),
            "token_mean": running.mean.detach().cpu(),
            "token_m2": running.m2.detach().cpu(),
            "num_ref_videos": int(seen),
        }

        if compute_llm_kv:
            if ref_k_sum is None or ref_v_sum is None:
                raise RuntimeError("Reference KV cache precompute did not record any samples.")
            payload["ref_k_sum"] = ref_k_sum.detach().cpu()
            payload["ref_v_sum"] = ref_v_sum.detach().cpu()

        if compute_vision_kv:
            if vision_seen_frames == 0 or vision_k_sum is None or vision_v_sum is None:
                raise RuntimeError("Vision reference KV cache precompute did not record any frames.")
            payload["vision_seen_frames"] = int(vision_seen_frames)
            payload["vision_k_sum"] = vision_k_sum.detach().cpu()
            payload["vision_v_sum"] = vision_v_sum.detach().cpu()

        return payload

    mean, var = running_mean_var_finalize(running)
    payload: Dict[str, Any] = {
        "token_mean": mean.cpu(),
        "token_var": var.cpu(),
        "num_ref_videos": seen,
    }

    if compute_llm_kv:
        if ref_k_sum is None or ref_v_sum is None:
            raise RuntimeError("Reference KV cache precompute did not record any samples.")
        payload["ref_k"] = (ref_k_sum / float(seen)).to(dtype=torch.float16).cpu()
        payload["ref_v"] = (ref_v_sum / float(seen)).to(dtype=torch.float16).cpu()

    if compute_vision_kv:
        if vision_seen_frames == 0 or vision_k_sum is None or vision_v_sum is None:
            raise RuntimeError("Vision reference KV cache precompute did not record any frames.")
        payload["vision_ref_k"] = (vision_k_sum / float(vision_seen_frames)).to(dtype=torch.float16).cpu()
        payload["vision_ref_v"] = (vision_v_sum / float(vision_seen_frames)).to(dtype=torch.float16).cpu()

    return payload


def _find_vision_linear_modules(model: Any) -> List[str]:
    names: List[str] = []
    for name, module in model.named_modules():
        if ".vision_tower." not in f".{name}.":
            continue
        if isinstance(module, torch.nn.Linear):
            names.append(name)
    return sorted(set(names))


def apply_vision_lora(model: Any, lora_args: VisionLoraArguments) -> Any:
    if not lora_args.enable_vision_lora:
        return model

    from peft import LoraConfig, get_peft_model

    target_modules = _find_vision_linear_modules(model)
    if not target_modules:
        raise RuntimeError("No vision linear layers found for LoRA injection.")

    lora_config = LoraConfig(
        r=lora_args.vision_lora_r,
        lora_alpha=lora_args.vision_lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_args.vision_lora_dropout,
        bias=lora_args.vision_lora_bias,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    for n, p in model.named_parameters():
        p.requires_grad = "lora_" in n

    return model


def _maybe_disable_hf_offline(model_args: LynXModelArguments) -> None:
    if not model_args.local_files_only:
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
            if str(os.environ.get(name, "")).strip() in ("1", "true", "True", "YES", "yes"):
                os.environ.pop(name, None)


def _resolve_distill_needs(loss_args: LynXLossArguments, *, default_scope: str) -> Tuple[bool, bool]:
    distill_scope = str(getattr(loss_args, "distill_scope", default_scope) or default_scope).lower()
    if distill_scope not in ("vision", "all"):
        raise ValueError(f"Unsupported --distill_scope {distill_scope!r}. Expected one of: vision, all.")
    need_llm_kv = bool(loss_args.lambda_distill > 0.0 and distill_scope == "all")
    need_vision_kv = bool(loss_args.lambda_distill > 0.0)
    return need_llm_kv, need_vision_kv


def _load_model_and_processors(
    *,
    model_args: LynXModelArguments,
    training_args: TrainingArguments,
) -> Tuple[Any, Any, Any, Any, torch.device, int, str]:
    resolved_model_path = resolve_local_model_path(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        local_files_only=model_args.local_files_only,
    )

    processor = AutoProcessor.from_pretrained(
        resolved_model_path,
        cache_dir=model_args.cache_dir,
        local_files_only=model_args.local_files_only,
    )
    tokenizer = processor.tokenizer
    video_processor = processor.video_processor

    device = torch.device(training_args.device)
    torch_dtype = torch.bfloat16 if (device.type == "cuda" and training_args.bf16) else (torch.float16 if device.type == "cuda" else torch.float32)

    attn_impl = resolve_attn_implementation(bool(model_args.use_flash_attn), device=device)
    if bool(model_args.use_flash_attn) and attn_impl != "flash_attention_2":
        rank, _ = _infer_rank_and_world_size(training_args)
        if rank == 0:
            print("[WARN] FlashAttention2 requested but not available; falling back to SDPA.")

    device_map = {"": str(device)} if device.type == "cuda" else None
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        resolved_model_path,
        cache_dir=model_args.cache_dir,
        local_files_only=model_args.local_files_only,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map=device_map,
        attn_implementation=attn_impl,
    )
    if device_map is None:
        model.to(device)

    vision_feature_layer = int(getattr(model.config, "vision_feature_layer", 0) or 0)
    vision_feature_select_strategy = str(getattr(model.config, "vision_feature_select_strategy", "default") or "default")

    return model, processor, tokenizer, video_processor, device, vision_feature_layer, vision_feature_select_strategy


def _build_dummy_prefix_suffix_ids(tokenizer: Any, loss_args: LynXLossArguments) -> Tuple[torch.Tensor, torch.Tensor]:
    dummy_ids, placeholder_pos = build_dummy_prompt_token_ids(
        tokenizer,
        prompt_text=loss_args.dummy_prompt,
        placeholder_token=loss_args.dummy_placeholder_token,
    )
    dummy_prefix_ids = dummy_ids[:placeholder_pos]
    dummy_suffix_ids = dummy_ids[placeholder_pos + 1 :]
    return dummy_prefix_ids, dummy_suffix_ids


def _reference_stats_needs_recompute(
    ref_path: str,
    *,
    lambda_distill: float,
    need_llm_kv: bool,
    need_vision_kv: bool,
) -> bool:
    if not os.path.exists(ref_path):
        return True

    try:
        probe = torch.load(ref_path, map_location="cpu")
    except Exception:
        return True

    if not isinstance(probe, dict):
        return True
    if "token_mean" not in probe or "token_var" not in probe:
        return True

    if float(lambda_distill) > 0.0:
        if need_llm_kv and (probe.get("ref_k") is None or probe.get("ref_v") is None):
            return True
        if need_vision_kv and (probe.get("vision_ref_k") is None or probe.get("vision_ref_v") is None):
            return True

    return False


def _load_or_create_reference_stats(
    *,
    ref_path: str,
    model: Any,
    training_args: TrainingArguments,
    loss_args: LynXLossArguments,
    need_llm_kv: bool,
    need_vision_kv: bool,
    compute_stats: Callable[[bool, bool], Dict[str, Any]],
) -> Dict[str, Any]:
    ref_dir = os.path.dirname(ref_path)
    if ref_dir:
        os.makedirs(ref_dir, exist_ok=True)

    rank, world_size = _infer_rank_and_world_size(training_args)
    need_ref = _reference_stats_needs_recompute(
        ref_path,
        lambda_distill=float(loss_args.lambda_distill),
        need_llm_kv=need_llm_kv,
        need_vision_kv=need_vision_kv,
    )

    if need_ref:
        disable_tqdm = bool(training_args.disable_tqdm or (world_size > 1 and rank != 0))
        partials_min_mtime = time.time() if world_size > 1 else None

        model.eval()
        ref_or_partial = compute_stats(world_size > 1, disable_tqdm)
        model.train()

        if world_size > 1:
            partial_dir = f"{ref_path}.partials_ws{world_size}"
            os.makedirs(partial_dir, exist_ok=True)
            partial_path = os.path.join(partial_dir, f"rank{rank}.pt")
            tmp_partial_path = partial_path + ".tmp"
            torch.save(ref_or_partial, tmp_partial_path)
            os.replace(tmp_partial_path, partial_path)

            if rank == 0:
                partial_paths = [os.path.join(partial_dir, f"rank{r}.pt") for r in range(world_size)]
                for path in partial_paths:
                    _wait_for_file(path, poll_s=5.0, timeout_s=0.0, min_mtime=partials_min_mtime)
                ref = _aggregate_reference_partials(partial_paths, compute_llm_kv=need_llm_kv, compute_vision_kv=need_vision_kv)
                tmp_ref_path = ref_path + ".tmp"
                torch.save(ref, tmp_ref_path)
                os.replace(tmp_ref_path, ref_path)

                for path in partial_paths:
                    try:
                        os.remove(path)
                    except OSError:
                        pass
                try:
                    os.rmdir(partial_dir)
                except OSError:
                    pass
            else:
                _wait_for_file(ref_path, poll_s=5.0, timeout_s=0.0)
        else:
            tmp_ref_path = ref_path + ".tmp"
            torch.save(ref_or_partial, tmp_ref_path)
            os.replace(tmp_ref_path, ref_path)

    if world_size > 1 and rank != 0:
        _wait_for_file(ref_path, poll_s=5.0, timeout_s=0.0)
    _maybe_barrier()

    ref = torch.load(ref_path, map_location="cpu")
    if not isinstance(ref, dict):
        raise TypeError(f"Expected reference stats at {ref_path} to be a dict, got {type(ref)}")
    if "token_mean" not in ref or "token_var" not in ref:
        raise KeyError(f"Reference stats file is missing token_mean/token_var: {ref_path}")

    if float(loss_args.lambda_distill) > 0.0:
        if need_llm_kv and (ref.get("ref_k") is None or ref.get("ref_v") is None):
            raise KeyError("Reference stats file is missing 'ref_k'/'ref_v' for attention distillation.")
        if need_vision_kv and (ref.get("vision_ref_k") is None or ref.get("vision_ref_v") is None):
            raise KeyError("Reference stats file is missing 'vision_ref_k'/'vision_ref_v' for vision attention distillation.")

    return ref


def _parse_resume_from_checkpoint(training_args: TrainingArguments) -> Any:
    resume_arg = getattr(training_args, "resume_from_checkpoint", None)
    if resume_arg is None:
        return None

    if isinstance(resume_arg, str):
        resume_str = resume_arg.strip()
        if resume_str.lower() in ("", "none", "null", "false", "0", "no"):
            return None
        if resume_str.lower() in ("true", "1", "yes"):
            return True
        return resume_str

    return resume_arg


def _make_trainer(
    *,
    model: Any,
    training_args: TrainingArguments,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset],
    data_collator: Any,
    tokenizer: Any,
    processor: Any,
    avqa_args: Any,
    loss_args: LynXLossArguments,
    audio_target_sr: int,
    audio_seconds: float,
    mel_n_mels: int,
    mel_n_fft: int,
    mel_hop_length: int,
    mel_win_length: int,
    ref_token_mean: torch.Tensor,
    ref_token_var: torch.Tensor,
    ref_k: Optional[torch.Tensor],
    ref_v: Optional[torch.Tensor],
    vision_ref_k: Optional[torch.Tensor],
    vision_ref_v: Optional[torch.Tensor],
    dummy_prefix_ids: torch.Tensor,
    dummy_suffix_ids: torch.Tensor,
    audio_repeat_frames: Optional[int],
    vision_feature_layer: int,
    vision_feature_select_strategy: str,
) -> LynXAVQATrainer:
    return LynXAVQATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        processor=processor,
        avqa_args=avqa_args,
        loss_args=loss_args,
        audio_target_sr=audio_target_sr,
        audio_seconds=audio_seconds,
        mel_n_mels=mel_n_mels,
        mel_n_fft=mel_n_fft,
        mel_hop_length=mel_hop_length,
        mel_win_length=mel_win_length,
        ref_token_mean=ref_token_mean,
        ref_token_var=ref_token_var,
        ref_k=ref_k,
        ref_v=ref_v,
        vision_ref_k=vision_ref_k,
        vision_ref_v=vision_ref_v,
        dummy_prefix_ids=dummy_prefix_ids,
        dummy_suffix_ids=dummy_suffix_ids,
        audio_repeat_frames=audio_repeat_frames,
        vision_feature_layer=vision_feature_layer,
        vision_feature_select_strategy=vision_feature_select_strategy,
    )


def _main_audio(argv: Optional[Sequence[str]] = None) -> None:
    parser = HfArgumentParser((LynXModelArguments, LynXDataArguments, LynXLossArguments, LynXAVQAEvalArguments, VisionLoraArguments, TrainingArguments))
    if argv is None:
        model_args, data_args, loss_args, avqa_args, lora_args, training_args = parser.parse_args_into_dataclasses()
    else:
        model_args, data_args, loss_args, avqa_args, lora_args, training_args = parser.parse_args_into_dataclasses(args=list(argv))

    _maybe_disable_hf_offline(model_args)

    if training_args.deepspeed:
        maybe_patch_torch_autograd_graph_for_deepspeed()

    if training_args.output_dir is None:
        raise ValueError("--output_dir is required")
    os.makedirs(training_args.output_dir, exist_ok=True)
    training_args.remove_unused_columns = False

    set_seed(training_args.seed)

    model, processor, tokenizer, video_processor, device, vision_feature_layer, vision_feature_select_strategy = _load_model_and_processors(
        model_args=model_args,
        training_args=training_args,
    )
    dummy_prefix_ids, dummy_suffix_ids = _build_dummy_prefix_suffix_ids(tokenizer, loss_args)

    need_llm_kv, need_vision_kv = _resolve_distill_needs(loss_args, default_scope="all")

    ref_path = data_args.reference_stats_path or os.path.join(training_args.output_dir, "reference_stats.pt")

    def compute_ref(return_partial: bool, disable_tqdm: bool) -> Dict[str, Any]:
        if data_args.reference_video_mapping:
            ref_video_paths = iter_video_paths_from_mapping(
                data_args.reference_video_mapping,
                data_args.reference_video_root,
                max_items=data_args.reference_max_videos,
            )
        else:
            ref_video_paths = iter_video_paths_from_dir(data_args.reference_video_root, max_items=data_args.reference_max_videos)

        rank, world_size = _infer_rank_and_world_size(training_args)
        ref_video_paths = sorted(ref_video_paths)
        shard_video_paths = ref_video_paths[rank::world_size] if world_size > 1 else ref_video_paths

        return _precompute_reference_stats(
            model=model,
            video_processor=video_processor,
            video_paths=shard_video_paths,
            num_frames=int(data_args.reference_num_frames),
            dummy_prefix_ids=dummy_prefix_ids,
            dummy_suffix_ids=dummy_suffix_ids,
            compute_traj=bool(loss_args.use_traj_loss and loss_args.lambda_traj > 0),
            traj_layers=str(loss_args.traj_layers),
            traj_ref_tokens=int(loss_args.traj_ref_tokens),
            video_backend=str(data_args.video_backend),
            vision_feature_layer=int(vision_feature_layer),
            vision_feature_select_strategy=str(vision_feature_select_strategy),
            frame_chunk_size=int(data_args.vision_frame_chunk_size),
            disable_tqdm=disable_tqdm,
            compute_vision_kv=need_vision_kv,
            compute_llm_kv=need_llm_kv,
            return_partial=return_partial,
        )

    ref = _load_or_create_reference_stats(
        ref_path=ref_path,
        model=model,
        training_args=training_args,
        loss_args=loss_args,
        need_llm_kv=need_llm_kv,
        need_vision_kv=need_vision_kv,
        compute_stats=compute_ref,
    )

    ref_token_mean = ref["token_mean"]
    ref_token_var = ref["token_var"]
    ref_k = ref.get("ref_k")
    ref_v = ref.get("ref_v")
    vision_ref_k = ref.get("vision_ref_k")
    vision_ref_v = ref.get("vision_ref_v")

    model = apply_vision_lora(model, lora_args)

    train_video_paths: List[str]
    if data_args.train_annotation_file and os.path.exists(data_args.train_annotation_file):
        with open(data_args.train_annotation_file, "r") as f:
            ann = json.load(f)
        if not isinstance(ann, list):
            raise TypeError(f"Expected a list in {data_args.train_annotation_file}, got {type(ann)}")

        train_vids: List[str] = []
        seen_vids: set[str] = set()

        for item in ann:
            if not isinstance(item, dict):
                continue
            vid = item.get("video_name") or item.get("video") or item.get("video_id") or item.get("image_id")
            if not isinstance(vid, str) or not vid:
                continue
            if vid in seen_vids:
                continue
            seen_vids.add(vid)
            train_vids.append(vid)

        mapping: Optional[Dict[str, str]] = None
        if data_args.train_video_mapping and os.path.exists(data_args.train_video_mapping):
            with open(data_args.train_video_mapping, "r") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                mapping = raw

        train_video_paths = []
        for vid in train_vids:
            if ".mp4" not in vid:
                filename = mapping.get(vid, f"{vid}.mp4") if isinstance(mapping, dict) else f"{vid}.mp4"
            else:
                filename = vid
            train_video_paths.append(os.path.join(data_args.train_video_root, filename))
    else:
        if data_args.train_video_mapping:
            train_video_paths = iter_video_paths_from_mapping(data_args.train_video_mapping, data_args.train_video_root, max_items=data_args.train_max_videos)
        else:
            train_video_paths = iter_video_paths_from_dir(data_args.train_video_root, max_items=data_args.train_max_videos)

    random.shuffle(train_video_paths)
    if data_args.train_max_videos is not None:
        train_video_paths = train_video_paths[: data_args.train_max_videos]

    train_dataset = AudioFromVideoDataset(
        train_video_paths,
        video_processor=video_processor,
        video_backend=data_args.video_backend,
        audio_target_sr=data_args.audio_target_sr,
        audio_seconds=data_args.audio_seconds,
        mel_n_mels=data_args.mel_n_mels,
        mel_n_fft=data_args.mel_n_fft,
        mel_hop_length=data_args.mel_hop_length,
        mel_win_length=data_args.mel_win_length,
    )
    if training_args.local_rank in (-1, 0):
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        per_device = int(training_args.per_device_train_batch_size)
        grad_acc = int(training_args.gradient_accumulation_steps)
        global_batch = per_device * max(1, world_size) * max(1, grad_acc)
        steps_floor = len(train_dataset) // global_batch if global_batch > 0 else 0
        steps_ceil = (len(train_dataset) + global_batch - 1) // global_batch if global_batch > 0 else 0
        print(
            f"[LynX] train videos={len(train_dataset)} world_size={world_size} "
            f"per_device_batch={per_device} grad_acc={grad_acc} global_batch={global_batch} "
            f"estimated_opt_steps_per_epoch={steps_floor} (ceil={steps_ceil})"
        )

    eval_dataset: Optional[Dataset] = train_dataset if training_args.eval_strategy != IntervalStrategy.NO else None

    trainer = _make_trainer(
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=LynXDataCollator(),
        tokenizer=tokenizer,
        processor=processor,
        avqa_args=avqa_args,
        loss_args=loss_args,
        audio_target_sr=int(data_args.audio_target_sr),
        audio_seconds=float(data_args.audio_seconds),
        mel_n_mels=int(data_args.mel_n_mels),
        mel_n_fft=int(data_args.mel_n_fft),
        mel_hop_length=int(data_args.mel_hop_length),
        mel_win_length=int(data_args.mel_win_length),
        ref_token_mean=ref_token_mean,
        ref_token_var=ref_token_var,
        ref_k=ref_k,
        ref_v=ref_v,
        vision_ref_k=vision_ref_k,
        vision_ref_v=vision_ref_v,
        dummy_prefix_ids=dummy_prefix_ids,
        dummy_suffix_ids=dummy_suffix_ids,
        audio_repeat_frames=data_args.audio_repeat_frames,
        vision_feature_layer=int(vision_feature_layer),
        vision_feature_select_strategy=str(vision_feature_select_strategy),
    )

    if avqa_args.avqa_eval_before_train:
        trainer.evaluate()

    trainer.train(resume_from_checkpoint=_parse_resume_from_checkpoint(training_args))

    # trainer.save_state()
    trainer.save_model(training_args.output_dir)

    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()


def _main_egoexo(argv: Optional[Sequence[str]] = None) -> None:
    parser = HfArgumentParser((LynXModelArguments, LynXEgoExoDataArguments, LynXLossArguments, LynXNoEvalArguments, VisionLoraArguments, TrainingArguments))
    if argv is None:
        model_args, data_args, loss_args, no_eval_args, lora_args, training_args = parser.parse_args_into_dataclasses()
    else:
        model_args, data_args, loss_args, no_eval_args, lora_args, training_args = parser.parse_args_into_dataclasses(args=list(argv))

    _maybe_disable_hf_offline(model_args)

    if training_args.deepspeed:
        maybe_patch_torch_autograd_graph_for_deepspeed()

    if training_args.output_dir is None:
        raise ValueError("--output_dir is required")
    os.makedirs(training_args.output_dir, exist_ok=True)
    training_args.remove_unused_columns = False

    set_seed(training_args.seed)

    model, processor, tokenizer, video_processor, device, vision_feature_layer, vision_feature_select_strategy = _load_model_and_processors(
        model_args=model_args,
        training_args=training_args,
    )
    dummy_prefix_ids, dummy_suffix_ids = _build_dummy_prefix_suffix_ids(tokenizer, loss_args)
    need_llm_kv, need_vision_kv = _resolve_distill_needs(loss_args, default_scope="vision")

    ref_path = data_args.reference_stats_path or os.path.join(training_args.output_dir, "reference_stats.pt")

    def compute_ref(return_partial: bool, disable_tqdm: bool) -> Dict[str, Any]:
        if data_args.reference_video_mapping:
            ref_video_paths = iter_video_paths_from_mapping(
                data_args.reference_video_mapping,
                data_args.reference_video_root,
                max_items=data_args.reference_max_videos,
            )
        else:
            ref_video_paths = iter_video_paths_from_dir(data_args.reference_video_root, max_items=data_args.reference_max_videos)

        rank, world_size = _infer_rank_and_world_size(training_args)
        ref_video_paths = sorted(ref_video_paths)
        shard_video_paths = ref_video_paths[rank::world_size] if world_size > 1 else ref_video_paths

        return _precompute_reference_stats(
            model=model,
            video_processor=video_processor,
            video_paths=shard_video_paths,
            num_frames=int(data_args.reference_num_frames),
            dummy_prefix_ids=dummy_prefix_ids,
            dummy_suffix_ids=dummy_suffix_ids,
            compute_traj=bool(loss_args.use_traj_loss and loss_args.lambda_traj > 0),
            traj_layers=str(loss_args.traj_layers),
            traj_ref_tokens=int(loss_args.traj_ref_tokens),
            video_backend=str(data_args.video_backend),
            vision_feature_layer=int(vision_feature_layer),
            vision_feature_select_strategy=str(vision_feature_select_strategy),
            frame_chunk_size=int(data_args.vision_frame_chunk_size),
            disable_tqdm=disable_tqdm,
            compute_vision_kv=need_vision_kv,
            compute_llm_kv=need_llm_kv,
            return_partial=return_partial,
        )

    ref = _load_or_create_reference_stats(
        ref_path=ref_path,
        model=model,
        training_args=training_args,
        loss_args=loss_args,
        need_llm_kv=need_llm_kv,
        need_vision_kv=need_vision_kv,
        compute_stats=compute_ref,
    )

    ref_token_mean = ref["token_mean"]
    ref_token_var = ref["token_var"]
    ref_k = ref.get("ref_k")
    ref_v = ref.get("ref_v")
    vision_ref_k = ref.get("vision_ref_k")
    vision_ref_v = ref.get("vision_ref_v")

    model = apply_vision_lora(model, lora_args)

    if data_args.train_video_mapping:
        train_video_paths = iter_video_paths_from_mapping(
            data_args.train_video_mapping,
            data_args.train_video_root,
            max_items=data_args.train_max_videos,
        )
    else:
        train_video_paths = iter_video_paths_from_dir(
            data_args.train_video_root,
            max_items=data_args.train_max_videos,
        )

    random.shuffle(train_video_paths)
    if data_args.train_max_videos is not None:
        train_video_paths = train_video_paths[: int(data_args.train_max_videos)]

    train_num_frames = int(data_args.train_num_frames) if data_args.train_num_frames is not None else int(data_args.reference_num_frames)
    train_dataset = VideoFromVideoDataset(
        train_video_paths,
        video_processor=video_processor,
        video_backend=str(data_args.video_backend),
        num_frames=train_num_frames,
    )

    eval_dataset: Optional[Dataset] = train_dataset if training_args.eval_strategy != IntervalStrategy.NO else None

    trainer = _make_trainer(
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=VideoFromVideoCollator(),
        tokenizer=tokenizer,
        processor=processor,
        avqa_args=no_eval_args,
        loss_args=loss_args,
        audio_target_sr=16000,
        audio_seconds=8.0,
        mel_n_mels=128,
        mel_n_fft=1024,
        mel_hop_length=256,
        mel_win_length=1024,
        ref_token_mean=ref_token_mean,
        ref_token_var=ref_token_var,
        ref_k=ref_k,
        ref_v=ref_v,
        vision_ref_k=vision_ref_k,
        vision_ref_v=vision_ref_v,
        dummy_prefix_ids=dummy_prefix_ids,
        dummy_suffix_ids=dummy_suffix_ids,
        audio_repeat_frames=None,
        vision_feature_layer=int(vision_feature_layer),
        vision_feature_select_strategy=str(vision_feature_select_strategy),
    )

    trainer.train(resume_from_checkpoint=_parse_resume_from_checkpoint(training_args))
    trainer.save_model(training_args.output_dir)

    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()


def _main_fastvideo(argv: Optional[Sequence[str]] = None) -> None:
    parser = HfArgumentParser((LynXModelArguments, LynXFastVideoDataArguments, LynXLossArguments, LynXNoEvalArguments, VisionLoraArguments, TrainingArguments))
    if argv is None:
        model_args, data_args, loss_args, no_eval_args, lora_args, training_args = parser.parse_args_into_dataclasses()
    else:
        model_args, data_args, loss_args, no_eval_args, lora_args, training_args = parser.parse_args_into_dataclasses(args=list(argv))

    _maybe_disable_hf_offline(model_args)

    if training_args.deepspeed:
        maybe_patch_torch_autograd_graph_for_deepspeed()

    if training_args.output_dir is None:
        raise ValueError("--output_dir is required")
    os.makedirs(training_args.output_dir, exist_ok=True)
    training_args.remove_unused_columns = False

    set_seed(training_args.seed)

    model, processor, tokenizer, video_processor, device, vision_feature_layer, vision_feature_select_strategy = _load_model_and_processors(
        model_args=model_args,
        training_args=training_args,
    )
    dummy_prefix_ids, dummy_suffix_ids = _build_dummy_prefix_suffix_ids(tokenizer, loss_args)
    need_llm_kv, need_vision_kv = _resolve_distill_needs(loss_args, default_scope="vision")

    ref_path = data_args.reference_stats_path or os.path.join(training_args.output_dir, "reference_stats.pt")

    ref_roots = _split_csv(data_args.reference_video_roots)
    ref_maps = _split_csv(data_args.reference_video_mappings)

    def compute_ref(return_partial: bool, disable_tqdm: bool) -> Dict[str, Any]:
        if not ref_roots or not ref_maps:
            raise ValueError("--reference_video_roots and --reference_video_mappings are required to precompute reference stats.")

        ref_video_paths = _iter_video_paths_from_mappings(ref_maps, ref_roots, max_items=data_args.reference_max_videos)

        rank, world_size = _infer_rank_and_world_size(training_args)
        ref_video_paths = sorted(ref_video_paths)
        shard_video_paths = ref_video_paths[rank::world_size] if world_size > 1 else ref_video_paths

        return _precompute_reference_stats(
            model=model,
            video_processor=video_processor,
            video_paths=shard_video_paths,
            num_frames=int(data_args.reference_num_frames),
            dummy_prefix_ids=dummy_prefix_ids,
            dummy_suffix_ids=dummy_suffix_ids,
            compute_traj=bool(loss_args.use_traj_loss and loss_args.lambda_traj > 0),
            traj_layers=str(loss_args.traj_layers),
            traj_ref_tokens=int(loss_args.traj_ref_tokens),
            video_backend=str(data_args.video_backend),
            vision_feature_layer=int(vision_feature_layer),
            vision_feature_select_strategy=str(vision_feature_select_strategy),
            frame_chunk_size=int(data_args.vision_frame_chunk_size),
            disable_tqdm=disable_tqdm,
            compute_vision_kv=need_vision_kv,
            compute_llm_kv=need_llm_kv,
            return_partial=return_partial,
        )

    ref = _load_or_create_reference_stats(
        ref_path=ref_path,
        model=model,
        training_args=training_args,
        loss_args=loss_args,
        need_llm_kv=need_llm_kv,
        need_vision_kv=need_vision_kv,
        compute_stats=compute_ref,
    )

    ref_token_mean = ref["token_mean"]
    ref_token_var = ref["token_var"]
    ref_k = ref.get("ref_k")
    ref_v = ref.get("ref_v")
    vision_ref_k = ref.get("vision_ref_k")
    vision_ref_v = ref.get("vision_ref_v")

    model = apply_vision_lora(model, lora_args)

    train_roots = _split_csv(data_args.train_video_roots) or ref_roots
    train_maps = _split_csv(data_args.train_video_mappings) or ref_maps
    if not train_roots or not train_maps:
        raise ValueError("--train_video_roots and --train_video_mappings are required (or omit them to reuse reference_*).")

    train_video_paths = _iter_video_paths_from_mappings(train_maps, train_roots, max_items=data_args.train_max_videos)
    random.shuffle(train_video_paths)
    if data_args.train_max_videos is not None:
        train_video_paths = train_video_paths[: int(data_args.train_max_videos)]

    fast_num_frames = int(data_args.fast_num_frames) if data_args.fast_num_frames is not None else int(data_args.fast_frame_multiplier) * int(data_args.reference_num_frames)
    train_dataset = FastVideoFromVideoDataset(
        train_video_paths,
        video_processor=video_processor,
        video_backend=str(data_args.video_backend),
        fast_num_frames=fast_num_frames,
        fast_tile_size=int(data_args.fast_tile_size),
    )

    eval_dataset: Optional[Dataset] = train_dataset if training_args.eval_strategy != IntervalStrategy.NO else None

    trainer = _make_trainer(
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=LynXFastVideoCollator(),
        tokenizer=tokenizer,
        processor=processor,
        avqa_args=no_eval_args,
        loss_args=loss_args,
        audio_target_sr=16000,
        audio_seconds=8.0,
        mel_n_mels=128,
        mel_n_fft=1024,
        mel_hop_length=256,
        mel_win_length=1024,
        ref_token_mean=ref_token_mean,
        ref_token_var=ref_token_var,
        ref_k=ref_k,
        ref_v=ref_v,
        vision_ref_k=vision_ref_k,
        vision_ref_v=vision_ref_v,
        dummy_prefix_ids=dummy_prefix_ids,
        dummy_suffix_ids=dummy_suffix_ids,
        audio_repeat_frames=None,
        vision_feature_layer=int(vision_feature_layer),
        vision_feature_select_strategy=str(vision_feature_select_strategy),
    )

    trainer.train(resume_from_checkpoint=_parse_resume_from_checkpoint(training_args))
    trainer.save_model(training_args.output_dir)

    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()


def _main_3d(argv: Optional[Sequence[str]] = None) -> None:
    parser = HfArgumentParser((LynXModelArguments, LynX3DDataArguments, LynXLossArguments, LynXNoEvalArguments, VisionLoraArguments, TrainingArguments))
    if argv is None:
        model_args, data_args, loss_args, no_eval_args, lora_args, training_args = parser.parse_args_into_dataclasses()
    else:
        model_args, data_args, loss_args, no_eval_args, lora_args, training_args = parser.parse_args_into_dataclasses(args=list(argv))

    _maybe_disable_hf_offline(model_args)

    if training_args.deepspeed:
        maybe_patch_torch_autograd_graph_for_deepspeed()

    if training_args.output_dir is None:
        raise ValueError("--output_dir is required")
    os.makedirs(training_args.output_dir, exist_ok=True)
    training_args.remove_unused_columns = False

    set_seed(training_args.seed)

    model, processor, tokenizer, video_processor, device, vision_feature_layer, vision_feature_select_strategy = _load_model_and_processors(
        model_args=model_args,
        training_args=training_args,
    )
    dummy_prefix_ids, dummy_suffix_ids = _build_dummy_prefix_suffix_ids(tokenizer, loss_args)
    need_llm_kv, need_vision_kv = _resolve_distill_needs(loss_args, default_scope="all")

    ref_path = data_args.reference_stats_path or os.path.join(training_args.output_dir, "reference_stats.pt")

    def compute_ref(return_partial: bool, disable_tqdm: bool) -> Dict[str, Any]:
        ref_scene_dirs = iter_scene_dirs(data_args.reference_frames_root, max_items=data_args.reference_max_videos)
        rank, world_size = _infer_rank_and_world_size(training_args)
        shard_scene_dirs = ref_scene_dirs[rank::world_size] if world_size > 1 else ref_scene_dirs

        return _precompute_reference_stats_from_scene_dirs(
            model=model,
            video_processor=video_processor,
            scene_dirs=shard_scene_dirs,
            color_subdir=str(data_args.color_subdir),
            num_frames=int(data_args.reference_num_frames),
            dummy_prefix_ids=dummy_prefix_ids,
            dummy_suffix_ids=dummy_suffix_ids,
            vision_feature_layer=int(vision_feature_layer),
            vision_feature_select_strategy=str(vision_feature_select_strategy),
            frame_chunk_size=int(data_args.vision_frame_chunk_size),
            disable_tqdm=disable_tqdm,
            compute_vision_kv=need_vision_kv,
            compute_llm_kv=need_llm_kv,
            return_partial=return_partial,
            use_all_frames=bool(data_args.reference_use_all_frames),
            chunk_stride=data_args.reference_chunk_stride,
        )

    ref = _load_or_create_reference_stats(
        ref_path=ref_path,
        model=model,
        training_args=training_args,
        loss_args=loss_args,
        need_llm_kv=need_llm_kv,
        need_vision_kv=need_vision_kv,
        compute_stats=compute_ref,
    )

    ref_token_mean = ref["token_mean"]
    ref_token_var = ref["token_var"]
    ref_k = ref.get("ref_k")
    ref_v = ref.get("ref_v")
    vision_ref_k = ref.get("vision_ref_k")
    vision_ref_v = ref.get("vision_ref_v")

    model = apply_vision_lora(model, lora_args)

    train_scene_dirs: List[str]
    if data_args.train_annotation_file and os.path.exists(data_args.train_annotation_file):
        with open(data_args.train_annotation_file, "r") as f:
            ann = json.load(f)
        if not isinstance(ann, list):
            raise TypeError(f"Expected a list in {data_args.train_annotation_file}, got {type(ann)}")

        train_ids: List[str] = []
        seen_ids: set[str] = set()
        for item in ann:
            if not isinstance(item, dict):
                continue
            vid = item.get("video_name") or item.get("video") or item.get("video_id") or item.get("scene_id") or item.get("scene")
            if not isinstance(vid, str) or not vid:
                continue
            if vid in seen_ids:
                continue
            seen_ids.add(vid)
            train_ids.append(vid)

        train_scene_dirs = []
        for vid in train_ids:
            scene_dir = os.path.join(data_args.train_frames_root, vid)
            if os.path.isdir(scene_dir):
                train_scene_dirs.append(scene_dir)
    else:
        train_scene_dirs = iter_scene_dirs(data_args.train_frames_root, max_items=data_args.train_max_videos)

    random.shuffle(train_scene_dirs)
    if data_args.train_max_videos is not None:
        train_scene_dirs = train_scene_dirs[: int(data_args.train_max_videos)]

    train_dataset = DepthFromFramesDataset(
        train_scene_dirs,
        video_processor=video_processor,
        depth_subdir=data_args.depth_subdir,
        depth_num_frames=data_args.depth_num_frames,
        depth_clip_min_mm=data_args.depth_clip_min_mm,
        depth_clip_max_mm=data_args.depth_clip_max_mm,
        depth_encoding=data_args.depth_encoding,
        depth_intrinsics_filename=data_args.depth_intrinsics_filename,
        depth_auto_scale_intrinsics=data_args.depth_auto_scale_intrinsics,
        use_all_frames=data_args.depth_use_all_frames,
        chunk_stride=data_args.depth_chunk_stride,
    )

    if training_args.local_rank in (-1, 0):
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        per_device = int(training_args.per_device_train_batch_size)
        grad_acc = int(training_args.gradient_accumulation_steps)
        global_batch = per_device * max(1, world_size) * max(1, grad_acc)
        steps_floor = len(train_dataset) // global_batch if global_batch > 0 else 0
        steps_ceil = (len(train_dataset) + global_batch - 1) // global_batch if global_batch > 0 else 0
        print(
            f"[LynX-3D] train samples={len(train_dataset)} world_size={world_size} "
            f"per_device_batch={per_device} grad_acc={grad_acc} global_batch={global_batch} "
            f"estimated_opt_steps_per_epoch={steps_floor} (ceil={steps_ceil})"
        )

    eval_dataset: Optional[Dataset] = train_dataset if training_args.eval_strategy != IntervalStrategy.NO else None

    trainer = _make_trainer(
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=LynX3DDataCollator(),
        tokenizer=tokenizer,
        processor=processor,
        avqa_args=no_eval_args,
        loss_args=loss_args,
        audio_target_sr=16000,
        audio_seconds=8.0,
        mel_n_mels=128,
        mel_n_fft=1024,
        mel_hop_length=256,
        mel_win_length=1024,
        ref_token_mean=ref_token_mean,
        ref_token_var=ref_token_var,
        ref_k=ref_k,
        ref_v=ref_v,
        vision_ref_k=vision_ref_k,
        vision_ref_v=vision_ref_v,
        dummy_prefix_ids=dummy_prefix_ids,
        dummy_suffix_ids=dummy_suffix_ids,
        audio_repeat_frames=data_args.depth_repeat_frames,
        vision_feature_layer=int(vision_feature_layer),
        vision_feature_select_strategy=str(vision_feature_select_strategy),
    )

    trainer.train(resume_from_checkpoint=_parse_resume_from_checkpoint(training_args))
    trainer.save_model(training_args.output_dir)

    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()


def _normalize_task_name(task: str) -> Optional[str]:
    normalized = str(task).lower().replace("-", "").replace("_", "")
    if normalized in ("audio", "avqa", "a"):
        return "audio"
    if normalized in ("3d", "depth", "scanqa", "sqa"):
        return "3d"
    if normalized in ("egoexo",):
        return "egoexo"
    if normalized in ("fastvideo", "fast", "llavavideo", "video"):
        return "fastvideo"
    return None


def _split_task_from_argv(argv: Sequence[str]) -> Tuple[str, List[str]]:
    args = list(argv)
    task: Optional[str] = None

    for idx, arg in enumerate(args):
        if arg == "--task":
            if idx + 1 >= len(args):
                raise ValueError("--task requires a value (audio|3d|egoexo|fastvideo).")
            task = args[idx + 1]
            del args[idx : idx + 2]
            break
        if arg.startswith("--task="):
            task = arg.split("=", 1)[1]
            del args[idx]
            break

    if task is not None:
        normalized = _normalize_task_name(task)
        if normalized is None:
            raise ValueError(f"Unknown --task={task!r}. Expected one of: audio, 3d, egoexo, fastvideo.")
        return normalized, args

    if args and not args[0].startswith("-"):
        normalized = _normalize_task_name(args[0])
        if normalized is not None:
            args.pop(0)
            return normalized, args

    return "audio", args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    task, task_argv = _split_task_from_argv(args)
    if task == "audio":
        _main_audio(task_argv)
        return
    if task == "egoexo":
        _main_egoexo(task_argv)
        return
    if task == "fastvideo":
        _main_fastvideo(task_argv)
        return
    if task == "3d":
        _main_3d(task_argv)
        return
    raise RuntimeError(f"Unhandled task {task!r}")


if __name__ == "__main__":
    main()
