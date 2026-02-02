import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from libs.model.lynx_onevision import IGNORE_INDEX
from libs.utils.lynx_3d import (
    encode_depth_to_rgb_frames,
    list_image_paths,
    numeric_stem,
    pil_rgb_to_chw_uint8,
    read_pose_c2w,
    sample_pose_aware_pairs,
    sample_uniform,
)
from lynx_utils import (
    load_json,
    load_video_frames_and_audio,
    log_mel_to_pil_rgb,
    resample_waveform,
    tile_pixel_values_videos_spatially,
    to_mono,
    waveform_to_imagebind_melspec_clips,
)


def _coerce_audio_waveform_and_sr(
    aframes: torch.Tensor,
    info: Dict[str, Any],
    *,
    fallback_sr: int,
    target_seconds: float,
) -> Tuple[torch.Tensor, int]:
    if aframes.numel() == 0:
        waveform = torch.zeros(int(fallback_sr * target_seconds), dtype=torch.float32)
        return waveform, fallback_sr

    aframes = aframes.detach().cpu()
    if aframes.ndim == 2 and aframes.shape[0] <= 8 and aframes.shape[1] > aframes.shape[0] * 100:
        aframes = aframes.transpose(0, 1)
    waveform = to_mono(aframes).float()
    orig_sr = int(info.get("audio_fps") or info.get("audio_sample_rate") or fallback_sr)
    return waveform, orig_sr


def _resolve_video_path(video_id: str, *, video_root: str, mapping: Optional[Dict[str, str]]) -> str:
    vid = str(video_id)
    if mapping:
        if vid in mapping:
            return os.path.join(video_root, str(mapping[vid]))
        if vid.endswith(".mp4") and vid[:-4] in mapping:
            return os.path.join(video_root, str(mapping[vid[:-4]]))
    if vid.endswith(".mp4"):
        return os.path.join(video_root, vid)
    return os.path.join(video_root, f"{vid}.mp4")


def _conv_to_messages(conversations: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        who = turn.get("from")
        value = turn.get("value")
        if not isinstance(value, str) or not value:
            continue
        if who in ("human", "user"):
            messages.append({"role": "user", "content": value})
        elif who in ("gpt", "assistant"):
            messages.append({"role": "assistant", "content": value})
    return messages


def _build_multiturn_labels(tokenizer: Any, messages: List[Dict[str, str]], full_ids: List[int]) -> torch.Tensor:
    labels = torch.full((len(full_ids),), IGNORE_INDEX, dtype=torch.long)
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        prompt_ids: List[int] = tokenizer.apply_chat_template(messages[:i], tokenize=True, add_generation_prompt=True)
        prefix_ids: List[int] = tokenizer.apply_chat_template(messages[: i + 1], tokenize=True, add_generation_prompt=False)
        if not isinstance(prompt_ids, list) or not isinstance(prefix_ids, list):
            raise TypeError("Tokenizer chat template did not return token ids as a list.")
        start = len(prompt_ids)
        end = len(prefix_ids)
        if end <= start:
            continue
        labels[start:end] = torch.tensor(full_ids[start:end], dtype=torch.long)
    return labels


def _pad_instruction_batch(tokenizer: Any, features: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    pad_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)

    input_ids_list = [f["input_ids"] for f in features]
    labels_list = [f["labels"] for f in features]
    max_len = max(int(x.shape[0]) for x in input_ids_list) if input_ids_list else 0

    batch_size = len(features)
    input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    labels = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    for i, (ids, lab) in enumerate(zip(input_ids_list, labels_list)):
        seq_len = int(ids.shape[0])
        input_ids[i, :seq_len] = ids
        labels[i, :seq_len] = lab
        attention_mask[i, :seq_len] = 1

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


class AVQAInstructionDataset(Dataset):
    def __init__(
        self,
        annotation_file: str,
        *,
        video_root: str,
        video_mapping: Optional[str],
        tokenizer: Any,
        video_processor: Any,
        num_frames: int,
        audio_target_sr: int,
        audio_seconds: float,
        audio_clip_duration_s: float,
        audio_clip_stride_s: float,
        mel_bins: int,
        mel_target_length: int,
        mel_mean: float,
        mel_std: float,
        max_samples: Optional[int],
    ) -> None:
        self._tokenizer = tokenizer
        self._video_processor = video_processor

        ann = load_json(annotation_file)
        if not isinstance(ann, list):
            raise TypeError(f"Expected a list in {annotation_file}, got {type(ann)}")
        if max_samples is not None:
            ann = ann[: int(max_samples)]

        mapping: Optional[Dict[str, str]] = None
        if video_mapping:
            raw = load_json(video_mapping)
            if isinstance(raw, dict):
                mapping = raw

        examples: List[Dict[str, Any]] = []
        for item in ann:
            if not isinstance(item, dict):
                continue
            vid = item.get("video") or item.get("video_name")
            if not isinstance(vid, str) or not vid:
                continue
            filename = mapping.get(vid, f"{vid}.mp4") if isinstance(mapping, dict) else f"{vid}.mp4"
            video_path = os.path.join(video_root, filename)
            if not os.path.exists(video_path):
                continue

            conv = item.get("conversations")
            if not isinstance(conv, list) or len(conv) < 2:
                continue
            user_text = None
            assistant_text = None
            for turn in conv:
                if not isinstance(turn, dict):
                    continue
                role = turn.get("from")
                value = turn.get("value")
                if not isinstance(value, str):
                    continue
                if role in ("human", "user") and user_text is None:
                    user_text = value
                elif role in ("gpt", "assistant") and assistant_text is None:
                    assistant_text = value
            if user_text is None or assistant_text is None:
                continue

            examples.append(
                {
                    "id": item.get("id"),
                    "video_path": video_path,
                    "user_text": user_text,
                    "assistant_text": assistant_text,
                }
            )

        self._examples = examples
        self._num_frames = int(num_frames)
        self._audio_target_sr = int(audio_target_sr)
        self._audio_seconds = float(audio_seconds)
        self._audio_clip_duration_s = float(audio_clip_duration_s)
        self._audio_clip_stride_s = float(audio_clip_stride_s)
        self._mel_bins = int(mel_bins)
        self._mel_target_length = int(mel_target_length)
        self._mel_mean = float(mel_mean)
        self._mel_std = float(mel_std)

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self._examples[idx]
        video_path = str(ex["video_path"])

        user_text = str(ex["user_text"])
        assistant_text = str(ex["assistant_text"])

        messages_user = [{"role": "user", "content": user_text}]
        messages_full = [{"role": "user", "content": user_text}, {"role": "assistant", "content": assistant_text}]

        prompt_ids: List[int] = self._tokenizer.apply_chat_template(messages_user, tokenize=True, add_generation_prompt=True)
        full_ids: List[int] = self._tokenizer.apply_chat_template(messages_full, tokenize=True, add_generation_prompt=False)
        if not isinstance(prompt_ids, list) or not isinstance(full_ids, list):
            raise TypeError("Tokenizer chat template did not return token ids as a list.")
        prompt_len = len(prompt_ids)

        input_ids = torch.tensor(full_ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[:prompt_len] = IGNORE_INDEX

        vframes, aframes, info = load_video_frames_and_audio(
            video_path,
            num_frames=self._num_frames,
            video_backend="torchvision",
            decode_audio=True,
        )

        waveform, orig_sr = _coerce_audio_waveform_and_sr(
            aframes,
            info,
            fallback_sr=self._audio_target_sr,
            target_seconds=self._audio_seconds,
        )
        waveform = resample_waveform(waveform, orig_sr=orig_sr, target_sr=self._audio_target_sr)

        audio_clips = waveform_to_imagebind_melspec_clips(
            waveform,
            sample_rate=self._audio_target_sr,
            num_clips=int(vframes.shape[0]),
            clip_duration_s=self._audio_clip_duration_s,
            clip_stride_s=self._audio_clip_stride_s,
            num_mel_bins=self._mel_bins,
            target_length=self._mel_target_length,
            mean=self._mel_mean,
            std=self._mel_std,
        )  # (F, 1, mel_bins, mel_target_length)
        audio_frames = [log_mel_to_pil_rgb(audio_clips[i, 0]) for i in range(audio_clips.shape[0])]

        pv_video = self._video_processor([vframes], return_tensors="pt")["pixel_values_videos"][0]
        pv_audio = self._video_processor([audio_frames], return_tensors="pt")["pixel_values_videos"][0]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values_videos": pv_video,
            "audio_pixel_values_videos": pv_audio,
        }


class AVQAInstructionCollator:
    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = _pad_instruction_batch(self._tokenizer, features)
        batch["pixel_values_videos"] = torch.stack([f["pixel_values_videos"] for f in features], dim=0)
        batch["audio_pixel_values_videos"] = torch.stack([f["audio_pixel_values_videos"] for f in features], dim=0)
        return batch


class AVSDInstructionDataset(Dataset):
    def __init__(
        self,
        annotation_file: str,
        *,
        video_root: str,
        video_mapping: Optional[str],
        tokenizer: Any,
        video_processor: Any,
        num_frames: int,
        audio_target_sr: int,
        audio_seconds: float,
        audio_clip_duration_s: float,
        audio_clip_stride_s: float,
        mel_bins: int,
        mel_target_length: int,
        mel_mean: float,
        mel_std: float,
        max_samples: Optional[int],
    ) -> None:
        self._tokenizer = tokenizer
        self._video_processor = video_processor

        ann = load_json(annotation_file)
        if not isinstance(ann, list):
            raise TypeError(f"Expected a list in {annotation_file}, got {type(ann)}")
        if max_samples is not None:
            ann = ann[: int(max_samples)]

        mapping: Optional[Dict[str, str]] = None
        if video_mapping:
            raw = load_json(video_mapping)
            if isinstance(raw, dict):
                mapping = {str(k): str(v) for k, v in raw.items()}

        examples: List[Dict[str, Any]] = []
        for item in ann:
            if not isinstance(item, dict):
                continue
            vid = item.get("video_name") or item.get("video") or item.get("video_id") or item.get("image_id")
            if not isinstance(vid, str) or not vid:
                continue

            video_path = _resolve_video_path(vid, video_root=video_root, mapping=mapping)
            if not os.path.exists(video_path):
                continue

            conv = item.get("conversations")
            if not isinstance(conv, list) or len(conv) < 2:
                continue
            messages = _conv_to_messages(conv)
            if not any(m.get("role") == "assistant" for m in messages):
                continue

            examples.append({"id": item.get("id"), "video_path": video_path, "messages": messages})

        self._examples = examples
        self._num_frames = int(num_frames)
        self._audio_target_sr = int(audio_target_sr)
        self._audio_seconds = float(audio_seconds)
        self._audio_clip_duration_s = float(audio_clip_duration_s)
        self._audio_clip_stride_s = float(audio_clip_stride_s)
        self._mel_bins = int(mel_bins)
        self._mel_target_length = int(mel_target_length)
        self._mel_mean = float(mel_mean)
        self._mel_std = float(mel_std)

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self._examples[idx]
        video_path = str(ex["video_path"])
        messages: List[Dict[str, str]] = list(ex["messages"])

        full_ids: List[int] = self._tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        if not isinstance(full_ids, list):
            raise TypeError("Tokenizer chat template did not return token ids as a list.")

        input_ids = torch.tensor(full_ids, dtype=torch.long)
        labels = _build_multiturn_labels(self._tokenizer, messages, full_ids)

        vframes, aframes, info = load_video_frames_and_audio(
            video_path,
            num_frames=self._num_frames,
            video_backend="torchvision",
            decode_audio=True,
        )

        waveform, orig_sr = _coerce_audio_waveform_and_sr(
            aframes,
            info,
            fallback_sr=self._audio_target_sr,
            target_seconds=self._audio_seconds,
        )
        waveform = resample_waveform(waveform, orig_sr=orig_sr, target_sr=self._audio_target_sr)

        audio_clips = waveform_to_imagebind_melspec_clips(
            waveform,
            sample_rate=self._audio_target_sr,
            num_clips=int(vframes.shape[0]),
            clip_duration_s=self._audio_clip_duration_s,
            clip_stride_s=self._audio_clip_stride_s,
            num_mel_bins=self._mel_bins,
            target_length=self._mel_target_length,
            mean=self._mel_mean,
            std=self._mel_std,
        )  # (F, 1, mel_bins, mel_target_length)
        audio_frames = [log_mel_to_pil_rgb(audio_clips[i, 0]) for i in range(audio_clips.shape[0])]

        pv_video = self._video_processor([vframes], return_tensors="pt")["pixel_values_videos"][0]
        pv_audio = self._video_processor([audio_frames], return_tensors="pt")["pixel_values_videos"][0]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values_videos": pv_video,
            "audio_pixel_values_videos": pv_audio,
        }


class EgoExoProficiencyInstructionDataset(Dataset):
    def __init__(
        self,
        annotation_file: str,
        *,
        video_root: str,
        take_id_to_video_mapping: str,
        tokenizer: Any,
        video_processor: Any,
        num_frames: int,
        max_samples: Optional[int],
        video_backend: str,
    ) -> None:
        with open(take_id_to_video_mapping, "r") as f:
            mapping = json.load(f)
        if not isinstance(mapping, dict):
            raise TypeError(f"Expected a dict in {take_id_to_video_mapping}, got {type(mapping)}")

        with open(annotation_file, "r") as f:
            ann = json.load(f)
        if not isinstance(ann, list):
            raise TypeError(f"Expected a list in {annotation_file}, got {type(ann)}")

        self._tokenizer = tokenizer
        self._video_processor = video_processor
        self._num_frames = int(num_frames)
        self._video_backend = str(video_backend)

        examples: List[Dict[str, Any]] = []
        for item in ann:
            if not isinstance(item, dict):
                continue
            video_id = item.get("video") or item.get("video_name") or item.get("video_id")
            if not isinstance(video_id, str) or not video_id:
                continue
            rel = mapping.get(video_id)
            if not isinstance(rel, str) or not rel:
                continue
            video_path = os.path.join(str(video_root), rel)
            if not os.path.exists(video_path):
                continue
            conv = item.get("conversations")
            messages = _conv_to_messages(conv) if isinstance(conv, list) else []
            if not messages or not any(m.get("role") == "assistant" for m in messages):
                continue
            examples.append({"id": item.get("id"), "video_path": video_path, "messages": messages})
            if max_samples is not None and len(examples) >= int(max_samples):
                break

        self._examples = examples

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self._examples[idx]
        video_path = str(ex["video_path"])
        messages: List[Dict[str, str]] = list(ex["messages"])

        full_ids: List[int] = self._tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        if not isinstance(full_ids, list):
            raise TypeError("Tokenizer chat template did not return token ids as a list.")

        input_ids = torch.tensor(full_ids, dtype=torch.long)
        labels = _build_multiturn_labels(self._tokenizer, messages, full_ids)

        frames, _, _ = load_video_frames_and_audio(
            video_path,
            num_frames=self._num_frames,
            video_backend=self._video_backend,
            decode_audio=False,
        )
        pv_video = self._video_processor([frames], return_tensors="pt")["pixel_values_videos"][0]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values_videos": pv_video,
            "fast_pixel_values_videos": pv_video,
        }


class EgoExoInstructionCollator:
    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = _pad_instruction_batch(self._tokenizer, features)
        batch["pixel_values_videos"] = torch.stack([f["pixel_values_videos"] for f in features], dim=0)
        batch["fast_pixel_values_videos"] = torch.stack([f["fast_pixel_values_videos"] for f in features], dim=0)
        return batch


class LlavaVideo178KFastInstructionDataset(Dataset):
    def __init__(
        self,
        annotation_files: Sequence[str],
        *,
        video_roots: Sequence[str],
        tokenizer: Any,
        video_processor: Any,
        slow_num_frames: int,
        fast_num_frames: int,
        fast_tile_size: int,
        max_samples: Optional[int],
        video_backend: str = "torchvision",
    ) -> None:
        if len(annotation_files) != len(video_roots):
            raise ValueError(
                f"Expected annotation_files and video_roots to have the same length, got {len(annotation_files)} vs {len(video_roots)}"
            )

        self._tokenizer = tokenizer
        self._video_processor = video_processor
        self._slow_num_frames = int(slow_num_frames)
        self._fast_num_frames = int(fast_num_frames)
        self._fast_tile_size = int(fast_tile_size)
        self._video_backend = str(video_backend)

        examples: List[Dict[str, Any]] = []
        for ann_path, root in zip(annotation_files, video_roots):
            with open(ann_path, "r") as f:
                ann = json.load(f)
            if not isinstance(ann, list):
                raise TypeError(f"Expected a list in {ann_path}, got {type(ann)}")

            for item in ann:
                if not isinstance(item, dict):
                    continue
                rel = item.get("video")
                if not isinstance(rel, str) or not rel:
                    continue
                video_path = os.path.join(str(root), rel)
                if not os.path.exists(video_path):
                    continue
                conv = item.get("conversations")
                messages = _conv_to_messages(conv) if isinstance(conv, list) else []
                if not messages or not any(m.get("role") == "assistant" for m in messages):
                    continue
                examples.append({"id": item.get("id"), "video_path": video_path, "messages": messages})
                if max_samples is not None and len(examples) >= int(max_samples):
                    break
            if max_samples is not None and len(examples) >= int(max_samples):
                break

        self._examples = examples

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self._examples[idx]
        video_path = str(ex["video_path"])
        messages: List[Dict[str, str]] = list(ex["messages"])

        full_ids: List[int] = self._tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        if not isinstance(full_ids, list):
            raise TypeError("Tokenizer chat template did not return token ids as a list.")

        input_ids = torch.tensor(full_ids, dtype=torch.long)
        labels = _build_multiturn_labels(self._tokenizer, messages, full_ids)

        slow_frames, _, _ = load_video_frames_and_audio(
            video_path,
            num_frames=self._slow_num_frames,
            video_backend=self._video_backend,
            decode_audio=False,
        )
        fast_frames, _, _ = load_video_frames_and_audio(
            video_path,
            num_frames=self._fast_num_frames,
            video_backend=self._video_backend,
            decode_audio=False,
        )

        pv_video = self._video_processor([slow_frames], return_tensors="pt")["pixel_values_videos"][0]
        pv_fast = self._video_processor([fast_frames], return_tensors="pt")["pixel_values_videos"][0]
        pv_fast = tile_pixel_values_videos_spatially(pv_fast, tile_size=self._fast_tile_size)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values_videos": pv_video,
            "fast_pixel_values_videos": pv_fast,
        }


class LlavaVideo178KFastInstructionCollator:
    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = _pad_instruction_batch(self._tokenizer, features)
        batch["pixel_values_videos"] = torch.stack([f["pixel_values_videos"] for f in features], dim=0)
        batch["fast_pixel_values_videos"] = torch.stack([f["fast_pixel_values_videos"] for f in features], dim=0)
        return batch


class ThreeDInstructionDataset(Dataset):
    def __init__(
        self,
        annotation_file: str,
        *,
        frames_root: str,
        color_subdir: str,
        depth_subdir: str,
        pose_subdir: str,
        pose_matrix_type: str,
        tokenizer: Any,
        video_processor: Any,
        frame_sampling: str,
        num_frames: int,
        depth_clip_min_mm: float,
        depth_clip_max_mm: float,
        depth_encoding: str,
        depth_normals_frame: str,
        depth_intrinsics_filename: str,
        depth_auto_scale_intrinsics: bool,
        max_samples: Optional[int],
    ) -> None:
        self._tokenizer = tokenizer
        self._video_processor = video_processor
        self._frames_root = str(frames_root)
        self._color_subdir = str(color_subdir)
        self._depth_subdir = str(depth_subdir)
        self._pose_subdir = str(pose_subdir)
        self._pose_matrix_type = str(pose_matrix_type)
        self._frame_sampling = str(frame_sampling or "uniform").lower().strip()
        self._num_frames = int(num_frames)

        self._depth_clip_min_mm = float(depth_clip_min_mm)
        self._depth_clip_max_mm = float(depth_clip_max_mm)
        self._depth_encoding = str(depth_encoding)
        self._depth_normals_frame = str(depth_normals_frame)
        self._depth_intrinsics_filename = str(depth_intrinsics_filename)
        self._depth_auto_scale_intrinsics = bool(depth_auto_scale_intrinsics)

        self._selected_pairs_cache: Dict[str, List[Tuple[str, str]]] = {}
        self._pose_cache: Dict[Tuple[str, int], Optional[np.ndarray]] = {}

        ann = load_json(annotation_file)
        if not isinstance(ann, list):
            raise TypeError(f"Expected a list in {annotation_file}, got {type(ann)}")
        if max_samples is not None:
            ann = ann[: int(max_samples)]

        examples: List[Dict[str, Any]] = []
        for item in ann:
            if not isinstance(item, dict):
                continue
            vid = item.get("video") or item.get("video_name") or item.get("video_id") or item.get("scene") or item.get("scene_id")
            if not isinstance(vid, str) or not vid:
                continue
            conv = item.get("conversations")
            if not isinstance(conv, list):
                continue
            messages = _conv_to_messages(conv)
            if not messages or not any(m.get("role") == "assistant" for m in messages):
                continue
            examples.append({"id": item.get("id"), "video": vid, "messages": messages})

        self._examples = examples

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self._examples[idx]
        vid = str(ex["video"])
        scene_dir = os.path.join(self._frames_root, vid)
        color_dir = os.path.join(scene_dir, self._color_subdir)
        depth_dir = os.path.join(scene_dir, self._depth_subdir)
        pairs = self._selected_pairs_cache.get(vid)
        if pairs is None:
            color_paths = list_image_paths(color_dir, exts=(".jpg", ".jpeg", ".png"))
            depth_paths = list_image_paths(depth_dir, exts=(".png", ".jpg", ".jpeg"))
            depth_by_key = {numeric_stem(p): p for p in depth_paths}
            pairs = []
            for cpath in color_paths:
                dpath = depth_by_key.get(numeric_stem(cpath))
                if dpath:
                    pairs.append((cpath, dpath))

            if self._frame_sampling == "pose":
                selected = sample_pose_aware_pairs(
                    pairs,
                    scene_dir=scene_dir,
                    num_frames=self._num_frames,
                    pose_subdir=self._pose_subdir,
                    pose_matrix_type=self._pose_matrix_type,
                )
            else:
                selected = sample_uniform(pairs, self._num_frames)
            self._selected_pairs_cache[vid] = selected
        else:
            selected = pairs
        if not selected:
            raise FileNotFoundError(f"No paired RGB/depth frames found for video={vid}")

        messages: List[Dict[str, str]] = list(ex["messages"])
        full_ids: List[int] = self._tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        if not isinstance(full_ids, list):
            raise TypeError("Tokenizer chat template did not return token ids as a list.")

        input_ids = torch.tensor(full_ids, dtype=torch.long)
        labels = _build_multiturn_labels(self._tokenizer, messages, full_ids)

        rgb_frames: List[torch.Tensor] = []
        depth_frames: List[torch.Tensor] = []
        depth_encoding = str(self._depth_encoding or "turbo").lower().strip()
        need_pose_for_normals = self._depth_normals_frame.lower().strip() == "world" and "normals" in depth_encoding
        for color_path, depth_path in selected:
            key = numeric_stem(color_path)
            pose_c2w = None
            if need_pose_for_normals:
                cache_key = (vid, int(key))
                if cache_key in self._pose_cache:
                    pose_c2w = self._pose_cache[cache_key]
                else:
                    pose_c2w = read_pose_c2w(
                        scene_dir,
                        frame_key=key,
                        pose_subdir=self._pose_subdir,
                        pose_matrix_type=self._pose_matrix_type,
                    )
                    self._pose_cache[cache_key] = pose_c2w
            rgb_frames.append(pil_rgb_to_chw_uint8(color_path))
            depth_frames.extend(
                encode_depth_to_rgb_frames(
                    depth_path,
                    scene_dir=scene_dir,
                    encoding=depth_encoding,
                    clip_min_mm=self._depth_clip_min_mm,
                    clip_max_mm=self._depth_clip_max_mm,
                    intrinsics_filename=self._depth_intrinsics_filename,
                    auto_scale_intrinsics=self._depth_auto_scale_intrinsics,
                    pose_c2w=pose_c2w,
                    normals_frame=self._depth_normals_frame,
                )
            )

        rgb_tensor = torch.stack(rgb_frames, dim=0)
        depth_tensor = torch.stack(depth_frames, dim=0)

        pv_rgb = self._video_processor([rgb_tensor], return_tensors="pt")["pixel_values_videos"][0]
        pv_depth = self._video_processor([depth_tensor], return_tensors="pt")["pixel_values_videos"][0]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values_videos": pv_rgb,
            "depth_pixel_values_videos": pv_depth,
        }


class ThreeDInstructionCollator:
    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = _pad_instruction_batch(self._tokenizer, features)
        batch["pixel_values_videos"] = torch.stack([f["pixel_values_videos"] for f in features], dim=0)
        batch["depth_pixel_values_videos"] = torch.stack([f["depth_pixel_values_videos"] for f in features], dim=0)
        return batch


__all__ = [
    "AVQAInstructionCollator",
    "AVQAInstructionDataset",
    "AVSDInstructionDataset",
    "EgoExoInstructionCollator",
    "EgoExoProficiencyInstructionDataset",
    "LlavaVideo178KFastInstructionCollator",
    "LlavaVideo178KFastInstructionDataset",
    "ThreeDInstructionCollator",
    "ThreeDInstructionDataset",
]

