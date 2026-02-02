import os
import pathlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

from libs.utils.lynx_3d import encode_depth_to_rgb_frames, list_image_paths, sample_uniform
from lynx_utils import (
    load_video_frames_and_audio,
    log_mel_to_pil_rgb,
    resample_waveform,
    tile_pixel_values_videos_spatially,
    to_mono,
    waveform_to_log_mel,
)


def chunk_starts(total: int, *, chunk_size: int, stride: int) -> List[int]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if stride <= 0:
        raise ValueError("stride must be > 0")
    if total <= 0:
        return [0]
    return list(range(0, total, stride))


def pad_to_length(paths: Sequence[str], *, length: int) -> List[str]:
    if length <= 0:
        raise ValueError("length must be > 0")
    out = list(paths)
    if not out:
        return out
    if len(out) < length:
        out.extend([out[-1]] * (length - len(out)))
    return out


def iter_scene_dirs(frames_root: str, *, max_items: Optional[int]) -> List[str]:
    root = pathlib.Path(frames_root)
    if not root.exists():
        return []
    dirs = [str(p) for p in root.iterdir() if p.is_dir()]
    dirs.sort()
    if max_items is not None:
        dirs = dirs[: int(max_items)]
    return dirs


class AudioFromVideoDataset(Dataset):
    def __init__(
        self,
        video_paths: Sequence[str],
        *,
        video_processor: Any,
        video_backend: str,
        audio_target_sr: int,
        audio_seconds: float,
        mel_n_mels: int,
        mel_n_fft: int,
        mel_hop_length: int,
        mel_win_length: int,
    ) -> None:
        self.video_paths = list(video_paths)
        self.video_processor = video_processor
        self.video_backend = str(video_backend)
        self.audio_target_sr = int(audio_target_sr)
        self.audio_seconds = float(audio_seconds)
        self.mel_n_mels = int(mel_n_mels)
        self.mel_n_fft = int(mel_n_fft)
        self.mel_hop_length = int(mel_hop_length)
        self.mel_win_length = int(mel_win_length)

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        video_path = self.video_paths[idx]
        try:
            _, aframes, info = load_video_frames_and_audio(video_path, num_frames=0, video_backend=self.video_backend)
        except Exception:
            aframes = torch.empty((0,), dtype=torch.float32)
            info = {"audio_fps": self.audio_target_sr}

        if aframes.numel() == 0:
            waveform = torch.zeros(int(self.audio_target_sr * self.audio_seconds), dtype=torch.float32)
            orig_sr = self.audio_target_sr
        else:
            aframes = aframes.detach().cpu()
            if aframes.ndim == 2 and aframes.shape[0] <= 8 and aframes.shape[1] > aframes.shape[0] * 100:
                aframes = aframes.transpose(0, 1)
            waveform = to_mono(aframes).float()
            orig_sr = int(info.get("audio_fps") or info.get("audio_sample_rate") or self.audio_target_sr)

        waveform = resample_waveform(waveform, orig_sr=orig_sr, target_sr=self.audio_target_sr)
        log_mel = waveform_to_log_mel(
            waveform,
            sample_rate=self.audio_target_sr,
            target_seconds=self.audio_seconds,
            n_fft=self.mel_n_fft,
            hop_length=self.mel_hop_length,
            win_length=self.mel_win_length,
            n_mels=self.mel_n_mels,
        )
        audio_img = log_mel_to_pil_rgb(log_mel)

        pv = self.video_processor([[audio_img]], return_tensors="pt")["pixel_values_videos"][0]
        return {"audio_pixel_values_videos": pv, "video_path": video_path}


class LynXDataCollator:
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {"audio_pixel_values_videos": torch.stack([f["audio_pixel_values_videos"] for f in features], dim=0)}


class VideoFromVideoDataset(Dataset):
    def __init__(
        self,
        video_paths: Sequence[str],
        *,
        video_processor: Any,
        video_backend: str,
        num_frames: int,
    ) -> None:
        self.video_paths = list(video_paths)
        self.video_processor = video_processor
        self.video_backend = str(video_backend)
        self.num_frames = int(num_frames)

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        video_path = self.video_paths[idx]
        vframes, _, _ = load_video_frames_and_audio(
            video_path,
            num_frames=self.num_frames,
            video_backend=self.video_backend,
            decode_audio=False,
        )
        pv = self.video_processor([vframes], return_tensors="pt")["pixel_values_videos"][0]
        return {"fast_pixel_values_videos": pv}


class VideoFromVideoCollator:
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {"fast_pixel_values_videos": torch.stack([f["fast_pixel_values_videos"] for f in features], dim=0)}


class FastVideoFromVideoDataset(Dataset):
    def __init__(
        self,
        video_paths: Sequence[str],
        *,
        video_processor: Any,
        video_backend: str,
        fast_num_frames: int,
        fast_tile_size: int,
    ) -> None:
        self.video_paths = list(video_paths)
        self.video_processor = video_processor
        self.video_backend = str(video_backend)
        self.fast_num_frames = int(fast_num_frames)
        self.fast_tile_size = int(fast_tile_size)

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        video_path = self.video_paths[idx]
        vframes, _, _ = load_video_frames_and_audio(
            video_path,
            num_frames=self.fast_num_frames,
            video_backend=self.video_backend,
            decode_audio=False,
        )
        pv = self.video_processor([vframes], return_tensors="pt")["pixel_values_videos"][0]  # (F, C, H, W)
        pv = tile_pixel_values_videos_spatially(pv, tile_size=self.fast_tile_size)  # (F', C, H, W)
        return {"fast_pixel_values_videos": pv}


class LynXFastVideoCollator:
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {"fast_pixel_values_videos": torch.stack([f["fast_pixel_values_videos"] for f in features], dim=0)}


class DepthFromFramesDataset(Dataset):
    def __init__(
        self,
        scene_dirs: Sequence[str],
        *,
        video_processor: Any,
        depth_subdir: str,
        depth_num_frames: int,
        depth_clip_min_mm: float,
        depth_clip_max_mm: float,
        depth_encoding: str,
        depth_intrinsics_filename: str,
        depth_auto_scale_intrinsics: bool,
        use_all_frames: bool,
        chunk_stride: Optional[int],
    ) -> None:
        self.scene_dirs = list(scene_dirs)
        self.video_processor = video_processor
        self.depth_subdir = str(depth_subdir)
        self.depth_num_frames = int(depth_num_frames)
        self.depth_clip_min_mm = float(depth_clip_min_mm)
        self.depth_clip_max_mm = float(depth_clip_max_mm)
        self.depth_encoding = str(depth_encoding)
        self.depth_intrinsics_filename = str(depth_intrinsics_filename)
        self.depth_auto_scale_intrinsics = bool(depth_auto_scale_intrinsics)

        self.use_all_frames = bool(use_all_frames)
        self.chunk_stride = int(chunk_stride) if chunk_stride is not None else self.depth_num_frames
        if self.chunk_stride <= 0:
            raise ValueError("depth_chunk_stride must be > 0")

        self._depth_paths: List[List[str]] = []
        self._samples: List[Tuple[int, int]] = []
        for scene_idx, scene_dir in enumerate(self.scene_dirs):
            depth_dir = os.path.join(scene_dir, self.depth_subdir)
            frame_paths = list_image_paths(depth_dir, exts=(".png", ".jpg", ".jpeg"))
            self._depth_paths.append(frame_paths)

            if not self.use_all_frames:
                self._samples.append((scene_idx, -1))
                continue

            if not frame_paths:
                self._samples.append((scene_idx, 0))
                continue

            for start in chunk_starts(len(frame_paths), chunk_size=self.depth_num_frames, stride=self.chunk_stride):
                self._samples.append((scene_idx, start))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        scene_idx, start = self._samples[idx]
        scene_dir = self.scene_dirs[scene_idx]

        frame_paths = self._depth_paths[scene_idx]
        selected: List[str]
        if not frame_paths:
            selected = []
        elif self.use_all_frames and start >= 0:
            selected = pad_to_length(frame_paths[start : start + self.depth_num_frames], length=self.depth_num_frames)
        else:
            selected = sample_uniform(frame_paths, self.depth_num_frames)

        frames: List[torch.Tensor] = []
        for path in selected:
            frames.extend(
                encode_depth_to_rgb_frames(
                    path,
                    scene_dir=scene_dir,
                    encoding=self.depth_encoding,
                    clip_min_mm=self.depth_clip_min_mm,
                    clip_max_mm=self.depth_clip_max_mm,
                    intrinsics_filename=self.depth_intrinsics_filename,
                    auto_scale_intrinsics=self.depth_auto_scale_intrinsics,
                    pose_c2w=None,
                    normals_frame="camera",
                )
            )

        if not frames:
            frames = [torch.zeros((3, 240, 320), dtype=torch.uint8)]

        pv = self.video_processor([torch.stack(frames, dim=0)], return_tensors="pt")["pixel_values_videos"][0]
        return {"depth_pixel_values_videos": pv, "scene_dir": scene_dir}


class LynX3DDataCollator:
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {"depth_pixel_values_videos": torch.stack([f["depth_pixel_values_videos"] for f in features], dim=0)}

