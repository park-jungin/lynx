import json
import math
import os
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _is_flash_attn_2_available() -> bool:
    try:
        from transformers.utils import is_flash_attn_2_available as _hf_check  # type: ignore
    except Exception:
        try:
            from transformers.utils.import_utils import is_flash_attn_2_available as _hf_check  # type: ignore
        except Exception:
            _hf_check = None

    if callable(_hf_check):
        try:
            return bool(_hf_check())
        except Exception:
            return False

    try:
        import flash_attn  # noqa: F401

        return True
    except Exception:
        return False


def resolve_attn_implementation(use_flash_attn: bool, *, device: Optional[torch.device | str] = None) -> str:
    if device is not None:
        device = torch.device(device)
        if device.type != "cuda":
            use_flash_attn = False

    if use_flash_attn and _is_flash_attn_2_available():
        return "flash_attention_2"

    return "sdpa" if hasattr(torch.nn.functional, "scaled_dot_product_attention") else "eager"


def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def iter_video_paths_from_mapping(
    mapping_path: str,
    video_root: str,
    *,
    max_items: Optional[int] = None,
) -> List[str]:
    mapping = load_json(mapping_path)
    if not isinstance(mapping, dict):
        raise TypeError(f"Expected a dict in {mapping_path}, got {type(mapping)}")

    video_paths: List[str] = []
    for _, rel_path in mapping.items():
        video_path = os.path.join(video_root, rel_path)
        video_paths.append(video_path)
        if max_items is not None and len(video_paths) >= max_items:
            break
    return video_paths


def iter_video_paths_from_dir(
    video_root: str,
    *,
    exts: Sequence[str] = (".mp4", ".avi", ".mov", ".mkv", ".webm"),
    max_items: Optional[int] = None,
) -> List[str]:
    video_paths: List[str] = []
    for root, _, files in os.walk(video_root):
        for name in files:
            if not name.lower().endswith(tuple(exts)):
                continue
            video_paths.append(os.path.join(root, name))
            if max_items is not None and len(video_paths) >= max_items:
                return video_paths
    return video_paths


def _parse_ffprobe_ratio(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    if "/" in value:
        num, den = value.split("/", 1)
        try:
            num_f = float(num)
            den_f = float(den)
        except ValueError:
            return None
        if den_f == 0:
            return None
        return num_f / den_f
    try:
        return float(value)
    except ValueError:
        return None


def _ffprobe_video_stream(video_path: str) -> Dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,duration,nb_frames,avg_frame_rate",
        "-of",
        "json",
        video_path,
    ]
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    payload = json.loads(out.decode("utf-8"))
    streams = payload.get("streams") or []
    if not streams:
        raise RuntimeError(f"No video stream found in {video_path}")
    stream = streams[0]
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    duration = float(stream.get("duration") or 0.0)
    nb_frames = int(float(stream.get("nb_frames") or 0))
    avg_frame_rate = _parse_ffprobe_ratio(stream.get("avg_frame_rate"))
    if duration <= 0.0 and nb_frames > 0 and avg_frame_rate:
        duration = float(nb_frames) / float(avg_frame_rate)
    return {"width": width, "height": height, "duration": duration, "nb_frames": nb_frames, "fps": avg_frame_rate}


def _ffmpeg_read_video_frames(
    video_path: str,
    *,
    num_frames: int,
    scale_size: Optional[int] = 384,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    if num_frames <= 0:
        raise ValueError("num_frames must be > 0")

    meta = _ffprobe_video_stream(video_path)
    width = int(meta.get("width") or 0)
    height = int(meta.get("height") or 0)
    duration = float(meta.get("duration") or 0.0)

    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video dimensions for {video_path}: {width}x{height}")

    fps_sample = float(num_frames) / duration if duration > 0 else 1.0
    vf = f"fps={fps_sample}"
    out_w, out_h = width, height
    if scale_size is not None and scale_size > 0:
        out_w = int(scale_size)
        out_h = int(scale_size)
        vf = f"{vf},scale={out_w}:{out_h}:flags=bicubic"

    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        video_path,
        "-vf",
        vf,
        "-frames:v",
        str(num_frames),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    raw = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    frame_bytes = out_w * out_h * 3
    t = len(raw) // frame_bytes
    if t <= 0:
        raise RuntimeError(f"Empty video tensor from {video_path}")

    raw = raw[: t * frame_bytes]
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(t, out_h, out_w, 3)
    vframes = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()
    if vframes.shape[0] < num_frames:
        pad = num_frames - vframes.shape[0]
        vframes = torch.cat([vframes, vframes[-1:].repeat(pad, 1, 1, 1)], dim=0)

    info: Dict[str, Any] = {}
    if meta.get("fps") is not None:
        info["video_fps"] = float(meta["fps"])
    return vframes, info


def _pyav_read_video_streaming(
    video_path: str,
    *,
    num_frames: int,
    decode_audio: bool,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    import av

    if num_frames <= 0:
        raise ValueError("num_frames must be > 0")

    with av.open(video_path) as container:
        if not container.streams.video:
            raise RuntimeError(f"No video stream found in {video_path}")

        vstream = container.streams.video[0]
        fps = float(vstream.average_rate) if vstream.average_rate is not None else None
        total_frames = int(vstream.frames) if vstream.frames is not None else 0
        if total_frames <= 0 and vstream.duration is not None and fps is not None and vstream.time_base is not None:
            duration_sec = float(vstream.duration * vstream.time_base)
            total_frames = max(1, int(round(duration_sec * fps)))

        if total_frames > 0:
            frame_idx = torch.linspace(0, total_frames - 1, steps=num_frames).long().tolist()
        else:
            frame_idx = list(range(num_frames))

        wanted = set(frame_idx)
        max_idx = max(frame_idx) if frame_idx else -1

        selected: List[torch.Tensor] = []
        for i, frame in enumerate(container.decode(video=0)):
            if i in wanted:
                arr = frame.to_ndarray(format="rgb24")
                selected.append(torch.from_numpy(arr).permute(2, 0, 1).contiguous())
            if i >= max_idx:
                break

    if not selected:
        raise RuntimeError(f"Empty video tensor from {video_path}")

    vframes = torch.stack(selected, dim=0)
    if vframes.shape[0] < num_frames:
        pad = num_frames - vframes.shape[0]
        vframes = torch.cat([vframes, vframes[-1:].repeat(pad, 1, 1, 1)], dim=0)

    aframes = torch.empty((0,), dtype=torch.float32)
    audio_fps: Optional[int] = None
    if decode_audio:
        with av.open(video_path) as container:
            if container.streams.audio:
                astream = container.streams.audio[0]
                audio_fps = int(getattr(astream, "rate", None) or getattr(astream, "sample_rate", None) or 0) or None
                audio_chunks: List[torch.Tensor] = []
                for aframe in container.decode(audio=0):
                    arr = aframe.to_ndarray()
                    audio_chunks.append(torch.from_numpy(arr))
                if audio_chunks:
                    aframes = torch.cat(audio_chunks, dim=1).to(dtype=torch.float32)

    info: Dict[str, Any] = {}
    if fps is not None:
        info["video_fps"] = fps
    if audio_fps is not None:
        info["audio_fps"] = audio_fps
    return vframes, aframes, info


def load_video_frames_and_audio(
    video_path: str,
    *,
    num_frames: int,
    video_backend: str = "torchvision",
    decode_audio: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    if video_backend != "torchvision":
        raise ValueError(f"Unsupported video_backend={video_backend!r} (only 'torchvision' is supported).")

    info: Dict[str, Any] = {}

    if num_frames < 0:
        raise ValueError("num_frames must be >= 0")

    if num_frames == 0:
        vframes = torch.empty((0, 3, 0, 0), dtype=torch.uint8)
    else:
        try:
            import av  # noqa: F401

            vframes, _, video_info = _pyav_read_video_streaming(video_path, num_frames=num_frames, decode_audio=False)
            info.update(video_info)
        except ImportError:
            vframes, video_info = _ffmpeg_read_video_frames(video_path, num_frames=num_frames, scale_size=384)
            info.update(video_info)

        if vframes.numel() == 0:
            raise RuntimeError(f"Empty video tensor from {video_path}")

    aframes = torch.empty((0,), dtype=torch.float32)
    if decode_audio:
        try:
            import torchaudio

            waveform, sr = torchaudio.load(video_path)
            aframes = waveform.to(dtype=torch.float32)
            info["audio_fps"] = int(sr)
        except Exception:
            aframes = torch.empty((0,), dtype=torch.float32)

    return vframes, aframes, info


def to_mono(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.ndim == 1:
        return waveform
    if waveform.ndim != 2:
        raise ValueError(f"Expected waveform of shape (num_samples,) or (num_samples, channels), got {tuple(waveform.shape)}")
    return waveform.mean(dim=1)


def resample_waveform(waveform: torch.Tensor, *, orig_sr: int, target_sr: int) -> torch.Tensor:
    if orig_sr == target_sr:
        return waveform
    if orig_sr <= 0 or target_sr <= 0:
        raise ValueError("Sample rates must be positive.")

    try:
        import torchaudio

        return torchaudio.functional.resample(waveform, orig_sr, target_sr)
    except Exception:
        waveform_ = waveform.unsqueeze(0).unsqueeze(0)
        new_len = int(math.floor(waveform.shape[0] * target_sr / orig_sr))
        waveform_ = F.interpolate(waveform_, size=new_len, mode="linear", align_corners=False)
        return waveform_.squeeze(0).squeeze(0)


def _hz_to_mel(freq_hz: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + freq_hz / 700.0)


def _mel_to_hz(freq_mel: torch.Tensor) -> torch.Tensor:
    return 700.0 * (10.0 ** (freq_mel / 2595.0) - 1.0)


def create_mel_filter(
    *,
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    f_min: float = 0.0,
    f_max: Optional[float] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if f_max is None:
        f_max = float(sample_rate) / 2.0

    if not (0.0 <= f_min < f_max):
        raise ValueError(f"Invalid mel range: f_min={f_min}, f_max={f_max}")

    n_freqs = n_fft // 2 + 1
    fft_freqs = torch.linspace(0, float(sample_rate) / 2.0, steps=n_freqs, device=device, dtype=dtype)

    mel_min = _hz_to_mel(torch.tensor([f_min], device=device, dtype=dtype))[0]
    mel_max = _hz_to_mel(torch.tensor([f_max], device=device, dtype=dtype))[0]
    mel_points = torch.linspace(mel_min, mel_max, steps=n_mels + 2, device=device, dtype=dtype)
    hz_points = _mel_to_hz(mel_points)

    bin_f = torch.floor((n_fft + 1) * hz_points / float(sample_rate)).long()
    bin_f = torch.clamp(bin_f, 0, n_freqs - 1)

    fb = torch.zeros((n_mels, n_freqs), device=device, dtype=dtype)
    for i in range(n_mels):
        left = bin_f[i].item()
        center = bin_f[i + 1].item()
        right = bin_f[i + 2].item()
        if center == left:
            center += 1
        if right == center:
            right += 1
        if right <= left:
            continue

        up = torch.arange(left, center, device=device, dtype=dtype)
        down = torch.arange(center, right, device=device, dtype=dtype)
        if up.numel() > 0:
            fb[i, left:center] = (up - float(left)) / float(center - left)
        if down.numel() > 0:
            fb[i, center:right] = (float(right) - down) / float(right - center)
    return fb


def waveform_to_log_mel(
    waveform: torch.Tensor,
    *,
    sample_rate: int,
    target_seconds: Optional[float] = None,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: Optional[int] = None,
    n_mels: int = 128,
    eps: float = 1e-6,
) -> torch.Tensor:
    if win_length is None:
        win_length = n_fft

    waveform = waveform.float()
    if target_seconds is not None:
        target_len = int(round(target_seconds * sample_rate))
        if waveform.shape[0] > target_len:
            waveform = waveform[:target_len]
        elif waveform.shape[0] < target_len:
            pad = target_len - waveform.shape[0]
            waveform = F.pad(waveform, (0, pad))

    window = torch.hann_window(win_length, device=waveform.device, dtype=waveform.dtype)
    stft = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        pad_mode="reflect",
        return_complex=True,
    )
    power_spec = (stft.real**2 + stft.imag**2)

    mel_fb = create_mel_filter(sample_rate=sample_rate, n_fft=n_fft, n_mels=n_mels, device=waveform.device, dtype=waveform.dtype)
    mel = mel_fb @ power_spec
    mel = torch.log(mel + eps)
    return mel


def waveform_to_imagebind_melspec(
    waveform: torch.Tensor,
    *,
    sample_rate: int,
    num_mel_bins: int = 128,
    target_length: int = 204,
    n_fft: int = 400,
    hop_length: int = 160,
    win_length: int = 400,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    ImageBind-style mel spectrogram used for audio pseudo-image conditioning.

    Returns a tensor of shape (1, num_mel_bins, target_length).
    """
    if waveform.ndim != 1:
        raise ValueError(f"Expected mono waveform of shape (num_samples,), got {tuple(waveform.shape)}")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if target_length <= 0 or num_mel_bins <= 0:
        raise ValueError("target_length and num_mel_bins must be positive")
    if n_fft <= 0 or hop_length <= 0 or win_length <= 0:
        raise ValueError("n_fft/hop_length/win_length must be positive")

    waveform = waveform.float()
    window = torch.hann_window(win_length, device=waveform.device, dtype=waveform.dtype)
    stft = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        pad_mode="reflect",
        return_complex=True,
    )
    power_spec = (stft.real**2 + stft.imag**2)

    mel_fb = create_mel_filter(
        sample_rate=sample_rate,
        n_fft=n_fft,
        n_mels=num_mel_bins,
        device=waveform.device,
        dtype=waveform.dtype,
    )
    mel = mel_fb @ power_spec
    mel = torch.log(mel + eps)

    if mel.shape[1] > target_length:
        mel = mel[:, :target_length]
    elif mel.shape[1] < target_length:
        mel = F.pad(mel, (0, target_length - mel.shape[1]))

    return mel.unsqueeze(0)


def waveform_to_imagebind_melspec_clips(
    waveform: torch.Tensor,
    *,
    sample_rate: int,
    num_clips: int,
    clip_duration_s: float = 2.0,
    clip_stride_s: float = 0.5,
    num_mel_bins: int = 128,
    target_length: int = 204,
    mean: float = -4.268,
    std: float = 9.138,
    n_fft: int = 400,
    hop_length: int = 160,
    win_length: int = 400,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Build a stack of ImageBind-style melspec clips from a mono waveform.

    Output shape: (num_clips, 1, num_mel_bins, target_length)
    """
    if num_clips <= 0:
        raise ValueError("num_clips must be > 0")
    if clip_duration_s <= 0:
        raise ValueError("clip_duration_s must be > 0")
    if clip_stride_s < 0:
        raise ValueError("clip_stride_s must be >= 0")
    if std == 0:
        raise ValueError("std must be non-zero")

    clip_len = int(round(float(clip_duration_s) * float(sample_rate)))
    stride = int(round(float(clip_stride_s) * float(sample_rate)))
    if clip_len <= 0:
        raise ValueError("clip_duration_s is too small for the given sample_rate")
    if stride < 0:
        raise ValueError("clip_stride_s is invalid for the given sample_rate")

    clips: List[torch.Tensor] = []
    for i in range(int(num_clips)):
        start = int(i * stride)
        end = start + clip_len
        wav = waveform[start:end]
        if wav.shape[0] < clip_len:
            wav = F.pad(wav, (0, clip_len - wav.shape[0]))

        mel = waveform_to_imagebind_melspec(
            wav,
            sample_rate=sample_rate,
            num_mel_bins=num_mel_bins,
            target_length=target_length,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            eps=eps,
        )
        mel = (mel - float(mean)) / float(std)
        clips.append(mel)

    return torch.stack(clips, dim=0)


def log_mel_to_pil_rgb(log_mel: torch.Tensor) -> Any:
    from PIL import Image

    x = log_mel.detach().cpu().float()
    x = x - x.min()
    denom = x.max().clamp_min(1e-6)
    x = x / denom
    x = (x * 255.0).clamp(0, 255).byte().numpy()
    x = np.stack([x, x, x], axis=-1)
    return Image.fromarray(x, mode="RGB")


def _iter_model_candidates(model: Any) -> List[Any]:
    model = getattr(model, "module", model)
    candidates: List[Any] = []

    def add(obj: Any) -> None:
        if obj is None:
            return
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


def _get_model_attr(model: Any, name: str) -> Any:
    for cand in _iter_model_candidates(model):
        if hasattr(cand, name):
            return getattr(cand, name)
    raise AttributeError(f"Could not find attribute {name!r} on model or common wrapped variants.")


def _llava_onevision_apply_pooling(image_features: torch.Tensor) -> torch.Tensor:
    if image_features.ndim != 3:
        raise ValueError(f"Expected image_features of shape (N, S, D), got {tuple(image_features.shape)}")

    batch_frames, seq_len, dim = image_features.shape

    grid = int(round(math.sqrt(seq_len)))
    patch = image_features
    if grid * grid != seq_len:
        grid2 = int(round(math.sqrt(seq_len - 1))) if seq_len > 1 else 0
        if grid2 > 0 and grid2 * grid2 == seq_len - 1:
            patch = image_features[:, 1:, :]
            seq_len = patch.shape[1]
            grid = grid2
        else:
            return image_features

    patch = patch.reshape(batch_frames, grid, grid, dim).permute(0, 3, 1, 2).contiguous()
    scaled_shape = [math.ceil(grid / 2), math.ceil(grid / 2)]
    patch = F.interpolate(patch, size=scaled_shape, mode="bilinear")
    patch = patch.permute(0, 2, 3, 1).contiguous().reshape(batch_frames, -1, dim)
    return patch


def encode_llava_onevision_projected(
    model: Any,
    *,
    pixel_values: torch.Tensor,
    vision_feature_layer: int,
    vision_feature_select_strategy: str,
) -> torch.Tensor:
    vision_tower = _get_model_attr(model, "vision_tower")
    projector = _get_model_attr(model, "multi_modal_projector")

    outputs = vision_tower(pixel_values, output_hidden_states=True)
    if isinstance(vision_feature_layer, int):
        selected = outputs.hidden_states[vision_feature_layer]
    else:
        hs_pool = [outputs.hidden_states[layer_idx] for layer_idx in vision_feature_layer]
        selected = torch.cat(hs_pool, dim=-1)
    if vision_feature_select_strategy == "default":
        selected = selected[:, 1:]
    elif vision_feature_select_strategy == "full":
        selected = selected
    else:
        raise ValueError(f"Unsupported vision_feature_select_strategy={vision_feature_select_strategy!r}")
    return projector(selected)


def encode_llava_onevision_video_tokens(
    model: Any,
    *,
    pixel_values_videos: torch.Tensor,
    vision_feature_layer: int,
    vision_feature_select_strategy: str,
    repeat_frames: Optional[int] = None,
    frame_chunk_size: int = 4,
) -> torch.Tensor:
    model = getattr(model, "module", model)
    if pixel_values_videos.ndim != 5:
        raise ValueError(f"Expected pixel_values_videos of shape (B, F, C, H, W), got {tuple(pixel_values_videos.shape)}")

    batch_size, frames, channels, height, width = pixel_values_videos.shape
    if frame_chunk_size <= 0:
        raise ValueError("frame_chunk_size must be > 0")

    def get_video_features_fallback(pv: torch.Tensor) -> torch.Tensor:
        get_video_features = None
        for cand in _iter_model_candidates(model):
            if hasattr(cand, "get_video_features"):
                get_video_features = getattr(cand, "get_video_features")
                break
        if callable(get_video_features):
            return get_video_features(pv, vision_feature_layer=vision_feature_layer, vision_feature_select_strategy=vision_feature_select_strategy)

        flat = pv.reshape(batch_size * pv.shape[1], channels, height, width)
        projected = encode_llava_onevision_projected(
            model,
            pixel_values=flat,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
        )
        projected = _llava_onevision_apply_pooling(projected)
        seq_len = projected.shape[1]
        return projected.reshape(batch_size, pv.shape[1] * seq_len, -1)

    if frames > frame_chunk_size and repeat_frames is None:
        chunks: List[torch.Tensor] = []
        for start in range(0, frames, frame_chunk_size):
            pv = pixel_values_videos[:, start : start + frame_chunk_size, ...]
            chunks.append(get_video_features_fallback(pv))
        pooled = torch.cat(chunks, dim=1)
    else:
        pooled = get_video_features_fallback(pixel_values_videos)

    if pooled.shape[1] % frames != 0:
        raise ValueError(f"Unexpected video feature length: {pooled.shape[1]} is not divisible by frames={frames}")
    seq_len = pooled.shape[1] // frames
    pooled = pooled.reshape(batch_size, frames, seq_len, -1)

    if repeat_frames is not None and repeat_frames != frames:
        if frames != 1:
            raise ValueError("repeat_frames is only supported when input has frames=1")
        pooled = pooled.repeat(1, repeat_frames, 1, 1)
        frames = repeat_frames

    pooled = pooled.reshape(batch_size, frames * seq_len, -1)
    newline_source = None
    for cand in _iter_model_candidates(model):
        if hasattr(cand, "image_newline"):
            newline_source = getattr(cand, "image_newline")
            break
    if newline_source is None:
        raise AttributeError("Expected LlavaOnevision model to expose `image_newline`.")

    newline = newline_source[None, None, :].to(device=pooled.device, dtype=pooled.dtype).repeat(batch_size, 1, 1)
    return torch.cat([pooled, newline], dim=1)


def tile_pixel_values_videos_spatially(
    pixel_values_videos: torch.Tensor,
    *,
    tile_size: int,
) -> torch.Tensor:
    """
    Spatially tile `tile_size` consecutive frames into a single frame by downsampling each frame
    to a sqrt(tile_size) x sqrt(tile_size) grid and placing them in row-major order.

    Supports:
      - (F, C, H, W) -> (F // tile_size, C, H, W)
      - (B, F, C, H, W) -> (B, F // tile_size, C, H, W)

    If `F` is not divisible by `tile_size`, the last frame is repeated to pad to a multiple.
    """

    if tile_size <= 0:
        raise ValueError("tile_size must be > 0")
    if tile_size == 1:
        return pixel_values_videos

    squeeze_batch = False
    pv = pixel_values_videos
    if pv.ndim == 4:
        pv = pv.unsqueeze(0)
        squeeze_batch = True
    if pv.ndim != 5:
        raise ValueError(f"Expected pixel_values_videos of shape (F, C, H, W) or (B, F, C, H, W), got {tuple(pixel_values_videos.shape)}")

    batch, frames, channels, height, width = pv.shape

    grid = int(math.isqrt(int(tile_size)))
    if grid * grid != int(tile_size):
        raise ValueError(f"tile_size must be a perfect square for spatial tiling, got tile_size={tile_size}")
    if height % grid != 0 or width % grid != 0:
        raise ValueError(f"Input HxW must be divisible by sqrt(tile_size)={grid}, got {height}x{width}")

    # pad frames to a multiple of tile_size by repeating the last frame (overlapping sampling for short videos)
    if frames % tile_size != 0:
        pad = int(tile_size - (frames % tile_size))
        last = pv[:, -1:, ...].expand(batch, pad, channels, height, width)
        pv = torch.cat([pv, last], dim=1)
        frames = pv.shape[1]

    out_frames = frames // tile_size
    sub_h = height // grid
    sub_w = width // grid

    # (B, outF, T, C, H, W) -> (B*outF*T, C, H, W)
    pv = pv.reshape(batch, out_frames, tile_size, channels, height, width)
    pv_flat = pv.reshape(batch * out_frames * tile_size, channels, height, width)
    pv_small = F.interpolate(pv_flat, size=(sub_h, sub_w), mode="bilinear", align_corners=False)
    pv_small = pv_small.reshape(batch, out_frames, tile_size, channels, sub_h, sub_w)

    out = torch.empty((batch, out_frames, channels, height, width), device=pv_small.device, dtype=pv_small.dtype)
    for t in range(tile_size):
        rr = t // grid
        cc = t % grid
        out[:, :, :, rr * sub_h : (rr + 1) * sub_h, cc * sub_w : (cc + 1) * sub_w] = pv_small[:, :, t, :, :, :]

    if squeeze_batch:
        return out[0]
    return out


@dataclass
class RunningMeanVar:
    count: int
    mean: torch.Tensor
    m2: torch.Tensor


def running_mean_var_init(hidden_size: int, *, device: torch.device) -> RunningMeanVar:
    return RunningMeanVar(
        count=0,
        mean=torch.zeros(hidden_size, device=device, dtype=torch.float64),
        m2=torch.zeros(hidden_size, device=device, dtype=torch.float64),
    )


def running_mean_var_update(state: RunningMeanVar, x: torch.Tensor) -> None:
    if x.ndim != 2:
        raise ValueError(f"Expected x to have shape (N, D), got {tuple(x.shape)}")
    x = x.detach().to(dtype=torch.float64)
    n = x.shape[0]
    if n == 0:
        return
    batch_mean = x.mean(dim=0)
    batch_var = x.var(dim=0, unbiased=False)

    if state.count == 0:
        state.count = n
        state.mean.copy_(batch_mean)
        state.m2.copy_(batch_var * n)
        return

    total = state.count + n
    delta = batch_mean - state.mean
    state.mean.add_(delta * (n / total))
    state.m2.add_(batch_var * n + (delta**2) * (state.count * n / total))
    state.count = total


def running_mean_var_finalize(state: RunningMeanVar) -> Tuple[torch.Tensor, torch.Tensor]:
    if state.count == 0:
        raise ValueError("Cannot finalize RunningMeanVar with count=0")
    mean = state.mean.to(dtype=torch.float32)
    var = (state.m2 / state.count).to(dtype=torch.float32)
    return mean, var


def build_dummy_prompt_token_ids(
    tokenizer: Any,
    *,
    prompt_text: str,
    placeholder_token: str,
) -> Tuple[torch.Tensor, int]:
    ids = tokenizer(prompt_text, return_tensors="pt").input_ids[0]
    placeholder_id = tokenizer.convert_tokens_to_ids(placeholder_token)
    if placeholder_id is None or placeholder_id == getattr(tokenizer, "unk_token_id", None):
        raise ValueError(f"placeholder_token={placeholder_token!r} is not a single known token for this tokenizer")
    pos = (ids == placeholder_id).nonzero()
    if pos.numel() != 1:
        raise ValueError(f"Expected exactly 1 placeholder token ({placeholder_token}) in dummy prompt, got {pos.numel()}")
    return ids, int(pos.item())
