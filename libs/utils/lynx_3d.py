import importlib.util
import os
import shutil
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image


def numeric_stem(path: str) -> int:
    stem = os.path.splitext(os.path.basename(path))[0]
    try:
        return int(stem)
    except ValueError:
        return 0


def list_image_paths(root: str, *, exts: Sequence[str]) -> List[str]:
    if not os.path.isdir(root):
        return []
    out: List[str] = []
    for name in os.listdir(root):
        if not any(name.lower().endswith(ext) for ext in exts):
            continue
        out.append(os.path.join(root, name))
    out.sort(key=numeric_stem)
    return out


def sample_uniform(items: Sequence[Any], num: int) -> List[Any]:
    if num <= 0:
        raise ValueError("num must be > 0")
    if not items:
        return []
    if num == 1:
        return [items[len(items) // 2]]
    idxs = torch.linspace(0, len(items) - 1, steps=num).long().tolist()
    return [items[i] for i in idxs]


def read_pose_c2w(
    scene_dir: str,
    *,
    frame_key: int,
    pose_subdir: str,
    pose_matrix_type: str,
) -> Optional[np.ndarray]:
    pose_path = os.path.join(scene_dir, str(pose_subdir), f"{int(frame_key)}.txt")
    if not os.path.exists(pose_path):
        return None
    rows: List[List[float]] = []
    with open(pose_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p for p in line.split() if p]
            try:
                rows.append([float(p) for p in parts])
            except ValueError:
                return None
    if len(rows) != 4 or any(len(r) < 4 for r in rows):
        return None
    mat = np.array([r[:4] for r in rows], dtype=np.float32)

    pose_matrix_type = str(pose_matrix_type or "c2w").lower().strip()
    if pose_matrix_type not in ("c2w", "w2c"):
        raise ValueError(f"Unsupported pose_matrix_type={pose_matrix_type!r}. Expected 'c2w' or 'w2c'.")
    if pose_matrix_type == "w2c":
        try:
            mat = np.linalg.inv(mat).astype(np.float32)
        except Exception:
            return None
    return mat


def pose_features(pose_c2w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    t = pose_c2w[:3, 3].astype(np.float32)
    view = pose_c2w[:3, 2].astype(np.float32)
    denom = float(np.linalg.norm(view)) + 1e-6
    view = view / denom
    return t, view


def farthest_point_sample(features: np.ndarray, k: int) -> List[int]:
    if features.ndim != 2:
        raise ValueError(f"Expected features of shape (N, D), got {features.shape}")
    n = int(features.shape[0])
    if k <= 0:
        raise ValueError("k must be > 0")
    if n == 0:
        return []
    if k >= n:
        return list(range(n))

    mean = features.mean(axis=0, keepdims=True)
    start = int(np.argmax(np.linalg.norm(features - mean, axis=1)))

    selected: List[int] = [start]
    min_dist = np.full((n,), np.inf, dtype=np.float32)
    for _ in range(k - 1):
        last = features[selected[-1]][None, :]
        dist = np.linalg.norm(features - last, axis=1).astype(np.float32)
        min_dist = np.minimum(min_dist, dist)
        nxt = int(np.argmax(min_dist))
        selected.append(nxt)
    return selected


def sample_pose_aware_pairs(
    pairs: Sequence[Tuple[str, str]],
    *,
    scene_dir: str,
    num_frames: int,
    pose_subdir: str,
    pose_matrix_type: str,
) -> List[Tuple[str, str]]:
    if not pairs:
        return []
    if num_frames <= 0:
        raise ValueError("num_frames must be > 0")
    if num_frames == 1:
        return [pairs[len(pairs) // 2]]

    translations: List[np.ndarray] = []
    views: List[np.ndarray] = []
    valid_pairs: List[Tuple[str, str]] = []
    for color_path, depth_path in pairs:
        key = numeric_stem(color_path)
        pose = read_pose_c2w(scene_dir, frame_key=key, pose_subdir=pose_subdir, pose_matrix_type=pose_matrix_type)
        if pose is None:
            continue
        t, view = pose_features(pose)
        translations.append(t)
        views.append(view)
        valid_pairs.append((color_path, depth_path))

    if len(valid_pairs) < 2:
        return sample_uniform(list(pairs), num_frames)

    t_arr = np.stack(translations, axis=0)
    v_arr = np.stack(views, axis=0)
    t_std = t_arr.std(axis=0, keepdims=True)
    t_std = np.maximum(t_std, 1e-6)
    t_norm = (t_arr - t_arr.mean(axis=0, keepdims=True)) / t_std

    view_weight = 1.0
    feats = np.concatenate([t_norm, v_arr * view_weight], axis=1).astype(np.float32)

    picked = farthest_point_sample(feats, int(num_frames))
    picked_pairs = [valid_pairs[i] for i in picked]

    picked_pairs.sort(key=lambda p: numeric_stem(p[0]))
    if len(picked_pairs) < int(num_frames):
        while len(picked_pairs) < int(num_frames):
            picked_pairs.append(picked_pairs[-1])
    return picked_pairs


def _turbo_lut_uint8() -> np.ndarray:
    xs = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    v4 = np.stack([np.ones_like(xs), xs, xs * xs, xs * xs * xs], axis=-1)
    v2 = np.stack([xs**4, xs**5], axis=-1)

    k_red_v4 = np.array([0.13572138, 4.61539260, -42.66032258, 132.13108234], dtype=np.float32)
    k_green_v4 = np.array([0.09140261, 2.19418839, 4.84296658, -14.18503333], dtype=np.float32)
    k_blue_v4 = np.array([0.10667330, 12.64194608, -60.58204836, 110.36276771], dtype=np.float32)

    k_red_v2 = np.array([-152.94239396, 59.28637943], dtype=np.float32)
    k_green_v2 = np.array([4.27729857, 2.82956604], dtype=np.float32)
    k_blue_v2 = np.array([-89.90310912, 27.34824973], dtype=np.float32)

    r = v4 @ k_red_v4 + v2 @ k_red_v2
    g = v4 @ k_green_v4 + v2 @ k_green_v2
    b = v4 @ k_blue_v4 + v2 @ k_blue_v2

    rgb = np.stack([r, g, b], axis=-1)
    rgb = np.clip(rgb, 0.0, 1.0)
    return (rgb * 255.0).round().astype(np.uint8)


_TURBO_LUT = _turbo_lut_uint8()


def read_intrinsics(scene_dir: str, *, filename: str) -> Optional[Tuple[float, float, float, float]]:
    path = os.path.join(scene_dir, filename)
    if not os.path.exists(path):
        return None
    rows: List[List[float]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p for p in line.split() if p]
            try:
                rows.append([float(p) for p in parts])
            except ValueError:
                continue
    if len(rows) < 2 or any(len(r) < 3 for r in rows[:2]):
        return None
    fx = float(rows[0][0])
    fy = float(rows[1][1])
    cx = float(rows[0][2])
    cy = float(rows[1][2])
    if fx <= 0.0 or fy <= 0.0:
        return None
    return fx, fy, cx, cy


def scale_intrinsics_to_image(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    *,
    width: int,
    height: int,
) -> Tuple[float, float, float, float]:
    base_w = max(1.0, 2.0 * float(cx) + 1.0)
    base_h = max(1.0, 2.0 * float(cy) + 1.0)
    scale_x = float(width) / base_w
    scale_y = float(height) / base_h
    return fx * scale_x, fy * scale_y, cx * scale_x, cy * scale_y


def load_depth_mm(path: str) -> np.ndarray:
    with Image.open(path) as im:
        arr = np.array(im)
    return arr.astype(np.float32)


def prepare_depth_mm(
    depth_mm: np.ndarray,
    *,
    clip_min_mm: float,
    clip_max_mm: float,
    invalid_to_max: bool,
) -> np.ndarray:
    if clip_max_mm <= clip_min_mm:
        raise ValueError(f"Invalid depth clip range: min={clip_min_mm}, max={clip_max_mm}")
    depth = depth_mm.astype(np.float32, copy=True)
    invalid = depth <= 0.0
    if invalid_to_max:
        depth[invalid] = float(clip_max_mm)
    depth = np.clip(depth, float(clip_min_mm), float(clip_max_mm))
    return depth


def depth_gray_rgb(depth_mm: np.ndarray, *, clip_min_mm: float, clip_max_mm: float) -> np.ndarray:
    depth = prepare_depth_mm(depth_mm, clip_min_mm=clip_min_mm, clip_max_mm=clip_max_mm, invalid_to_max=True)
    scaled = (depth - float(clip_min_mm)) / float(clip_max_mm - clip_min_mm)
    img = (scaled * 255.0).round().astype(np.uint8)
    return np.stack([img, img, img], axis=-1)


def depth_disparity_turbo_rgb(depth_mm: np.ndarray, *, clip_min_mm: float, clip_max_mm: float) -> np.ndarray:
    depth = prepare_depth_mm(depth_mm, clip_min_mm=clip_min_mm, clip_max_mm=clip_max_mm, invalid_to_max=True)
    depth_m = depth / 1000.0
    eps = 1e-6
    disp = 1.0 / (depth_m + eps)
    disp_min = 1.0 / (float(clip_max_mm) / 1000.0 + eps)
    disp_max = 1.0 / (float(clip_min_mm) / 1000.0 + eps)
    denom = max(eps, float(disp_max - disp_min))
    disp_norm = (disp - float(disp_min)) / denom
    disp_norm = np.clip(disp_norm, 0.0, 1.0)
    idx = (disp_norm * 255.0).round().astype(np.uint8)
    return _TURBO_LUT[idx]


def depth_normals_rgb(
    depth_mm: np.ndarray,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    clip_min_mm: float,
    clip_max_mm: float,
    rotation_c2w: Optional[np.ndarray] = None,
) -> np.ndarray:
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError(f"Invalid intrinsics: fx={fx}, fy={fy}")

    invalid = depth_mm <= 0.0
    depth = prepare_depth_mm(depth_mm, clip_min_mm=clip_min_mm, clip_max_mm=clip_max_mm, invalid_to_max=False)
    depth_m = depth / 1000.0
    depth_m = np.where(invalid, np.nan, depth_m)

    height, width = depth_m.shape
    u = np.arange(width, dtype=np.float32)
    v = np.arange(height, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    x = (uu - float(cx)) / float(fx) * depth_m
    y = (vv - float(cy)) / float(fy) * depth_m
    z = depth_m
    points = np.stack([x, y, z], axis=-1)

    vx = points[:, 1:, :] - points[:, :-1, :]
    vy = points[1:, :, :] - points[:-1, :, :]
    n = np.cross(vx[:-1, :, :], vy[:, :-1, :])

    normals = np.zeros((height, width, 3), dtype=np.float32)
    normals[:-1, :-1, :] = n
    normals[-1, :-1, :] = normals[-2, :-1, :]
    normals[:-1, -1, :] = normals[:-1, -2, :]
    normals[-1, -1, :] = normals[-2, -2, :]

    normals = np.where(np.isfinite(normals), normals, 0.0)
    flip = normals[..., 2] < 0.0
    normals[flip] *= -1.0
    denom = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = np.divide(normals, denom, out=np.zeros_like(normals), where=denom > 0.0)

    if rotation_c2w is not None:
        rot = np.asarray(rotation_c2w, dtype=np.float32)
        if rot.shape != (3, 3):
            raise ValueError(f"rotation_c2w must be (3,3), got {rot.shape}")
        normals = normals @ rot.T
        denom2 = np.linalg.norm(normals, axis=-1, keepdims=True)
        normals = np.divide(normals, denom2, out=np.zeros_like(normals), where=denom2 > 0.0)
    return ((normals * 0.5 + 0.5) * 255.0).round().astype(np.uint8)


def pil_rgb_to_chw_uint8(path: str) -> torch.Tensor:
    with Image.open(path) as im:
        arr = np.array(im.convert("RGB"))
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def encode_depth_to_rgb_frames(
    depth_path: str,
    *,
    scene_dir: str,
    encoding: str,
    clip_min_mm: float,
    clip_max_mm: float,
    intrinsics_filename: str,
    auto_scale_intrinsics: bool,
    pose_c2w: Optional[np.ndarray] = None,
    normals_frame: str = "camera",
) -> List[torch.Tensor]:
    depth_mm = load_depth_mm(depth_path)
    encoding = str(encoding or "turbo").lower()
    if encoding not in ("gray", "turbo", "normals", "turbo+normals"):
        raise ValueError(f"Unsupported depth encoding: {encoding!r}")

    if encoding == "gray":
        rgb = depth_gray_rgb(depth_mm, clip_min_mm=clip_min_mm, clip_max_mm=clip_max_mm)
        return [torch.from_numpy(rgb).permute(2, 0, 1).contiguous()]

    if encoding == "turbo":
        rgb = depth_disparity_turbo_rgb(depth_mm, clip_min_mm=clip_min_mm, clip_max_mm=clip_max_mm)
        return [torch.from_numpy(rgb).permute(2, 0, 1).contiguous()]

    intr = read_intrinsics(scene_dir, filename=intrinsics_filename)
    if intr is None:
        intr = (1.0, 1.0, 0.0, 0.0)
    fx, fy, cx, cy = intr

    height, width = depth_mm.shape
    if auto_scale_intrinsics:
        fx, fy, cx, cy = scale_intrinsics_to_image(fx, fy, cx, cy, width=width, height=height)

    normals_frame = str(normals_frame or "camera").lower().strip()
    if normals_frame not in ("camera", "world"):
        raise ValueError(f"Unsupported normals_frame={normals_frame!r}. Expected 'camera' or 'world'.")
    rotation_c2w = None
    if normals_frame == "world" and pose_c2w is not None:
        rotation_c2w = np.asarray(pose_c2w, dtype=np.float32)[:3, :3]

    normals = depth_normals_rgb(
        depth_mm,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        clip_min_mm=clip_min_mm,
        clip_max_mm=clip_max_mm,
        rotation_c2w=rotation_c2w,
    )
    normals_t = torch.from_numpy(normals).permute(2, 0, 1).contiguous()

    if encoding == "normals":
        return [normals_t]

    turbo = depth_disparity_turbo_rgb(depth_mm, clip_min_mm=clip_min_mm, clip_max_mm=clip_max_mm)
    turbo_t = torch.from_numpy(turbo).permute(2, 0, 1).contiguous()
    return [turbo_t, normals_t]


def load_module_from_file(module_name: str, path: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {module_name!r} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sanitize_metric_key(prefix: str, raw: str) -> str:
    key = str(raw).strip().lower()
    key = key.replace(prefix.lower(), prefix.lower().strip("[]"))
    key = key.replace("[", "").replace("]", "")
    key = key.replace(" ", "_").replace("-", "_").replace("/", "_")
    key = key.replace("__", "_")
    return key


def is_rank_zero(training_args: Any) -> bool:
    local_rank = int(getattr(training_args, "local_rank", -1) or -1)
    return local_rank in (-1, 0)


def warn_optional_eval_deps(
    dataset_name: str,
    *,
    scanqa_args: Any,
    sqa3d_args: Any,
) -> None:
    dataset_name = str(dataset_name or "").lower().strip()

    if dataset_name == "scanqa" and bool(getattr(scanqa_args, "enable_scanqa_eval", False)):
        missing: List[str] = []
        try:
            import pycocoevalcap  # type: ignore[import-not-found]  # noqa: F401
        except Exception:
            missing.append("pycocoevalcap")

        has_java = shutil.which("java") is not None
        if missing or not has_java:
            parts: List[str] = []
            if missing:
                parts.append(f"missing {', '.join(missing)}")
            if not has_java:
                parts.append("java not found (needed by METEOR/SPICE in pycocoevalcap)")
            detail = "; ".join(parts)
            print(
                f"[Deps] ScanQA eval enabled but {detail}. Metrics may be skipped.\n"
                f"       Install: `pip install pycocoevalcap` and ensure `java` is in PATH, or disable eval with `--enable_scanqa_eval False`.",
                file=sys.stderr,
            )

    if dataset_name == "sqa" and bool(getattr(sqa3d_args, "enable_sqa3d_eval", False)):
        return


def load_video_rgb_depth_tensors(
    video_name: str,
    *,
    frames_root: str,
    color_subdir: str,
    depth_subdir: str,
    pose_subdir: str = "pose",
    pose_matrix_type: str = "c2w",
    num_frames: int,
    depth_encoding: str,
    depth_clip_min_mm: float,
    depth_clip_max_mm: float,
    depth_normals_frame: str = "camera",
    depth_intrinsics_filename: str,
    depth_auto_scale_intrinsics: bool,
    pairs_cache: Dict[str, List[Tuple[str, str]]],
    selected_cache: Optional[Dict[str, List[Tuple[str, str]]]] = None,
    pose_cache: Optional[Dict[Tuple[str, int], Optional[np.ndarray]]] = None,
    frame_sampling: str = "pose",
) -> Tuple[torch.Tensor, torch.Tensor]:
    video_name = str(video_name)
    scene_dir = os.path.join(str(frames_root), video_name)
    pairs = pairs_cache.get(video_name)
    if pairs is None:
        color_dir = os.path.join(scene_dir, str(color_subdir))
        depth_dir = os.path.join(scene_dir, str(depth_subdir))
        color_paths = list_image_paths(color_dir, exts=(".jpg", ".jpeg", ".png"))
        depth_paths = list_image_paths(depth_dir, exts=(".png", ".jpg", ".jpeg"))
        depth_by_key = {numeric_stem(p): p for p in depth_paths}
        pairs = []
        for cpath in color_paths:
            dpath = depth_by_key.get(numeric_stem(cpath))
            if dpath:
                pairs.append((cpath, dpath))
        pairs_cache[video_name] = pairs

    selected: List[Tuple[str, str]]
    if selected_cache is not None and video_name in selected_cache:
        selected = selected_cache[video_name]
    else:
        frame_sampling = str(frame_sampling or "uniform").lower().strip()
        if frame_sampling == "pose":
            selected = sample_pose_aware_pairs(
                pairs,
                scene_dir=scene_dir,
                num_frames=int(num_frames),
                pose_subdir=str(pose_subdir),
                pose_matrix_type=str(pose_matrix_type),
            )
        else:
            selected = sample_uniform(pairs, int(num_frames))
        if selected_cache is not None:
            selected_cache[video_name] = selected
    if not selected:
        raise FileNotFoundError(f"No paired RGB/depth frames found for video={video_name}")

    rgb_frames: List[torch.Tensor] = []
    depth_frames: List[torch.Tensor] = []
    depth_encoding_l = str(depth_encoding or "turbo").lower().strip()
    need_pose_for_normals = str(depth_normals_frame or "camera").lower().strip() == "world" and "normals" in depth_encoding_l
    for color_path, depth_path in selected:
        key = numeric_stem(color_path)
        pose_c2w = None
        if need_pose_for_normals:
            if pose_cache is None:
                pose_c2w = read_pose_c2w(scene_dir, frame_key=key, pose_subdir=str(pose_subdir), pose_matrix_type=str(pose_matrix_type))
            else:
                cache_key = (video_name, int(key))
                if cache_key in pose_cache:
                    pose_c2w = pose_cache[cache_key]
                else:
                    pose_c2w = read_pose_c2w(scene_dir, frame_key=key, pose_subdir=str(pose_subdir), pose_matrix_type=str(pose_matrix_type))
                    pose_cache[cache_key] = pose_c2w
        rgb_frames.append(pil_rgb_to_chw_uint8(color_path))
        depth_frames.extend(
            encode_depth_to_rgb_frames(
                depth_path,
                scene_dir=scene_dir,
                encoding=depth_encoding_l,
                clip_min_mm=float(depth_clip_min_mm),
                clip_max_mm=float(depth_clip_max_mm),
                intrinsics_filename=str(depth_intrinsics_filename),
                auto_scale_intrinsics=bool(depth_auto_scale_intrinsics),
                pose_c2w=pose_c2w,
                normals_frame=str(depth_normals_frame),
            )
        )

    return torch.stack(rgb_frames, dim=0), torch.stack(depth_frames, dim=0)

