import argparse
import json
import os
import shutil
import sys
import time
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm.auto import tqdm
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from libs.model.lynx_onevision import LynXOnevisionWrapper
from lynx_utils import load_json, resolve_attn_implementation
from libs.utils.lynx_3d import load_module_from_file as _load_module_from_file, load_video_rgb_depth_tensors as _load_video_rgb_depth_tensors


def _short_exc(exc: BaseException, *, max_len: int = 300) -> str:
    msg = str(exc)
    if len(msg) > max_len:
        msg = msg[:max_len] + "..."
    return msg


def _is_cuda_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "cublas_status_alloc_failed" in msg


def resolve_local_model_path(model_name_or_path: str, *, cache_dir: Optional[str], local_files_only: bool) -> str:
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


def resolve_adapter_dir(path: str, adapter_name: str) -> str:
    if os.path.isfile(os.path.join(path, "adapter_config.json")):
        return path
    cand = os.path.join(path, adapter_name)
    if os.path.isfile(os.path.join(cand, "adapter_config.json")):
        return cand
    return path


def _init_distributed(local_rank: int) -> Tuple[int, int, torch.device]:
    if torch.cuda.is_available() and local_rank >= 0:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 and not torch.distributed.is_initialized():
        backend = "nccl" if device.type == "cuda" else "gloo"
        torch.distributed.init_process_group(backend=backend, init_method="env://")

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    else:
        rank = 0
        world_size = 1

    return rank, world_size, device


def _warn_optional_eval_deps() -> None:
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
            f"[Deps] ScanQA evaluator may skip metrics: {detail}.\n"
            f"       Install: `pip install pycocoevalcap` and ensure `java` is in PATH.",
            file=sys.stderr,
        )


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", "-1")))

    parser.add_argument("--model_name_or_path", type=str, default="llava-hf/llava-onevision-qwen2-0.5b-ov-hf")
    parser.add_argument("--vision_adapter_path", type=str, required=True)
    parser.add_argument("--llm_adapter_path", type=str, required=True)
    parser.add_argument("--vision_adapter_name", type=str, default="vision")
    parser.add_argument("--llm_adapter_name", type=str, default="llm")

    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--local_files_only", action="store_true", default=True)
    parser.add_argument("--no_local_files_only", dest="local_files_only", action="store_false")

    parser.add_argument("--question_file", type=str, default="./data/video_instruction_tuning/scannet/llava-3d-scanqa_val_question.json")
    parser.add_argument("--gt_file", type=str, default="./data/video_instruction_tuning/scannet/llava3d_scanqa_val_answer.json")
    parser.add_argument("--pred_save", type=str, default="./scanqa_predictions.jsonl")

    parser.add_argument("--frames_root", type=str, default="./data/video_instruction_tuning/3d/frames_square")
    parser.add_argument("--color_subdir", type=str, default="color")
    parser.add_argument("--depth_subdir", type=str, default="depth")
    parser.add_argument("--pose_subdir", type=str, default="pose")
    parser.add_argument("--pose_matrix_type", type=str, default="c2w")
    parser.add_argument("--frame_sampling", type=str, default="pose", help="uniform or pose (pose-aware sampling).")

    parser.add_argument("--num_frames", type=int, default=20)
    parser.add_argument("--frame_chunk_size", type=int, default=4)
    parser.add_argument("--pool_video_tokens", action="store_true", default=False)
    parser.add_argument("--no_pool_video_tokens", dest="pool_video_tokens", action="store_false")

    parser.add_argument("--depth_encoding", type=str, default="turbo")
    parser.add_argument("--depth_clip_min_mm", type=float, default=200.0)
    parser.add_argument("--depth_clip_max_mm", type=float, default=10000.0)
    parser.add_argument("--depth_intrinsics_filename", type=str, default="intrinsic_depth.txt")
    parser.add_argument("--depth_auto_scale_intrinsics", action="store_true", default=True)
    parser.add_argument("--no_depth_auto_scale_intrinsics", dest="depth_auto_scale_intrinsics", action="store_false")
    parser.add_argument("--depth_normals_frame", type=str, default="camera", help="camera or world (requires pose).")
    parser.add_argument("--depth_merge_frame_pairs", action="store_true", default=True)
    parser.add_argument("--no_depth_merge_frame_pairs", dest="depth_merge_frame_pairs", action="store_false")

    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--min_new_tokens", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=None)

    parser.add_argument("--use_flash_attn", action="store_true", default=True)
    parser.add_argument("--no_use_flash_attn", dest="use_flash_attn", action="store_false")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no_bf16", dest="bf16", action="store_false")
    parser.add_argument("--tf32", action="store_true", default=True)
    parser.add_argument("--no_tf32", dest="tf32", action="store_false")
    args = parser.parse_args()

    _warn_optional_eval_deps()

    rank, world_size, device = _init_distributed(args.local_rank)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
        torch.backends.cudnn.allow_tf32 = bool(args.tf32)

    resolved_model_path = resolve_local_model_path(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        local_files_only=bool(args.local_files_only),
    )

    processor = AutoProcessor.from_pretrained(resolved_model_path, cache_dir=args.cache_dir, local_files_only=bool(args.local_files_only))
    tokenizer = processor.tokenizer

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = getattr(tokenizer, "eos_token_id", None)

    if device.type == "cuda" and bool(args.bf16):
        torch_dtype = torch.bfloat16
    elif device.type == "cuda":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    attn_impl = resolve_attn_implementation(bool(args.use_flash_attn), device=device)
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        resolved_model_path,
        cache_dir=args.cache_dir,
        local_files_only=bool(args.local_files_only),
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        attn_implementation=attn_impl,
    )
    model.to(device)

    from peft import PeftModel

    vision_adapter_name = str(args.vision_adapter_name or "vision")
    llm_adapter_name = str(args.llm_adapter_name or "llm")

    vision_adapter_dir = resolve_adapter_dir(str(args.vision_adapter_path), vision_adapter_name)
    llm_adapter_dir = resolve_adapter_dir(str(args.llm_adapter_path), llm_adapter_name)

    model = PeftModel.from_pretrained(model, vision_adapter_dir, adapter_name=vision_adapter_name, is_trainable=False)
    model.load_adapter(llm_adapter_dir, adapter_name=llm_adapter_name, is_trainable=False)
    model.set_adapter(llm_adapter_name)

    model.eval()
    model_dtype = next(model.parameters()).dtype

    placeholder_id = tokenizer.convert_tokens_to_ids("<image>")
    if placeholder_id is None or placeholder_id == getattr(tokenizer, "unk_token_id", None):
        raise ValueError("Tokenizer does not recognize '<image>' as a single token.")

    depth_encoding = str(args.depth_encoding or "turbo").lower().strip()
    merge_pairs = bool(args.depth_merge_frame_pairs) and depth_encoding == "turbo+normals"

    lynx_model = LynXOnevisionWrapper(
        model,
        placeholder_token_id=int(placeholder_id),
        pool_video_tokens=bool(args.pool_video_tokens),
        frame_chunk_size=int(args.frame_chunk_size),
        vision_adapter_name=vision_adapter_name,
        llm_adapter_name=llm_adapter_name,
        detach_modality_tokens=True,
        merge_modality_frame_pairs=merge_pairs,
    ).to(device)

    questions = load_json(args.question_file)
    if not isinstance(questions, list):
        raise TypeError(f"Expected a list in {args.question_file}, got {type(questions)}")
    if args.max_samples is not None:
        questions = questions[: int(args.max_samples)]

    num_samples = len(questions)
    indices = list(range(rank, num_samples, world_size))
    eval_batch_size = max(1, int(args.eval_batch_size))

    amp_ctx = nullcontext()
    if device.type == "cuda":
        amp_ctx = torch.autocast(device_type="cuda", dtype=model_dtype)

    pairs_cache: Dict[str, List[Tuple[str, str]]] = {}
    selected_cache: Dict[str, List[Tuple[str, str]]] = {}
    pose_cache: Dict[Tuple[str, int], Any] = {}
    local_preds: List[Dict[str, Any]] = []

    split_events = 0

    def run_generation(batch_rows: List[Dict[str, Any]], batch_videos: List[torch.Tensor], batch_depths: List[torch.Tensor]) -> List[str]:
        pv_video = processor.video_processor(batch_videos, return_tensors="pt")["pixel_values_videos"].to(device=device, dtype=model_dtype)
        pv_depth = processor.video_processor(batch_depths, return_tensors="pt")["pixel_values_videos"].to(device=device, dtype=model_dtype)

        with amp_ctx:
            z_v, z_d = lynx_model._encode_streams(pixel_values_videos=pv_video, depth_pixel_values_videos=pv_depth)
            z = torch.cat([z_v, z_d], dim=1)

            core_model = getattr(lynx_model.base_model, "module", lynx_model.base_model)
            core_model_unwrapped = getattr(core_model, "module", core_model)
            embed = core_model_unwrapped.get_input_embeddings()

            inputs_list: List[torch.Tensor] = []
            seq_lens: List[int] = []
            for i, row in enumerate(batch_rows):
                prefix_ids = row["prefix_ids"].to(device)
                suffix_ids = row["suffix_ids"].to(device)
                prefix_emb = embed(prefix_ids).unsqueeze(0)
                suffix_emb = embed(suffix_ids).unsqueeze(0)
                inputs_embeds = torch.cat([prefix_emb, z[i : i + 1], suffix_emb], dim=1)
                inputs_list.append(inputs_embeds[0])
                seq_lens.append(int(inputs_embeds.shape[1]))

            max_len = max(seq_lens)
            batch_size = len(inputs_list)
            dim = int(inputs_list[0].shape[-1])
            inputs_embeds_padded = torch.zeros((batch_size, max_len, dim), device=device, dtype=model_dtype)
            attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
            for i, emb in enumerate(inputs_list):
                seq_len = int(emb.shape[0])
                start_pos = max_len - seq_len
                inputs_embeds_padded[i, start_pos:max_len, :] = emb
                attention_mask[i, start_pos:max_len] = 1

            gen_ids = core_model_unwrapped.generate(
                inputs_embeds=inputs_embeds_padded,
                attention_mask=attention_mask,
                max_new_tokens=int(args.max_new_tokens),
                min_new_tokens=max(0, int(args.min_new_tokens)),
                do_sample=False,
                num_beams=1,
                pad_token_id=int(tokenizer.pad_token_id),
                use_cache=True,
            )

        prompt_len = int(inputs_embeds_padded.shape[1])
        if gen_ids.ndim == 2 and gen_ids.shape[1] > prompt_len:
            gen_ids = gen_ids[:, prompt_len:]

        preds: List[str] = []
        for i in range(len(batch_rows)):
            seq = gen_ids[i].tolist() if gen_ids.ndim == 2 else []
            preds.append(tokenizer.decode(seq, skip_special_tokens=True).strip())
        return preds

    def generate_with_fallback(
        batch_rows: List[Dict[str, Any]],
        batch_videos: List[torch.Tensor],
        batch_depths: List[torch.Tensor],
    ) -> None:
        nonlocal split_events
        if not batch_rows:
            return
        try:
            preds = run_generation(batch_rows, batch_videos, batch_depths)
            for row, pred in zip(batch_rows, preds):
                local_preds.append({"question_id": row["question_id"], "video": row["video"], "text": pred})
        except Exception as exc:
            err_msg = _short_exc(exc)
            err = f"scanqa_eval_error: {type(exc).__name__}: {err_msg}"
            if len(batch_rows) > 1:
                if device.type == "cuda" and _is_cuda_oom(exc):
                    torch.cuda.empty_cache()
                if rank == 0 and split_events < 3:
                    print(
                        f"[ScanQA Eval] Batch size {len(batch_rows)} failed ({type(exc).__name__}: {err_msg}); "
                        "retrying with smaller micro-batches.",
                        file=sys.stderr,
                    )
                split_events += 1
                mid = len(batch_rows) // 2
                generate_with_fallback(batch_rows[:mid], batch_videos[:mid], batch_depths[:mid])
                generate_with_fallback(batch_rows[mid:], batch_videos[mid:], batch_depths[mid:])
                return

            local_preds.append({"question_id": batch_rows[0]["question_id"], "video": batch_rows[0]["video"], "text": "", "error": err})

    processed_global = 0
    pbar = None
    if rank == 0:
        pbar = tqdm(total=num_samples, desc="ScanQA", dynamic_ncols=True, leave=False)

    for start in range(0, len(indices), eval_batch_size):
        batch_indices = indices[start : start + eval_batch_size]

        batch_rows: List[Dict[str, Any]] = []
        batch_videos: List[torch.Tensor] = []
        batch_depths: List[torch.Tensor] = []

        for idx in batch_indices:
            item = questions[idx]
            question_id = str(item.get("question_id") or idx)
            video_name = str(item.get("video") or item.get("scene_id") or "")
            question_text = str(item.get("text") or "").strip()
            if not video_name or not question_text:
                local_preds.append({"question_id": question_id, "video": video_name, "text": "", "error": "invalid_sample"})
                continue

            messages = [{"role": "user", "content": f"<image>\n{question_text}"}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            try:
                prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
                pos = (prompt_ids == placeholder_id).nonzero()
                if pos.numel() != 1:
                    raise ValueError(f"Expected exactly 1 <image> token in prompt, got {pos.numel()}")
                ph = int(pos.item())
                prefix_ids = prompt_ids[:ph]
                suffix_ids = prompt_ids[ph + 1 :]

                rgb, depth = _load_video_rgb_depth_tensors(
                    video_name,
                    frames_root=args.frames_root,
                    color_subdir=args.color_subdir,
                    depth_subdir=args.depth_subdir,
                    pose_subdir=args.pose_subdir,
                    pose_matrix_type=args.pose_matrix_type,
                    num_frames=int(args.num_frames),
                    depth_encoding=depth_encoding,
                    depth_clip_min_mm=float(args.depth_clip_min_mm),
                    depth_clip_max_mm=float(args.depth_clip_max_mm),
                    depth_normals_frame=str(args.depth_normals_frame),
                    depth_intrinsics_filename=str(args.depth_intrinsics_filename),
                    depth_auto_scale_intrinsics=bool(args.depth_auto_scale_intrinsics),
                    pairs_cache=pairs_cache,
                    selected_cache=selected_cache,
                    pose_cache=pose_cache,
                    frame_sampling=str(args.frame_sampling),
                )

                batch_rows.append({"question_id": question_id, "video": video_name, "prefix_ids": prefix_ids, "suffix_ids": suffix_ids})
                batch_videos.append(rgb)
                batch_depths.append(depth)
            except Exception as exc:
                err = f"scanqa_eval_error: {type(exc).__name__}: {_short_exc(exc)}"
                local_preds.append({"question_id": question_id, "video": video_name, "text": "", "error": err})

        if not batch_rows:
            remaining = num_samples - processed_global
            step = min(world_size * len(batch_indices), remaining) if remaining > 0 else 0
            processed_global += step
            if pbar is not None:
                pbar.update(step)
            continue

        generate_with_fallback(batch_rows, batch_videos, batch_depths)

        remaining = num_samples - processed_global
        step = min(world_size * len(batch_indices), remaining) if remaining > 0 else 0
        processed_global += step
        if pbar is not None:
            pbar.update(step)

    if pbar is not None:
        pbar.close()

    pred_save = str(args.pred_save)
    pred_dir = os.path.dirname(pred_save)
    if pred_dir:
        os.makedirs(pred_dir, exist_ok=True)
    base, ext = os.path.splitext(pred_save)
    if not ext:
        base = pred_save
        ext = ".jsonl"
    rank_path = f"{base}_rank{rank}{ext}"
    with open(rank_path, "w") as f:
        for row in local_preds:
            f.write(json.dumps(row) + "\n")

    if world_size > 1:
        if rank != 0:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                try:
                    torch.distributed.destroy_process_group()
                except Exception:
                    pass
            return

        expected = [f"{base}_rank{r}{ext}" for r in range(world_size)]
        deadline = time.time() + 600
        missing = [p for p in expected if not os.path.exists(p)]
        while missing and time.time() < deadline:
            time.sleep(1)
            missing = [p for p in expected if not os.path.exists(p)]
        if missing:
            raise RuntimeError(f"Missing rank prediction files: {missing[:3]} (and {max(0, len(missing) - 3)} more)")

        merged: List[Dict[str, Any]] = []
        for path in expected:
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    merged.append(json.loads(line))
        gathered = [merged]
    else:
        gathered = [local_preds]

    if rank == 0:
        merged: List[Dict[str, Any]] = []
        for part in gathered:
            merged.extend(part)

        pred_by_qid: Dict[str, Dict[str, Any]] = {}
        for row in merged:
            qid = str(row.get("question_id"))
            if qid and qid not in pred_by_qid:
                pred_by_qid[qid] = row

        aligned_preds: List[Dict[str, Any]] = []
        errors: List[Dict[str, str]] = []
        for item in questions:
            qid = str(item.get("question_id") or "")
            meta = pred_by_qid.get(qid, {})
            pred = str(meta.get("text", "")).strip()
            aligned_preds.append({"question_id": qid, "text": pred})
            err = meta.get("error")
            if err:
                errors.append({"question_id": qid, "error": str(err)})

        with open(pred_save, "w") as f:
            for row in aligned_preds:
                f.write(json.dumps(row) + "\n")

        if errors:
            base, ext = os.path.splitext(pred_save)
            err_path = f"{base}_errors{ext or '.jsonl'}"
            with open(err_path, "w") as f:
                for row in errors:
                    f.write(json.dumps(row) + "\n")
            print(f"[ScanQA Eval] {len(errors)}/{len(aligned_preds)} samples failed. See: {err_path}", file=sys.stderr)

        metrics: Dict[str, float] = {}
        try:
            evaluator_path = os.path.join(PROJECT_ROOT, "tools", "3d", "scanqa", "scanqa_evaluator.py")
            evaluator = _load_module_from_file("scanqa_evaluator", evaluator_path)
            calc_scanqa_score = getattr(evaluator, "calc_scanqa_score", None)
            scorers = getattr(evaluator, "scorers", None)
            eval_tokenizer = getattr(evaluator, "tokenizer", None)
            if not callable(calc_scanqa_score) or scorers is None or eval_tokenizer is None:
                raise AttributeError("ScanQA evaluator missing required symbols (calc_scanqa_score/tokenizer/scorers).")

            gts_all = load_json(args.gt_file)
            if not isinstance(gts_all, list):
                raise TypeError(f"Expected a list in {args.gt_file}, got {type(gts_all)}")
            gt_by_qid = {str(row.get("question_id") or ""): row for row in gts_all}
            aligned_gts: List[Dict[str, Any]] = []
            missing_gts: List[str] = []
            for item in questions:
                qid = str(item.get("question_id") or "")
                gt_row = gt_by_qid.get(qid)
                if gt_row is None:
                    missing_gts.append(qid)
                    continue
                aligned_gts.append(gt_row)
            if missing_gts:
                raise KeyError(f"Missing {len(missing_gts)}/{len(questions)} GT rows (example qid={missing_gts[0]!r}).")
            if len(aligned_gts) != len(aligned_preds):
                raise ValueError(f"GT/pred length mismatch: {len(aligned_gts)} gts vs {len(aligned_preds)} preds.")

            scores_raw = calc_scanqa_score(aligned_preds, aligned_gts, eval_tokenizer, scorers)
            metrics = {str(k): float(v) for k, v in scores_raw.items()}
        except Exception as exc:
            print(f"[ScanQA Eval] Metrics unavailable: {exc}", file=sys.stderr)

        metrics_out = {"scanqa_total": float(len(aligned_preds))}
        for k, v in metrics.items():
            metrics_out[str(k)] = float(v)
        print(json.dumps(metrics_out))
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        try:
            torch.distributed.destroy_process_group()
        except Exception:
            pass


if __name__ == "__main__":
    main()
