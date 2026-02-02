import argparse
import json
import os
import sys
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm.auto import tqdm
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from libs.model.lynx_onevision import LynXOnevisionWrapper
from lynx_utils import (
    load_json,
    load_video_frames_and_audio,
    log_mel_to_pil_rgb,
    resolve_attn_implementation,
    resample_waveform,
    to_mono,
    waveform_to_imagebind_melspec_clips,
)


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


def _is_peft_adapter_dir(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "adapter_config.json"))


def _resolve_checkpoint_state_dict(path: str) -> Optional[str]:
    if os.path.isfile(path):
        return path
    if not os.path.isdir(path):
        return None
    for name in ("model.safetensors", "pytorch_model.safetensors", "pytorch_model.bin", "model.bin"):
        cand = os.path.join(path, name)
        if os.path.isfile(cand):
            return cand
    return None


def _strip_peft_prefix(name: str) -> str:
    for prefix in ("base_model.model.",):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def _find_llm_linear_modules(model: Any) -> List[str]:
    names: List[str] = []
    for raw_name, module in model.named_modules():
        name = _strip_peft_prefix(raw_name)
        if ".language_model.model.layers." not in f".{name}." and ".language_model.layers." not in f".{name}.":
            continue
        if not isinstance(module, torch.nn.Linear):
            continue
        leaf = name.split(".")[-1]
        if leaf not in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
            continue
        names.append(name)
    return sorted(set(names))


def _normalize_llm_lora_key(key: str) -> str:
    if key.startswith("base_model.base_model."):
        key = key[len("base_model.") :]
    key = key.replace("base_model.model.language_model.", "base_model.model.model.language_model.")
    key = key.replace(".language_model.model.layers.", ".language_model.layers.")
    return key


def _infer_lora_rank_from_safetensors(state_path: str, *, adapter_name: str) -> Optional[int]:
    try:
        from safetensors import safe_open
    except Exception:
        return None

    needle = f".lora_A.{adapter_name}.weight"
    try:
        with safe_open(state_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                if needle in k:
                    w = f.get_tensor(k)
                    if w.ndim == 2:
                        return int(w.shape[0])
                    return None
    except Exception:
        return None
    return None


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


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", "-1")))

    parser.add_argument("--model_name_or_path", type=str, default="llava-hf/llava-onevision-qwen2-0.5b-ov-hf")
    parser.add_argument("--vision_adapter_path", type=str, required=True)
    parser.add_argument("--llm_adapter_path", type=str, required=True)
    parser.add_argument("--vision_adapter_name", type=str, default="vision")
    parser.add_argument("--llm_adapter_name", type=str, default="llm")
    parser.add_argument("--llm_lora_r", type=int, default=None, help="Only used when --llm_adapter_path points to a full checkpoint (model.safetensors).")
    parser.add_argument("--llm_lora_alpha", type=int, default=None, help="Only used when --llm_adapter_path points to a full checkpoint (model.safetensors).")
    parser.add_argument("--llm_lora_dropout", type=float, default=0.05, help="Only used when --llm_adapter_path points to a full checkpoint (model.safetensors).")
    parser.add_argument("--llm_lora_bias", type=str, default="none", help="Only used when --llm_adapter_path points to a full checkpoint (model.safetensors).")

    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--local_files_only", action="store_true", default=True)
    parser.add_argument("--no_local_files_only", dest="local_files_only", action="store_false")

    parser.add_argument("--annotation_file", type=str, default="./data/video_instruction_tuning/music_avqa/music_avqa_updated_avqa-test.json")
    parser.add_argument("--video_root", type=str, default="./data/video_instruction_tuning/music_avqa")
    parser.add_argument("--video_mapping", type=str, default="./data/video_instruction_tuning/music_avqa/music_avqa_all_videos_mapping.json")
    parser.add_argument("--pred_save", type=str, default="./music_avqa_predictions.json")

    parser.add_argument("--num_frames", type=int, default=60)
    parser.add_argument("--frame_chunk_size", type=int, default=4)
    parser.add_argument("--pool_video_tokens", action="store_true", default=False)
    parser.add_argument("--no_pool_video_tokens", dest="pool_video_tokens", action="store_false")

    parser.add_argument("--audio_target_sr", type=int, default=16000)
    parser.add_argument("--audio_seconds", type=float, default=8.0)
    parser.add_argument("--audio_clip_duration_s", type=float, default=2.0)
    parser.add_argument("--audio_clip_stride_s", type=float, default=1.0)
    parser.add_argument("--mel_bins", type=int, default=128)
    parser.add_argument("--mel_target_length", type=int, default=204)
    parser.add_argument("--mel_mean", type=float, default=-4.268)
    parser.add_argument("--mel_std", type=float, default=9.138)

    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=None)

    parser.add_argument("--use_flash_attn", action="store_true", default=True)
    parser.add_argument("--no_use_flash_attn", dest="use_flash_attn", action="store_false")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no_bf16", dest="bf16", action="store_false")
    parser.add_argument("--tf32", action="store_true", default=True)
    parser.add_argument("--no_tf32", dest="tf32", action="store_false")
    args = parser.parse_args()

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

    if _is_peft_adapter_dir(llm_adapter_dir):
        model.load_adapter(llm_adapter_dir, adapter_name=llm_adapter_name, is_trainable=False)
        model.set_adapter(llm_adapter_name)
    else:
        state_path = _resolve_checkpoint_state_dict(str(args.llm_adapter_path))
        if state_path is None:
            raise ValueError(
                f"--llm_adapter_path must point to a PEFT adapter folder (adapter_config.json) or a model checkpoint "
                f"(model.safetensors), got: {args.llm_adapter_path}"
            )

        from peft import LoraConfig

        target_modules = _find_llm_linear_modules(model)
        if not target_modules:
            raise RuntimeError("No LLM linear layers found for LoRA injection.")

        inferred_r = _infer_lora_rank_from_safetensors(state_path, adapter_name=llm_adapter_name)
        llm_r = int(args.llm_lora_r) if args.llm_lora_r is not None else (int(inferred_r) if inferred_r is not None else 64)
        llm_alpha = int(args.llm_lora_alpha) if args.llm_lora_alpha is not None else (2 * llm_r)
        llm_cfg = LoraConfig(
            r=llm_r,
            lora_alpha=llm_alpha,
            target_modules=target_modules,
            lora_dropout=float(args.llm_lora_dropout),
            bias=str(args.llm_lora_bias),
            task_type="CAUSAL_LM",
        )
        model.add_adapter(llm_adapter_name, llm_cfg)
        model.set_adapter(llm_adapter_name)

        llm_state: Dict[str, torch.Tensor] = {}
        if state_path.endswith(".safetensors"):
            from safetensors import safe_open

            with safe_open(state_path, framework="pt", device="cpu") as f:
                for k in f.keys():
                    if "lora_" not in k or f".{llm_adapter_name}." not in k:
                        continue
                    llm_state[_normalize_llm_lora_key(k)] = f.get_tensor(k)
        else:
            payload = torch.load(state_path, map_location="cpu")
            if not isinstance(payload, dict):
                raise TypeError(f"Unsupported checkpoint format at {state_path} (expected a state_dict dict).")
            for k, v in payload.items():
                if not isinstance(k, str) or not isinstance(v, torch.Tensor):
                    continue
                if "lora_" not in k or f".{llm_adapter_name}." not in k:
                    continue
                llm_state[_normalize_llm_lora_key(k)] = v

        model.load_state_dict(llm_state, strict=False)
        if rank == 0:
            infer_tag = "inferred" if inferred_r is not None else "could not infer rank"
            print(f"[Music-AVQA Eval] Loaded LLM LoRA from checkpoint={state_path} with llm_r={llm_r} ({infer_tag}).")

    model.eval()
    model_dtype = next(model.parameters()).dtype

    placeholder_id = tokenizer.convert_tokens_to_ids("<image>")
    if placeholder_id is None or placeholder_id == getattr(tokenizer, "unk_token_id", None):
        raise ValueError("Tokenizer does not recognize '<image>' as a single token.")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = getattr(tokenizer, "eos_token_id", None)

    lynx_model = LynXOnevisionWrapper(
        model,
        placeholder_token_id=int(placeholder_id),
        pool_video_tokens=bool(args.pool_video_tokens),
        frame_chunk_size=int(args.frame_chunk_size),
        vision_adapter_name=vision_adapter_name,
        llm_adapter_name=llm_adapter_name,
    )

    ann = load_json(args.annotation_file)
    if not isinstance(ann, list):
        raise TypeError(f"Expected a list in {args.annotation_file}, got {type(ann)}")
    if args.max_samples is not None:
        ann = ann[: int(args.max_samples)]

    mapping: Optional[Dict[str, str]] = None
    if args.video_mapping:
        raw = load_json(args.video_mapping)
        if isinstance(raw, dict):
            mapping = {str(k): str(v) for k, v in raw.items()}

    num_samples = len(ann)
    indices = list(range(rank, num_samples, world_size))

    local_preds: List[Dict[str, Any]] = []
    processed_global = 0
    pbar = None
    if rank == 0:
        pbar = tqdm(total=num_samples, desc="Music-AVQA", dynamic_ncols=True, leave=False)

    eval_batch_size = int(args.eval_batch_size) if args.eval_batch_size else 1
    eval_batch_size = max(1, eval_batch_size)

    amp_ctx = nullcontext()
    if device.type == "cuda" and bool(args.bf16):
        amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    for start in range(0, len(indices), eval_batch_size):
        batch_indices = indices[start : start + eval_batch_size]

        batch_rows: List[Dict[str, Any]] = []
        batch_videos: List[torch.Tensor] = []
        batch_audios: List[List[Any]] = []

        for idx in batch_indices:
            item = ann[idx]
            video_id = str(item.get("video_id", ""))
            question = str(item.get("question_content", "")).strip()
            answer = str(item.get("anser", "")).strip()
            if not video_id or not question:
                local_preds.append({"index": idx, "video_id": video_id, "pred": "", "answer": answer, "error": "invalid_sample"})
                continue

            rel = mapping.get(video_id) if isinstance(mapping, dict) else None
            filename = rel if rel else f"{video_id}.mp4"
            video_path = os.path.join(str(args.video_root), filename)

            messages = [{"role": "user", "content": f"<image>\n{question}"}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            try:
                if not os.path.exists(video_path):
                    raise FileNotFoundError(video_path)

                vframes, aframes, info = load_video_frames_and_audio(
                    video_path,
                    num_frames=int(args.num_frames),
                    video_backend="torchvision",
                    decode_audio=True,
                )

                waveform, orig_sr = _coerce_audio_waveform_and_sr(
                    aframes,
                    info,
                    fallback_sr=int(args.audio_target_sr),
                    target_seconds=float(args.audio_seconds),
                )
                waveform = resample_waveform(waveform, orig_sr=orig_sr, target_sr=int(args.audio_target_sr))

                audio_clips = waveform_to_imagebind_melspec_clips(
                    waveform,
                    sample_rate=int(args.audio_target_sr),
                    num_clips=int(vframes.shape[0]),
                    clip_duration_s=float(args.audio_clip_duration_s),
                    clip_stride_s=float(args.audio_clip_stride_s),
                    num_mel_bins=int(args.mel_bins),
                    target_length=int(args.mel_target_length),
                    mean=float(args.mel_mean),
                    std=float(args.mel_std),
                )  # (F, 1, mel_bins, mel_target_length)
                audio_frames = [log_mel_to_pil_rgb(audio_clips[i, 0]) for i in range(audio_clips.shape[0])]

                prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
                pos = (prompt_ids == placeholder_id).nonzero()
                if pos.numel() != 1:
                    raise ValueError(f"Expected exactly 1 <image> token in prompt, got {pos.numel()}")
                ph = int(pos.item())
                prefix_ids = prompt_ids[:ph]
                suffix_ids = prompt_ids[ph + 1 :]

                batch_rows.append(
                    {
                        "index": idx,
                        "video_id": video_id,
                        "answer": answer,
                        "prefix_ids": prefix_ids,
                        "suffix_ids": suffix_ids,
                    }
                )
                batch_videos.append(vframes)
                batch_audios.append(audio_frames)
            except Exception as exc:
                msg = str(exc)
                if len(msg) > 300:
                    msg = msg[:300] + "..."
                err = f"music_avqa_eval_error: {type(exc).__name__}: {msg}"
                local_preds.append({"index": idx, "video_id": video_id, "pred": "", "answer": answer, "error": err})

        if batch_rows:
            try:
                pv_video = processor.video_processor(batch_videos, return_tensors="pt")["pixel_values_videos"].to(device=device, dtype=model_dtype)
                pv_audio = processor.video_processor(batch_audios, return_tensors="pt")["pixel_values_videos"].to(device=device, dtype=model_dtype)

                with amp_ctx:
                    z_v, z_a = lynx_model._encode_streams(pixel_values_videos=pv_video, audio_pixel_values_videos=pv_audio)
                    z = torch.cat([z_v, z_a], dim=1)

                    core_model = getattr(lynx_model.base_model, "module", lynx_model.base_model)
                    embed = core_model.get_input_embeddings()

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

                    gen_ids = core_model.generate(
                        inputs_embeds=inputs_embeds_padded,
                        attention_mask=attention_mask,
                        max_new_tokens=int(args.max_new_tokens),
                        do_sample=False,
                        num_beams=1,
                        pad_token_id=int(tokenizer.pad_token_id),
                        use_cache=True,
                    )

                    prompt_len = int(inputs_embeds_padded.shape[1])
                    if gen_ids.ndim == 2 and gen_ids.shape[1] > prompt_len:
                        gen_ids = gen_ids[:, prompt_len:]

                    for i, row in enumerate(batch_rows):
                        seq = gen_ids[i].tolist() if gen_ids.ndim == 2 else []
                        pred = tokenizer.decode(seq, skip_special_tokens=True).strip()
                        local_preds.append(
                            {
                                "index": row["index"],
                                "video_id": row["video_id"],
                                "pred": pred,
                                "answer": row["answer"],
                            }
                        )
            except Exception as exc:
                msg = str(exc)
                if len(msg) > 300:
                    msg = msg[:300] + "..."
                err = f"music_avqa_eval_error: {type(exc).__name__}: {msg}"
                for row in batch_rows:
                    local_preds.append(
                        {
                            "index": row["index"],
                            "video_id": row["video_id"],
                            "pred": "",
                            "answer": row["answer"],
                            "error": err,
                        }
                    )

        remaining = num_samples - processed_global
        step = min(world_size * len(batch_indices), remaining) if remaining > 0 else 0
        processed_global += step
        if pbar is not None:
            pbar.update(step)

    if pbar is not None:
        pbar.close()

    gathered: List[List[Dict[str, Any]]] = []
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if rank == 0:
            gathered = [None for _ in range(world_size)]  # type: ignore[list-item]
            torch.distributed.gather_object(local_preds, gathered, dst=0)
        else:
            torch.distributed.gather_object(local_preds, None, dst=0)
    else:
        gathered = [local_preds]

    if rank == 0:
        merged: List[Dict[str, Any]] = []
        for part in gathered:
            merged.extend(part)
        merged.sort(key=lambda x: int(x.get("index", 0)))

        correct = 0
        for ele in merged:
            pred = str(ele.get("pred", "")).lower()
            ans = str(ele.get("answer", "")).lower()
            if ans and ans in pred:
                correct += 1
        total = len(merged)
        acc = float(correct) / float(total) if total > 0 else 0.0
        print(json.dumps({"music_avqa_acc": acc, "music_avqa_correct": correct, "music_avqa_total": total}))

        pred_save = str(args.pred_save)
        pred_dir = os.path.dirname(pred_save)
        if pred_dir:
            os.makedirs(pred_dir, exist_ok=True)
        with open(pred_save, "w") as f:
            json.dump(merged, f)

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


if __name__ == "__main__":
    main()
