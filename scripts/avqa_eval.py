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


def build_avqa_question(item: Dict[str, Any]) -> Dict[str, str]:
    start_prompt = (
        "Select the best answer to the following multiple-choice question based on the video and the subtitles. "
        "Respond with only the letter (A, B, C, or D) of the correct option.\n"
    )
    end_prompt = "Answer with the option's letter from the given choices directly."
    idx_to_letter = {0: "A", 1: "B", 2: "C", 3: "D"}

    question = start_prompt + str(item["question_text"]).strip() + "\n"
    options = item["multi_choice"]
    question += f"A.{options[0]}\nB.{options[1]}\nC.{options[2]}\nD.{options[3]}\n"
    question += end_prompt
    answer = idx_to_letter[int(item["answer"])]
    return {"question": question, "answer": answer}


def _get_choice_token_ids(tokenizer: Any, device: torch.device) -> torch.Tensor:
    ids: List[int] = []
    for letter in ("A", "B", "C", "D"):
        for variant in (letter, f" {letter}"):
            enc = tokenizer.encode(variant, add_special_tokens=False)
            if len(enc) != 1:
                raise ValueError(f"Expected {variant!r} to be a single token, got ids={enc}")
            ids.append(int(enc[0]))
    return torch.tensor(ids, dtype=torch.long, device=device)


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

    parser.add_argument("--cache_dir", type=str, default='/mnt/hdd1')
    parser.add_argument("--local_files_only", action="store_true", default=True)
    parser.add_argument("--no_local_files_only", dest="local_files_only", action="store_false")

    parser.add_argument("--annotation_file", type=str, default="./data/video_instruction_tuning/avqa/val_qa.json")
    parser.add_argument("--video_root", type=str, default="./data/video_instruction_tuning/avqa/videos")
    parser.add_argument("--video_mapping", type=str, default="./data/video_instruction_tuning/avqa/avqa_from_vid_to_video_name.json")
    parser.add_argument("--pred_save", type=str, default="./avqa_predictions.json")

    parser.add_argument("--num_frames", type=int, default=20)
    parser.add_argument("--frame_chunk_size", type=int, default=4)
    parser.add_argument("--pool_video_tokens", action="store_true", default=False)
    parser.add_argument("--no_pool_video_tokens", dest="pool_video_tokens", action="store_false")

    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=1)

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
        # Fallback: the Stage-3 run may have saved a full model state_dict (e.g., Deepspeed checkpoint)
        # instead of a PEFT adapter folder. In that case, reconstruct the LLM LoRA modules and load
        # only the LoRA weights from the checkpoint.
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

        missing, unexpected = model.load_state_dict(llm_state, strict=False)
        expected_llm_keys = {n for n, _ in model.named_parameters() if ("lora_" in n and f".{llm_adapter_name}." in n)}
        provided_llm_keys = set(llm_state.keys())
        missing_llm_keys = sorted(expected_llm_keys - provided_llm_keys)
        unexpected_llm_keys = sorted(provided_llm_keys - expected_llm_keys)

        if rank == 0:
            infer_tag = "inferred" if inferred_r is not None else "could not infer rank"
            print(f"[AVQA Eval] Loaded LLM LoRA from checkpoint={state_path} with llm_r={llm_r} ({infer_tag}).")
            print(
                f"[AVQA Eval] LLM LoRA key check: expected={len(expected_llm_keys)} provided={len(provided_llm_keys)} "
                f"missing={len(missing_llm_keys)} unexpected={len(unexpected_llm_keys)}"
            )
            if missing_llm_keys:
                print("[AVQA Eval] Missing LLM LoRA keys (first 5):")
                for k in missing_llm_keys[:5]:
                    print(f"  {k}")
            if unexpected_llm_keys:
                print("[AVQA Eval] Unexpected LLM LoRA keys (first 5):")
                for k in unexpected_llm_keys[:5]:
                    print(f"  {k}")
            if unexpected:
                # This indicates some provided keys could not be matched to parameters on the model.
                print(f"[AVQA Eval] Warning: load_state_dict reported unexpected keys (strict=False): {len(unexpected)}")

    model.eval()
    model_dtype = next(model.parameters()).dtype

    placeholder_id = tokenizer.convert_tokens_to_ids("<image>")
    if placeholder_id is None or placeholder_id == getattr(tokenizer, "unk_token_id", None):
        raise ValueError("Tokenizer does not recognize '<image>' as a single token.")

    lynx_model = LynXOnevisionWrapper(
        model,
        placeholder_token_id=int(placeholder_id),
        pool_video_tokens=bool(args.pool_video_tokens),
        frame_chunk_size=int(args.frame_chunk_size),
        vision_adapter_name=vision_adapter_name,
        llm_adapter_name=llm_adapter_name,
        detach_modality_tokens=True,
    ).to(device)

    ann = load_json(args.annotation_file)
    if not isinstance(ann, list):
        raise TypeError(f"Expected a list in {args.annotation_file}, got {type(ann)}")
    if args.max_samples is not None:
        ann = ann[: int(args.max_samples)]

    video_mapping = load_json(args.video_mapping) if args.video_mapping else None
    num_samples = len(ann)

    choice_ids = _get_choice_token_ids(tokenizer, device)

    local_preds: List[Dict[str, Any]] = []
    eval_batch_size = max(1, int(args.eval_batch_size))
    indices = list(range(rank, num_samples, world_size))

    amp_ctx = nullcontext()
    if device.type == "cuda":
        amp_ctx = torch.autocast(device_type="cuda", dtype=model_dtype)

    processed_global = 0
    pbar = None
    if rank == 0:
        pbar = tqdm(total=num_samples, desc="AVQA", dynamic_ncols=True, leave=False)
    for start in range(0, len(indices), eval_batch_size):
        batch_indices = indices[start : start + eval_batch_size]

        batch_rows: List[Dict[str, Any]] = []
        batch_videos: List[torch.Tensor] = []
        batch_audios: List[List[Any]] = []

        for idx in batch_indices:
            item = ann[idx]
            vid = item.get("video_name")
            if not isinstance(vid, str) or not vid:
                continue

            filename = video_mapping.get(vid, f"{vid}.mp4") if isinstance(video_mapping, dict) else f"{vid}.mp4"
            video_path = os.path.join(args.video_root, filename)
            if not os.path.exists(video_path):
                continue

            qa = build_avqa_question(item)
            messages = [{"role": "user", "content": f"<image>\n{qa['question']}"}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            try:
                vframes, aframes, info = load_video_frames_and_audio(
                    video_path,
                    num_frames=int(args.num_frames),
                    video_backend="torchvision",
                    decode_audio=True,
                )

                waveform, orig_sr = _coerce_audio_waveform_and_sr(
                    aframes,
                    info,
                    fallback_sr=16000,
                    target_seconds=8.0,
                )
                waveform = resample_waveform(waveform, orig_sr=orig_sr, target_sr=16000)

                audio_clips = waveform_to_imagebind_melspec_clips(
                    waveform,
                    sample_rate=16000,
                    num_clips=int(vframes.shape[0]),
                    clip_duration_s=2.0,
                    clip_stride_s=0.5,
                    num_mel_bins=128,
                    target_length=204,
                    mean=-4.268,
                    std=9.138,
                )  # (F, 1, 128, 204)
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
                        "question_id": item.get("id", idx),
                        "image_id": vid,
                        "answer": qa["answer"],
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
                err = f"avqa_eval_error: {type(exc).__name__}: {msg}"
                local_preds.append(
                    {
                        "question_id": item.get("id", idx),
                        "image_id": vid,
                        "caption": "",
                        "answer": qa["answer"],
                        "error": err,
                    }
                )

        if not batch_rows:
            remaining = num_samples - processed_global
            step = min(world_size * len(batch_indices), remaining) if remaining > 0 else 0
            processed_global += step
            if pbar is not None:
                pbar.update(step)
            continue

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
                    prefix_ids = row["prefix_ids"]
                    suffix_ids = row["suffix_ids"]
                    prefix_emb = embed(prefix_ids.to(device)).unsqueeze(0)
                    suffix_emb = embed(suffix_ids.to(device)).unsqueeze(0)
                    inputs_embeds = torch.cat([prefix_emb, z[i : i + 1], suffix_emb], dim=1)
                    inputs_list.append(inputs_embeds[0])
                    seq_lens.append(int(inputs_embeds.shape[1]))

                max_len = max(seq_lens)
                batch_size = len(inputs_list)
                dim = int(inputs_list[0].shape[-1])
                inputs_embeds_padded = torch.zeros((batch_size, max_len, dim), device=device, dtype=model_dtype)
                attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
                position_ids = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
                for i, emb in enumerate(inputs_list):
                    seq_len = int(emb.shape[0])
                    start_pos = max_len - seq_len
                    inputs_embeds_padded[i, start_pos:max_len, :] = emb
                    attention_mask[i, start_pos:max_len] = 1
                    position_ids[i, start_pos:max_len] = torch.arange(seq_len, device=device)

                out = core_model(
                    inputs_embeds=inputs_embeds_padded,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                    num_logits_to_keep=1,
                    return_dict=True,
                )
            logits_last = out.logits[:, -1]
            logits = logits_last.index_select(1, choice_ids)
            best = torch.argmax(logits, dim=1)
            pred_ids = choice_ids[best]

            for i, row in enumerate(batch_rows):
                pred = tokenizer.decode([int(pred_ids[i].item())]).strip()
                local_preds.append(
                    {
                        "question_id": row["question_id"],
                        "image_id": row["image_id"],
                        "caption": pred,
                        "answer": row["answer"],
                    }
                )
        except Exception as exc:
            msg = str(exc)
            if len(msg) > 300:
                msg = msg[:300] + "..."
            err = f"avqa_eval_error: {type(exc).__name__}: {msg}"
            for row in batch_rows:
                local_preds.append(
                    {
                        "question_id": row["question_id"],
                        "image_id": row["image_id"],
                        "caption": "",
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
        merged.sort(key=lambda x: int(x.get("question_id", 0)))

        correct = sum(1 for ele in merged if ele.get("answer", "") in ele.get("caption", ""))
        total = len(merged)
        acc = float(correct) / float(total) if total > 0 else 0.0
        metrics = {"avqa_acc": acc, "avqa_correct": correct, "avqa_total": total}
        print(json.dumps(metrics))

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
