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
from lynx_utils import load_json, load_video_frames_and_audio, resolve_attn_implementation


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


def _coerce_messages(conversations: Any) -> List[Dict[str, str]]:
    if not isinstance(conversations, list):
        return []
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


def _extract_prompt_and_answer(tokenizer: Any, item: Dict[str, Any]) -> Tuple[str, str]:
    conversations = item.get("conversations")
    messages = _coerce_messages(conversations)
    if len(messages) < 2:
        raise ValueError("Expected at least one user and one assistant turn in conversations.")
    if messages[-1].get("role") != "assistant":
        raise ValueError("Expected the last conversation turn to be assistant with the ground-truth answer.")

    answer = str(messages[-1].get("content") or "").strip()
    prompt_msgs = messages[:-1]
    prompt = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
    return str(prompt), answer


def _extract_answer_only(item: Dict[str, Any]) -> str:
    conversations = item.get("conversations")
    messages = _coerce_messages(conversations)
    if not messages:
        return ""
    last = messages[-1]
    if last.get("role") != "assistant":
        return ""
    return str(last.get("content") or "").strip()


def _maybe_set_tokenizer_padding(tokenizer: Any) -> None:
    if getattr(tokenizer, "pad_token_id", None) is not None:
        return
    eos = getattr(tokenizer, "eos_token_id", None)
    bos = getattr(tokenizer, "bos_token_id", None)
    if isinstance(eos, int):
        tokenizer.pad_token_id = eos
    elif isinstance(bos, int):
        tokenizer.pad_token_id = bos
    else:
        tokenizer.pad_token_id = 0


@torch.no_grad()
def _greedy_decode(
    *,
    model: Any,
    embed: torch.nn.Module,
    tokenizer: Any,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    max_new_tokens: int,
) -> str:
    device = inputs_embeds.device
    past_key_values = None
    generated: List[int] = []
    eos_id = getattr(tokenizer, "eos_token_id", None)

    for _ in range(int(max_new_tokens)):
        out = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            past_key_values=past_key_values,
            return_dict=True,
        )
        past_key_values = out.past_key_values
        logits = out.logits[:, -1, :]
        next_id = int(torch.argmax(logits, dim=-1).item())
        generated.append(next_id)

        if isinstance(eos_id, int) and next_id == eos_id:
            break

        next_embed = embed(torch.tensor([next_id], device=device)).unsqueeze(0)  # (1, 1, D)
        inputs_embeds = next_embed
        attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=device, dtype=attention_mask.dtype)], dim=1)
        last_pos = int(position_ids[0, -1].item()) if position_ids.numel() else -1
        position_ids = torch.cat([position_ids, torch.tensor([[last_pos + 1]], device=device, dtype=position_ids.dtype)], dim=1)

    text = tokenizer.decode(generated, skip_special_tokens=True)
    return str(text).strip()


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="llava-hf/llava-onevision-qwen2-0.5b-ov-hf")
    parser.add_argument("--cache_dir", type=str, default="/mnt/hdd1/")
    parser.add_argument("--local_files_only", action="store_true", default=True)
    parser.add_argument("--no_local_files_only", dest="local_files_only", action="store_false")

    parser.add_argument("--vision_adapter_path", type=str, default=None)
    parser.add_argument("--llm_adapter_path", type=str, default=None)
    parser.add_argument("--vision_adapter_name", type=str, default="vision")
    parser.add_argument("--llm_adapter_name", type=str, default="llm")

    parser.add_argument("--annotation_file", type=str, default="./data/video_instruction_tuning/egoexo/proficiency_demonstrator_train_instruct.json")
    parser.add_argument("--video_root", type=str, default="./data/video_instruction_tuning/egoexo")
    parser.add_argument("--video_mapping", type=str, default="./data/video_instruction_tuning/egoexo/from_take_id_to_video.json")
    parser.add_argument("--pred_save", type=str, default="./egoexo_predictions.json")

    parser.add_argument("--num_frames", type=int, default=20)
    parser.add_argument("--frame_chunk_size", type=int, default=4)
    parser.add_argument("--pool_video_tokens", action="store_true", default=False)
    parser.add_argument("--no_pool_video_tokens", dest="pool_video_tokens", action="store_false")
    parser.add_argument("--max_samples", type=int, default=None)

    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--dry_run", action="store_true", default=False, help="Skip model loading; only validate paths and write an empty-caption prediction file.")
    parser.add_argument("--use_flash_attn", action="store_true", default=True)
    parser.add_argument("--no_use_flash_attn", dest="use_flash_attn", action="store_false")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no_bf16", dest="bf16", action="store_false")
    parser.add_argument("--tf32", action="store_true", default=True)
    parser.add_argument("--no_tf32", dest="tf32", action="store_false")
    args = parser.parse_args()

    ann = load_json(args.annotation_file)
    if not isinstance(ann, list):
        raise TypeError(f"Expected a list in {args.annotation_file}, got {type(ann)}")
    if args.max_samples is not None:
        ann = ann[: int(args.max_samples)]

    take_to_video = load_json(args.video_mapping) if args.video_mapping else None
    if take_to_video is not None and not isinstance(take_to_video, dict):
        raise TypeError(f"Expected a dict in {args.video_mapping}, got {type(take_to_video)}")

    if bool(args.dry_run):
        preds: List[Dict[str, Any]] = []
        for item in ann:
            if not isinstance(item, dict):
                continue
            take_id = item.get("video") or item.get("video_id") or item.get("video_name") or item.get("id")
            if not isinstance(take_id, str) or not take_id:
                continue
            answer = _extract_answer_only(item)
            rel = take_to_video.get(take_id) if isinstance(take_to_video, dict) else None
            if not isinstance(rel, str) or not rel:
                preds.append({"image_id": take_id, "caption": "", "answer": answer, "error": "missing_take_to_video_mapping"})
                continue
            video_path = os.path.join(str(args.video_root), rel)
            if not os.path.exists(video_path):
                preds.append({"image_id": take_id, "caption": "", "answer": answer, "error": f"missing_video: {video_path}"})
            else:
                preds.append({"image_id": take_id, "caption": "", "answer": answer, "error": "dry_run"})

        os.makedirs(os.path.dirname(args.pred_save) or ".", exist_ok=True)
        with open(args.pred_save, "w") as f:
            json.dump(preds, f)
        print(f"[EgoExo Eval] dry_run saved={args.pred_save} samples={len(preds)}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    _maybe_set_tokenizer_padding(tokenizer)

    torch_dtype = torch.float32
    if device.type == "cuda":
        torch_dtype = torch.bfloat16 if bool(args.bf16) else torch.float16

    attn_impl = resolve_attn_implementation(bool(args.use_flash_attn), device=device)
    device_map = {"": str(device)} if device.type == "cuda" else None
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        resolved_model_path,
        cache_dir=args.cache_dir,
        local_files_only=bool(args.local_files_only),
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map=device_map,
        attn_implementation=attn_impl,
    )
    if device_map is None:
        model.to(device)

    vision_adapter_name: Optional[str] = None
    llm_adapter_name: Optional[str] = None
    if args.vision_adapter_path:
        from peft import PeftModel

        vision_adapter_name = str(args.vision_adapter_name or "vision")
        model = PeftModel.from_pretrained(
            model,
            args.vision_adapter_path,
            adapter_name=vision_adapter_name,
            is_trainable=False,
        )

    if args.llm_adapter_path:
        from peft import PeftModel

        llm_adapter_name = str(args.llm_adapter_name or "llm")
        model = PeftModel.from_pretrained(
            model,
            args.llm_adapter_path,
            adapter_name=llm_adapter_name,
            is_trainable=False,
        )

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

    amp_ctx = nullcontext()
    if device.type == "cuda":
        amp_ctx = torch.autocast(device_type="cuda", dtype=model_dtype)

    preds: List[Dict[str, Any]] = []
    pbar = tqdm(ann, desc="EgoExo", dynamic_ncols=True)
    for idx, item in enumerate(pbar):
        if not isinstance(item, dict):
            continue

        take_id = item.get("video") or item.get("video_id") or item.get("video_name") or item.get("id")
        if not isinstance(take_id, str) or not take_id:
            continue

        rel = take_to_video.get(take_id) if isinstance(take_to_video, dict) else None
        if not isinstance(rel, str) or not rel:
            preds.append({"image_id": take_id, "caption": "", "answer": "", "error": "missing_take_to_video_mapping"})
            continue
        video_path = os.path.join(str(args.video_root), rel)
        if not os.path.exists(video_path):
            preds.append({"image_id": take_id, "caption": "", "answer": "", "error": f"missing_video: {video_path}"})
            continue

        try:
            prompt, answer = _extract_prompt_and_answer(tokenizer, item)
        except Exception as exc:
            preds.append({"image_id": take_id, "caption": "", "answer": "", "error": f"bad_annotation: {type(exc).__name__}: {exc}"})
            continue

        try:
            prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
            pos = (prompt_ids == placeholder_id).nonzero()
            if pos.numel() != 1:
                raise ValueError(f"Expected exactly 1 <image> token in prompt, got {pos.numel()}")
            ph = int(pos.item())
            prefix_ids = prompt_ids[:ph]
            suffix_ids = prompt_ids[ph + 1 :]

            vframes, _, _ = load_video_frames_and_audio(
                video_path,
                num_frames=int(args.num_frames),
                video_backend="torchvision",
                decode_audio=False,
            )
            pv = processor.video_processor([vframes], return_tensors="pt")["pixel_values_videos"].to(device=device, dtype=model_dtype)

            with amp_ctx:
                z_v, z_a = lynx_model._encode_streams(pixel_values_videos=pv, fast_pixel_values_videos=pv)
                z = torch.cat([z_v, z_a], dim=1)

                core_model = getattr(lynx_model.base_model, "module", lynx_model.base_model)
                peft_model = getattr(core_model, "module", core_model)
                if llm_adapter_name is not None:
                    setter = getattr(peft_model, "set_adapter", None)
                    if callable(setter):
                        peft_model.set_adapter(llm_adapter_name)

                embed = core_model.get_input_embeddings()
                prefix_emb = embed(prefix_ids.to(device)).unsqueeze(0)
                suffix_emb = embed(suffix_ids.to(device)).unsqueeze(0)
                full_embeds = torch.cat([prefix_emb, z, suffix_emb], dim=1)

                attention_mask = torch.ones((1, full_embeds.shape[1]), device=device, dtype=torch.long)
                position_ids = torch.arange(full_embeds.shape[1], device=device).unsqueeze(0)

                caption = _greedy_decode(
                    model=core_model,
                    embed=embed,
                    tokenizer=tokenizer,
                    inputs_embeds=full_embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    max_new_tokens=int(args.max_new_tokens),
                )

            preds.append({"image_id": take_id, "caption": caption, "answer": answer})
        except Exception as exc:
            msg = str(exc)
            if len(msg) > 300:
                msg = msg[:300] + "..."
            preds.append({"image_id": take_id, "caption": "", "answer": answer, "error": f"infer_error: {type(exc).__name__}: {msg}"})

    os.makedirs(os.path.dirname(args.pred_save) or ".", exist_ok=True)
    with open(args.pred_save, "w") as f:
        json.dump(preds, f)

    print(f"[EgoExo Eval] saved={args.pred_save} samples={len(preds)}")


if __name__ == "__main__":
    main()
