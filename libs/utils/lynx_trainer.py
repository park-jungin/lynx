import gc
import json
import os
import math
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm
from transformers import Trainer

from libs.model.attention_distill import get_siglip_encoder_layers, get_vision_tower, llava_onevision_spatial_pool_query_indices
from libs.utils.lynx_3d import (
    is_rank_zero as _is_rank_zero,
    load_module_from_file as _load_module_from_file,
    load_video_rgb_depth_tensors as _load_video_rgb_depth_tensors,
    sanitize_metric_key as _sanitize_metric_key,
    warn_optional_eval_deps as _warn_optional_eval_deps,
)
from lynx_utils import (
    encode_llava_onevision_video_tokens,
    load_json,
    load_video_frames_and_audio,
    log_mel_to_pil_rgb,
    resample_waveform,
    to_mono,
    waveform_to_imagebind_melspec_clips,
)


def temporal_mean_pool_video_tokens(tokens: torch.Tensor, *, frames: int) -> torch.Tensor:
    """
    Pool per-frame visual tokens to keep the sequence short.

    `encode_llava_onevision_video_tokens` returns (B, frames * seq_len + 1, D) where the last token is `image_newline`.
    We average tokens across frames at each patch position and keep the newline token.
    """

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
    return torch.cat([pooled, newline], dim=1)


def _linear_cka(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    xty = x.T @ y
    xtx = x.T @ x
    yty = y.T @ y
    numerator = (xty**2).sum()
    denom = torch.sqrt((xtx**2).sum().clamp_min(eps) * (yty**2).sum().clamp_min(eps))
    return numerator / denom.clamp_min(eps)


def build_avqa_question(item: Dict[str, Any]) -> Dict[str, str]:
    start_prompt = (
        "Select the best answer to the following multiple-choice question based on the video and the subtitles. "
        "Respond with only the letter (A, B, C, or D) of the correct option.\n"
    )
    end_prompt = "Answer with the option's letter from the given choices directly."
    idx_to_letter = {0: "A", 1: "B", 2: "C", 3: "D"}

    question = start_prompt + item["question_text"].strip() + "\n"
    options = item["multi_choice"]
    question += f"A.{options[0]}\nB.{options[1]}\nC.{options[2]}\nD.{options[3]}\n"
    question += end_prompt
    # question += "Hint: Please answer the question and provide the final answer at the end.\nFinal answer:"
    answer = idx_to_letter[int(item["answer"])]
    return {"question": question, "answer": answer}


@contextmanager
def _maybe_disable_adapter(model: Any):
    peft_model = getattr(model, "module", model)
    disable = getattr(peft_model, "disable_adapter", None)
    if callable(disable):
        with peft_model.disable_adapter():
            yield
        return
    yield


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


class LynXAVQATrainer(Trainer):
    """
    Trainer that:
    - Optimizes LynX alignment losses (stat/attn[/traj]) via custom `compute_loss`.
    - Optionally evaluates AVQA during training.
    """

    def __init__(
        self,
        *args: Any,
        processor: Any,
        avqa_args: Any,
        loss_args: Any,
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
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._processor = processor
        self._tokenizer = processor.tokenizer

        self._avqa_args = avqa_args
        self._loss_args = loss_args

        self._audio_target_sr = int(audio_target_sr)
        self._audio_seconds = float(audio_seconds)
        self._mel_n_mels = int(mel_n_mels)
        self._mel_n_fft = int(mel_n_fft)
        self._mel_hop_length = int(mel_hop_length)
        self._mel_win_length = int(mel_win_length)

        self._ref_token_mean = ref_token_mean
        self._ref_token_var = ref_token_var
        self._ref_k_cpu = ref_k
        self._ref_v_cpu = ref_v
        self._ref_k: Optional[torch.Tensor] = None
        self._ref_v: Optional[torch.Tensor] = None
        self._vision_ref_k_cpu = vision_ref_k
        self._vision_ref_v_cpu = vision_ref_v
        self._vision_ref_k: Optional[torch.Tensor] = None
        self._vision_ref_v: Optional[torch.Tensor] = None
        self._dummy_prefix_ids = dummy_prefix_ids
        self._dummy_suffix_ids = dummy_suffix_ids
        self._audio_repeat_frames = audio_repeat_frames

        self._vision_feature_layer = vision_feature_layer
        self._vision_feature_select_strategy = vision_feature_select_strategy

        self._avqa_ann: Optional[List[Dict[str, Any]]] = None
        self._avqa_mapping: Optional[Dict[str, str]] = None
        self._avqa_choice_token_ids: Optional[torch.Tensor] = None

    def create_optimizer(self):  # type: ignore[override]
        if self.optimizer is not None:
            return self.optimizer

        weight_decay = float(getattr(self.args, "weight_decay", 0.0) or 0.0)
        if weight_decay != 0.0:
            return super().create_optimizer()

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        params = [p for p in self.model.parameters() if getattr(p, "requires_grad", False)]
        if not params:
            raise ValueError("No trainable parameters found when creating the optimizer.")
        self.optimizer = optimizer_cls(params, **optimizer_kwargs)
        return self.optimizer

    @staticmethod
    def _trainable_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        state: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if not getattr(param, "requires_grad", False):
                continue
            if not isinstance(param, torch.Tensor):
                continue
            state[name] = param.detach().cpu()
        return state

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False) -> None:
        """
        When using DeepSpeed + LoRA/PEFT, Transformers' default save path gathers the full base model state dict,
        which is prohibitively expensive for 7B models. We only need to persist the trainable adapter weights.
        """

        if output_dir is None:
            output_dir = self.args.output_dir
        if not self.args.should_save:
            return

        if self.is_deepspeed_enabled:
            try:
                from peft import PeftModel  # type: ignore[import-not-found]
            except Exception:
                PeftModel = None  # type: ignore[assignment]

            unwrapped = self.accelerator.unwrap_model(self.model, keep_torch_compile=False)
            if PeftModel is not None and isinstance(unwrapped, PeftModel):
                state_dict = self._trainable_state_dict(unwrapped)
                self._save(output_dir, state_dict=state_dict)
                return

        super().save_model(output_dir=output_dir, _internal_call=_internal_call)

    def _get_language_model(self, model: Any) -> Any:
        def unwrap_lm(obj: Any) -> Optional[Any]:
            if obj is None:
                return None
            if hasattr(obj, "layers") and hasattr(obj, "rotary_emb"):
                return obj
            inner = getattr(obj, "model", None)
            if inner is not None and hasattr(inner, "layers") and hasattr(inner, "rotary_emb"):
                return inner
            return None

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
                add(getter())
            except Exception:
                pass

        seen: set[int] = set()
        for cand in candidates:
            if cand is None or id(cand) in seen:
                continue
            seen.add(id(cand))

            if hasattr(cand, "language_model"):
                lm = getattr(cand, "language_model")
                found = unwrap_lm(lm)
                if found is not None:
                    return found

            inner = getattr(cand, "model", None)
            if inner is not None and hasattr(inner, "language_model"):
                lm = getattr(inner, "language_model")
                found = unwrap_lm(lm)
                if found is not None:
                    return found

        raise AttributeError("Could not locate Qwen2 language model on the provided LynX model instance.")

    def _load_avqa_assets(self) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, str]]]:
        if self._avqa_ann is None:
            if not os.path.exists(self._avqa_args.avqa_annotation_file):
                raise FileNotFoundError(f"AVQA annotation file not found: {self._avqa_args.avqa_annotation_file}")
            with open(self._avqa_args.avqa_annotation_file, "r") as f:
                ann = json.load(f)
            if not isinstance(ann, list):
                raise TypeError(f"Expected a list in {self._avqa_args.avqa_annotation_file}, got {type(ann)}")
            self._avqa_ann = ann

        if self._avqa_mapping is None and self._avqa_args.avqa_video_mapping:
            if not os.path.exists(self._avqa_args.avqa_video_mapping):
                raise FileNotFoundError(f"AVQA mapping file not found: {self._avqa_args.avqa_video_mapping}")
            with open(self._avqa_args.avqa_video_mapping, "r") as f:
                mapping = json.load(f)
            if not isinstance(mapping, dict):
                raise TypeError(f"Expected a dict in {self._avqa_args.avqa_video_mapping}, got {type(mapping)}")
            self._avqa_mapping = mapping

        return self._avqa_ann, self._avqa_mapping

    def _get_avqa_choice_token_ids(self, device: torch.device) -> torch.Tensor:
        if self._avqa_choice_token_ids is not None:
            return self._avqa_choice_token_ids.to(device)

        ids: List[int] = []
        for letter in ("A", "B", "C", "D"):
            for variant in (letter, f" {letter}"):
                enc = self._tokenizer.encode(variant, add_special_tokens=False)
                if len(enc) != 1:
                    raise ValueError(f"Expected {variant!r} to be a single token, got ids={enc}")
                ids.append(int(enc[0]))
        self._avqa_choice_token_ids = torch.tensor(ids, dtype=torch.long)
        return self._avqa_choice_token_ids.to(device)

    def compute_loss(  # type: ignore[override]
        self,
        model: Any,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
        **kwargs: Any,
    ):
        core_model = getattr(model, "module", model)

        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        if "audio_pixel_values_videos" in inputs:
            modality_pv = inputs["audio_pixel_values_videos"]
        elif "depth_pixel_values_videos" in inputs:
            modality_pv = inputs["depth_pixel_values_videos"]
        elif "fast_pixel_values_videos" in inputs:
            modality_pv = inputs["fast_pixel_values_videos"]
        else:
            raise KeyError("Expected one of: audio_pixel_values_videos, depth_pixel_values_videos, fast_pixel_values_videos")

        modality_pv = modality_pv.to(device=device, dtype=dtype)
        batch_size = int(modality_pv.shape[0])

        lambda_distill = float(getattr(self._loss_args, "lambda_distill", 0.0) or 0.0)
        distill_scope = str(getattr(self._loss_args, "distill_scope", "all") or "all").lower()
        if distill_scope not in ("vision", "all"):
            distill_scope = "all"
        distill_vision = lambda_distill > 0.0
        distill_llm = lambda_distill > 0.0 and distill_scope == "all"

        l_distill_sum = torch.zeros((), device=device, dtype=dtype)
        distill_layers = 0

        vision_handles: List[Any] = []
        if distill_vision:
            if self._vision_ref_k_cpu is None or self._vision_ref_v_cpu is None:
                raise KeyError("Trainer is missing vision reference KV cache (vision_ref_k/vision_ref_v). Recompute reference_stats.pt.")

            if self._vision_ref_k is None or self._vision_ref_v is None:
                self._vision_ref_k = self._vision_ref_k_cpu.to(device=device)
                self._vision_ref_v = self._vision_ref_v_cpu.to(device=device)

            vision_ref_k = self._vision_ref_k.to(dtype=dtype)
            vision_ref_v = self._vision_ref_v.to(dtype=dtype)

            vision_tower = get_vision_tower(core_model)
            vision_layers = get_siglip_encoder_layers(vision_tower)
            num_vision_layers = len(vision_layers)
            if num_vision_layers == 0:
                raise RuntimeError("Vision tower exposes no encoder layers for attention distillation.")
            if vision_ref_k.shape[0] != num_vision_layers or vision_ref_v.shape[0] != num_vision_layers:
                raise ValueError(f"Vision reference KV cache layers mismatch: ref has {vision_ref_k.shape[0]}, model has {num_vision_layers}")

            seq_len = int(vision_ref_k.shape[2])
            q_idx = llava_onevision_spatial_pool_query_indices(seq_len, device=device)

            def make_vision_hook(layer_idx: int):
                def hook(module: Any, module_inputs: Tuple[Any, ...], module_kwargs: Dict[str, Any], module_outputs: Any) -> None:
                    nonlocal l_distill_sum, distill_layers

                    hidden_states = module_inputs[0] if len(module_inputs) > 0 else module_kwargs.get("hidden_states")
                    if hidden_states is None:
                        raise RuntimeError("Failed to capture vision attention hidden_states for distillation (missing kwargs support).")
                    out_new = module_outputs[0] if isinstance(module_outputs, (tuple, list)) else module_outputs

                    num_heads = int(getattr(module, "num_heads", 0) or 0)
                    head_dim = int(getattr(module, "head_dim", 0) or 0)
                    scale = float(getattr(module, "scale", head_dim**-0.5 if head_dim else 1.0))
                    if num_heads <= 0 or head_dim <= 0:
                        raise ValueError("Could not infer SigLip attention num_heads/head_dim for distillation.")

                    q = module.q_proj(hidden_states)
                    batch_frames, q_seq, _ = q.shape
                    if q_seq != seq_len:
                        raise ValueError(f"Vision seq_len mismatch for distillation: got {q_seq}, expected {seq_len}")

                    q = q.view(batch_frames, q_seq, num_heads, head_dim).transpose(1, 2)  # (Bf, H, S, Hd)
                    q = q.index_select(2, q_idx)  # (Bf, H, S', Hd)
                    out_new = out_new.index_select(1, q_idx)  # (Bf, S', D)

                    k_ref = vision_ref_k[layer_idx].unsqueeze(0).expand(batch_frames, -1, -1, -1)
                    v_ref = vision_ref_v[layer_idx].unsqueeze(0).expand(batch_frames, -1, -1, -1)

                    attn_mask = module_inputs[1] if len(module_inputs) > 1 else module_kwargs.get("attention_mask")
                    if attn_mask is not None:
                        if attn_mask.ndim == 4 and attn_mask.shape[2] == q_seq:
                            attn_mask = attn_mask.index_select(2, q_idx)
                        else:
                            attn_mask = None

                    q_scaled = q * (scale * math.sqrt(float(head_dim)))
                    ctx_ref = F.scaled_dot_product_attention(q_scaled, k_ref, v_ref, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
                    ctx_ref = ctx_ref.transpose(1, 2).contiguous().view(batch_frames, q.shape[2], num_heads * head_dim)
                    out_ref = module.out_proj(ctx_ref)

                    l_distill_sum = l_distill_sum + F.l1_loss(out_new, out_ref, reduction="mean")
                    distill_layers += 1

                return hook

            for layer_idx, layer in enumerate(vision_layers):
                attn = getattr(layer, "self_attn", None)
                if attn is None:
                    raise AttributeError(f"Vision encoder layer {layer_idx} does not expose `self_attn` for distillation.")
                hook_fn = make_vision_hook(layer_idx)
                try:
                    vision_handles.append(attn.register_forward_hook(hook_fn, with_kwargs=True))
                except TypeError:
                    # Older PyTorch: kwargs won't be available, but some implementations pass hidden_states positionally.
                    def hook_no_kwargs(module: Any, module_inputs: Tuple[Any, ...], module_outputs: Any, *, _hook=hook_fn) -> None:
                        _hook(module, module_inputs, {}, module_outputs)

                    vision_handles.append(attn.register_forward_hook(hook_no_kwargs))

        try:
            z_m = encode_llava_onevision_video_tokens(
                core_model,
                pixel_values_videos=modality_pv,
                vision_feature_layer=self._vision_feature_layer,
                vision_feature_select_strategy=self._vision_feature_select_strategy,
                repeat_frames=self._audio_repeat_frames,
            )  # (B, V, D)
        finally:
            for h in vision_handles:
                try:
                    h.remove()
                except Exception:
                    pass

        flat = z_m.flatten(0, 1)
        batch_mean = flat.mean(dim=0)
        batch_var = flat.var(dim=0, unbiased=False)
        l_stat = torch.norm(batch_mean - self._ref_token_mean.to(device), p=2) + torch.norm(batch_var - self._ref_token_var.to(device), p=2)

        outputs: Any = None

        if distill_llm or return_outputs:
            embed = core_model.get_input_embeddings()
            prefix_emb = embed(self._dummy_prefix_ids.to(device)).unsqueeze(0).expand(batch_size, -1, -1)
            suffix_emb = embed(self._dummy_suffix_ids.to(device)).unsqueeze(0).expand(batch_size, -1, -1)
            inputs_embeds = torch.cat([prefix_emb, z_m, suffix_emb], dim=1)

            seq_len = inputs_embeds.shape[1]
            attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=device)
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

            lm = self._get_language_model(core_model)
            llm_layers = list(getattr(lm, "layers"))
            num_layers = len(llm_layers)
            if num_layers == 0:
                raise RuntimeError("Language model exposes no layers for attention distillation.")

            if distill_llm:
                if self._ref_k_cpu is None or self._ref_v_cpu is None:
                    raise KeyError("Trainer is missing LLM reference KV cache (ref_k/ref_v). Recompute reference_stats.pt.")
                if self._ref_k is None or self._ref_v is None:
                    self._ref_k = self._ref_k_cpu.to(device=device)
                    self._ref_v = self._ref_v_cpu.to(device=device)

            ref_k = None if self._ref_k is None else self._ref_k.to(dtype=dtype)
            ref_v = None if self._ref_v is None else self._ref_v.to(dtype=dtype)

            v_start = int(prefix_emb.shape[1])
            v_end = v_start + int(z_m.shape[1])
            if distill_llm:
                if ref_k is None or ref_v is None:
                    raise RuntimeError("Missing reference KV cache tensors after initialization.")
                if ref_k.shape[0] != num_layers or ref_v.shape[0] != num_layers:
                    raise ValueError(f"Reference KV cache layers mismatch: ref_k has {ref_k.shape[0]}, model has {num_layers}")
                if ref_k.shape[2] != (v_end - v_start) or ref_v.shape[2] != (v_end - v_start):
                    raise ValueError(
                        f"Reference KV cache length mismatch: ref_k has {ref_k.shape[2]}, expected {(v_end - v_start)} "
                        f"(check reference_num_frames vs audio_repeat_frames)."
                    )

            if distill_llm:
                q_pos = seq_len - 1
                q_inputs: List[Optional[torch.Tensor]] = [None for _ in range(num_layers)]
                handles: List[Any] = []

                def make_hook(layer_idx: int):
                    def hook(module: Any, module_inputs: Tuple[Any, ...], module_kwargs: Dict[str, Any], module_outputs: Any) -> None:
                        hidden_states = module_inputs[0] if len(module_inputs) > 0 else module_kwargs.get("hidden_states")
                        if hidden_states is None:
                            raise RuntimeError("Failed to capture LLM attention hidden_states for distillation (missing kwargs support).")
                        q_inputs[layer_idx] = hidden_states[:, q_pos : q_pos + 1, :]

                    return hook

                for layer_idx, layer in enumerate(llm_layers):
                    hook_fn = make_hook(layer_idx)
                    try:
                        handles.append(layer.self_attn.register_forward_hook(hook_fn, with_kwargs=True))
                    except TypeError:
                        def hook_no_kwargs(module: Any, module_inputs: Tuple[Any, ...], module_outputs: Any, *, _hook=hook_fn) -> None:
                            _hook(module, module_inputs, {}, module_outputs)

                        handles.append(layer.self_attn.register_forward_hook(hook_no_kwargs))

                try:
                    outputs = model(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        use_cache=True,
                        num_logits_to_keep=1,
                        return_dict=True,
                    )
                finally:
                    for h in handles:
                        try:
                            h.remove()
                        except Exception:
                            pass

                pkv = getattr(outputs, "past_key_values", None)
                if pkv is None:
                    raise RuntimeError("Model did not return past_key_values with use_cache=True (required for distillation).")

                head_dim = int(pkv[0][0].shape[-1])
                cos, sin = lm.rotary_emb(
                    torch.zeros((batch_size, 1, 1, head_dim), device=device, dtype=dtype),
                    position_ids[:, q_pos : q_pos + 1],
                )

                from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

                if ref_k is None or ref_v is None:
                    raise RuntimeError("Missing reference KV cache tensors after initialization.")

                for layer_idx, layer in enumerate(llm_layers):
                    attn = layer.self_attn
                    hs_q = q_inputs[layer_idx]
                    if hs_q is None:
                        raise RuntimeError(f"Failed to capture attention input for layer {layer_idx}.")

                    q = attn.q_proj(hs_q)
                    num_heads = int(q.shape[-1] // head_dim)
                    if num_heads <= 0:
                        raise ValueError(f"Invalid num_heads={num_heads} inferred for distillation.")
                    q = q.view(batch_size, 1, num_heads, head_dim).transpose(1, 2)  # (B, H, 1, Hd)
                    q, _ = apply_rotary_pos_emb(q, torch.zeros((batch_size, 1, 1, head_dim), device=device, dtype=q.dtype), cos, sin)

                    k_l, v_l = pkv[layer_idx]
                    k = k_l[:, :, v_start:v_end, :]
                    v = v_l[:, :, v_start:v_end, :]

                    kv_groups = int(num_heads // k.shape[1])
                    k_rep = repeat_kv(k, kv_groups)
                    v_rep = repeat_kv(v, kv_groups)
                    logits = torch.matmul(q, k_rep.transpose(-2, -1)) / math.sqrt(float(head_dim))
                    weights = torch.softmax(logits, dim=-1)
                    ctx = torch.matmul(weights, v_rep)
                    ctx = ctx.transpose(1, 2).contiguous().view(batch_size, 1, num_heads * head_dim)
                    out_new = attn.o_proj(ctx)

                    k_ref = ref_k[layer_idx].unsqueeze(0).expand(batch_size, -1, -1, -1)
                    v_ref = ref_v[layer_idx].unsqueeze(0).expand(batch_size, -1, -1, -1)
                    k_ref_rep = repeat_kv(k_ref, kv_groups)
                    v_ref_rep = repeat_kv(v_ref, kv_groups)
                    logits_ref = torch.matmul(q, k_ref_rep.transpose(-2, -1)) / math.sqrt(float(head_dim))
                    weights_ref = torch.softmax(logits_ref, dim=-1)
                    ctx_ref = torch.matmul(weights_ref, v_ref_rep)
                    ctx_ref = ctx_ref.transpose(1, 2).contiguous().view(batch_size, 1, num_heads * head_dim)
                    out_ref = attn.o_proj(ctx_ref)

                    l_distill_sum = l_distill_sum + F.l1_loss(out_new, out_ref, reduction="mean")
                    distill_layers += 1
            elif return_outputs:
                outputs = model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                    num_logits_to_keep=1,
                    return_dict=True,
                )

        l_distill = l_distill_sum / float(distill_layers) if distill_layers > 0 else torch.zeros((), device=device, dtype=dtype)
        # print("L_stat: ", l_stat, "L_dist: ", l_distill)
        loss = self._loss_args.lambda_stat * l_stat + lambda_distill * l_distill
        return (loss, outputs) if return_outputs else loss

    @torch.no_grad()
    def evaluate(  # type: ignore[override]
        self,
        eval_dataset: Optional[Any] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        if not getattr(self._avqa_args, "enable_avqa_eval", True):
            metrics = {f"{metric_key_prefix}_avqa_skipped": 1.0}
            self.log(metrics)
            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
            return metrics

        try:
            ann, mapping = self._load_avqa_assets()
        except Exception as exc:
            metrics = {f"{metric_key_prefix}_avqa_error": 1.0}
            if self.is_world_process_zero():
                print(f"[AVQA Eval] Skipping: {exc}")
            self.log(metrics)
            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
            return metrics

        if getattr(self._avqa_args, "avqa_max_samples", None) is not None:
            ann = ann[: int(self._avqa_args.avqa_max_samples)]

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1

        core_model = getattr(self.model, "module", self.model)
        was_training = core_model.training
        core_model.eval()

        device = next(self.model.parameters()).device
        model_dtype = next(self.model.parameters()).dtype

        local_preds: List[Dict[str, Any]] = []
        num_samples = len(ann)
        processed_global = 0
        pbar = None
        if rank == 0:
            pbar = tqdm(total=num_samples, desc="AVQA", dynamic_ncols=True, leave=False)

        eval_batch_size = int(getattr(self._avqa_args, "avqa_eval_batch_size", 1) or 1)
        if eval_batch_size <= 0:
            eval_batch_size = 1

        indices = list(range(rank, num_samples, world_size))
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

                filename = mapping.get(vid, f"{vid}.mp4") if isinstance(mapping, dict) else f"{vid}.mp4"
                video_path = os.path.join(self._avqa_args.avqa_video_root, filename)
                if not os.path.exists(video_path):
                    continue

                qa = build_avqa_question(item)
                messages = [{"role": "user", "content": f"<image>\n{qa['question']}"}]
                prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

                try:
                    vframes, aframes, info = load_video_frames_and_audio(
                        video_path,
                        num_frames=int(self._avqa_args.avqa_num_frames),
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
                        clip_duration_s=2.0,
                        clip_stride_s=0.5,
                        num_mel_bins=128,
                        target_length=204,
                        mean=-4.268,
                        std=9.138,
                    )  # (F, 1, 128, 204)
                    audio_frames = [log_mel_to_pil_rgb(audio_clips[i, 0]) for i in range(audio_clips.shape[0])]

                    placeholder_id = self._tokenizer.convert_tokens_to_ids("<image>")
                    prompt_ids = self._tokenizer(prompt, return_tensors="pt").input_ids[0]
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
                if pbar is not None:
                    remaining = num_samples - processed_global
                    step = min(world_size * len(batch_indices), remaining) if remaining > 0 else 0
                    processed_global += step
                    pbar.update(step)
                continue

            try:
                pv_video = self._processor.video_processor(batch_videos, return_tensors="pt")["pixel_values_videos"].to(device=device, dtype=model_dtype)
                pv_audio = self._processor.video_processor(batch_audios, return_tensors="pt")["pixel_values_videos"].to(device=device, dtype=model_dtype)

                # Ensure eval uses the same mixed-precision context as Trainer's built-in loops.
                # This also avoids dtype mismatches when LoRA adapter weights are fp32.
                with self.autocast_smart_context_manager():
                    with _maybe_disable_adapter(core_model):
                        z_v = encode_llava_onevision_video_tokens(
                            core_model,
                            pixel_values_videos=pv_video,
                            vision_feature_layer=core_model.config.vision_feature_layer,
                            vision_feature_select_strategy=core_model.config.vision_feature_select_strategy,
                            frame_chunk_size=int(getattr(self._avqa_args, "avqa_frame_chunk_size", 4)),
                        )

                    z_a = encode_llava_onevision_video_tokens(
                        core_model,
                        pixel_values_videos=pv_audio,
                        vision_feature_layer=core_model.config.vision_feature_layer,
                        vision_feature_select_strategy=core_model.config.vision_feature_select_strategy,
                    )

                # Be defensive: PEFT LoRA weights are often fp32 even when the base model is bf16/fp16,
                # which can promote activations to fp32 and break torch.cat.
                z_v = z_v.to(dtype=model_dtype)
                z_a = z_a.to(dtype=model_dtype)

                if getattr(self._avqa_args, "avqa_pool_video_tokens", False):
                    z_v = temporal_mean_pool_video_tokens(z_v, frames=int(pv_video.shape[1]))
                    z_a = temporal_mean_pool_video_tokens(z_a, frames=int(pv_audio.shape[1]))

                z = torch.cat([z_v, z_a], dim=1)

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

                with self.autocast_smart_context_manager():
                    out = core_model(
                        inputs_embeds=inputs_embeds_padded,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        use_cache=False,
                        num_logits_to_keep=1,
                        return_dict=True,
                    )
                choice_ids = self._get_avqa_choice_token_ids(device)
                logits_last = out.logits[:, -1]
                logits = logits_last.index_select(1, choice_ids)
                best = torch.argmax(logits, dim=1)
                pred_ids = choice_ids[best]

                for i, row in enumerate(batch_rows):
                    pred = self._tokenizer.decode([int(pred_ids[i].item())]).strip()
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

            if pbar is not None:
                remaining = num_samples - processed_global
                step = min(world_size * len(batch_indices), remaining) if remaining > 0 else 0
                processed_global += step
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

        metrics: Optional[Dict[str, float]] = None
        if rank == 0:
            merged: List[Dict[str, Any]] = []
            for part in gathered:
                merged.extend(part)
            merged.sort(key=lambda x: int(x.get("question_id", 0)))

            correct = sum(1 for ele in merged if ele["answer"] in ele["caption"])
            total = len(merged)
            acc = float(correct) / float(total) if total > 0 else 0.0

            out_dir = getattr(self._avqa_args, "avqa_predictions_dir", None) or os.path.join(self.args.output_dir, "avqa_eval")
            os.makedirs(out_dir, exist_ok=True)
            if getattr(self._avqa_args, "avqa_save_predictions", True):
                pred_path = os.path.join(out_dir, f"avqa_predictions_step{self.state.global_step}.json")
                with open(pred_path, "w") as f:
                    json.dump(merged, f)

            metrics = {
                f"{metric_key_prefix}_avqa_acc": acc,
                f"{metric_key_prefix}_avqa_correct": float(correct),
                f"{metric_key_prefix}_avqa_total": float(total),
            }

            if getattr(self._avqa_args, "avqa_save_metrics", True):
                metrics_path = getattr(self._avqa_args, "avqa_metrics_path", None) or os.path.join(out_dir, "avqa_metrics.jsonl")
                record = {
                    "global_step": int(self.state.global_step),
                    "epoch": None if self.state.epoch is None else float(self.state.epoch),
                    **metrics,
                }
                with open(metrics_path, "a") as f:
                    f.write(json.dumps(record) + "\n")

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            payload: List[Optional[Dict[str, float]]] = [metrics]
            torch.distributed.broadcast_object_list(payload, src=0)
            metrics = payload[0]

        if metrics is None:
            metrics = {f"{metric_key_prefix}_avqa_error": 1.0}

        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        if was_training:
            core_model.train()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        return metrics


def _coerce_dialog_text(value: Any) -> Optional[str]:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value:
        first = value[0]
        if isinstance(first, str):
            return first
    return None


def _build_avsd_messages(dialog: Any) -> List[Dict[str, str]]:
    if not isinstance(dialog, list):
        raise TypeError(f"Expected dialog to be a list, got {type(dialog)}")

    turns: List[Dict[str, Optional[str]]] = []
    for turn in dialog:
        if not isinstance(turn, dict):
            continue
        q = _coerce_dialog_text(turn.get("question"))
        if q is None:
            continue
        a = _coerce_dialog_text(turn.get("answer"))
        turns.append({"question": q, "answer": a})

    if not turns:
        raise ValueError("Dialog has no valid turns.")

    messages: List[Dict[str, str]] = []
    for i, turn in enumerate(turns):
        q = (turn.get("question") or "").strip()
        if i == 0:
            q = f"<image>\n{q}"
        messages.append({"role": "user", "content": q})
        if i < len(turns) - 1:
            a = (turn.get("answer") or "").strip()
            messages.append({"role": "assistant", "content": a})
    return messages


def _get_choice_token_ids(tokenizer: Any, device: torch.device) -> torch.Tensor:
    ids: List[int] = []
    for letter in ("A", "B", "C", "D"):
        for variant in (letter, f" {letter}"):
            enc = tokenizer.encode(variant, add_special_tokens=False)
            if len(enc) != 1:
                raise ValueError(f"Expected {variant!r} to be a single token, got ids={enc}")
            ids.append(int(enc[0]))
    return torch.tensor(ids, dtype=torch.long, device=device)


class _LynXPeftSavingTrainer(Trainer):
    def create_optimizer(self):  # type: ignore[override]
        if self.optimizer is not None:
            return self.optimizer

        weight_decay = float(getattr(self.args, "weight_decay", 0.0) or 0.0)
        if weight_decay != 0.0:
            return super().create_optimizer()

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        params = [p for p in self.model.parameters() if getattr(p, "requires_grad", False)]
        if not params:
            raise ValueError("No trainable parameters found when creating the optimizer.")
        self.optimizer = optimizer_cls(params, **optimizer_kwargs)
        return self.optimizer

    @staticmethod
    def _trainable_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        state: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if not getattr(param, "requires_grad", False):
                continue
            if not isinstance(param, torch.Tensor):
                continue
            state[name] = param.detach().cpu()
        return state

    @staticmethod
    def _infer_adapter_names_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> List[str]:
        names: set[str] = set()
        for key in state_dict.keys():
            for needle in (".lora_A.", ".lora_B."):
                if needle not in key:
                    continue
                suffix = key.split(needle, 1)[1]
                adapter = suffix.split(".", 1)[0]
                if adapter:
                    names.add(adapter)
        return sorted(names)

    def _find_peft_model(self) -> Optional[Any]:
        try:
            from peft import PeftModel  # type: ignore[import-not-found]
        except Exception:
            return None

        candidates: List[Any] = []

        def add(obj: Any) -> None:
            if obj is None:
                return
            candidates.append(obj)

        add(self.model)
        add(getattr(self.model, "module", None))

        wrapper = getattr(self.model, "module", self.model)
        add(getattr(wrapper, "base_model", None))
        add(getattr(wrapper, "model", None))

        for cand in candidates:
            if cand is None:
                continue
            if isinstance(cand, PeftModel):
                return cand
        return None

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False) -> None:
        """
        Avoid saving the full base model weights during DeepSpeed runs.
        Instead, persist only the trainable PEFT adapter weights (LLM LoRA during SFT).
        """

        if output_dir is None:
            output_dir = self.args.output_dir
        if not self.args.should_save:
            return

        if self.is_deepspeed_enabled:
            peft_model = self._find_peft_model()
            if peft_model is not None:
                state_dict = self._trainable_state_dict(peft_model)
                adapter_names = self._infer_adapter_names_from_state_dict(state_dict)
                if adapter_names:
                    supported = set(getattr(peft_model, "peft_config", {}).keys())
                    selected = [name for name in adapter_names if name in supported]
                else:
                    selected = []

                if selected:
                    peft_model.save_pretrained(
                        output_dir,
                        state_dict=state_dict,
                        safe_serialization=bool(getattr(self.args, "save_safetensors", True)),
                        selected_adapters=selected,
                        is_main_process=True,
                    )

                    if getattr(self, "processing_class", None) is not None:
                        self.processing_class.save_pretrained(output_dir)  # type: ignore[union-attr]
                    elif (
                        self.data_collator is not None
                        and hasattr(self.data_collator, "tokenizer")
                        and self.data_collator.tokenizer is not None
                    ):
                        self.data_collator.tokenizer.save_pretrained(output_dir)

                    torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
                    return

        super().save_model(output_dir=output_dir, _internal_call=_internal_call)


class LynXSFTAVQATrainer(_LynXPeftSavingTrainer):
    """
    Standard SFT Trainer + periodic AVQA validation evaluation (accuracy).

    Evaluation uses all modalities (video + audio + text) from AVQA.
    """

    def __init__(
        self,
        *args: Any,
        processor: Any,
        avqa_args: Any,
        audio_target_sr: int,
        audio_seconds: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._processor = processor
        self._tokenizer = processor.tokenizer
        self._avqa_args = avqa_args

        self._audio_target_sr = int(audio_target_sr)
        self._audio_seconds = float(audio_seconds)

        self._avqa_ann: Optional[List[Dict[str, Any]]] = None
        self._avqa_mapping: Optional[Dict[str, str]] = None
        self._avqa_choice_token_ids: Optional[torch.Tensor] = None

    def _load_avqa_assets(self) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
        if self._avqa_ann is None:
            ann = load_json(self._avqa_args.avqa_annotation_file)
            if not isinstance(ann, list):
                raise TypeError(f"Expected a list in {self._avqa_args.avqa_annotation_file}, got {type(ann)}")
            self._avqa_ann = ann
        if self._avqa_mapping is None:
            mapping: Dict[str, str] = {}
            if self._avqa_args.avqa_video_mapping:
                raw = load_json(self._avqa_args.avqa_video_mapping)
                if isinstance(raw, dict):
                    mapping = {str(k): str(v) for k, v in raw.items()}
            self._avqa_mapping = mapping
        return self._avqa_ann, self._avqa_mapping

    def _get_avqa_choice_token_ids(self, device: torch.device) -> torch.Tensor:
        if self._avqa_choice_token_ids is None or self._avqa_choice_token_ids.device != device:
            self._avqa_choice_token_ids = _get_choice_token_ids(self._tokenizer, device=device)
        return self._avqa_choice_token_ids

    @torch.no_grad()
    def evaluate(  # type: ignore[override]
        self,
        eval_dataset: Optional[Any] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        if not getattr(self._avqa_args, "enable_avqa_eval", True):
            metrics = {f"{metric_key_prefix}_avqa_skipped": 1.0}
            self.log(metrics)
            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
            return metrics

        try:
            ann, mapping = self._load_avqa_assets()
        except Exception as exc:
            metrics = {f"{metric_key_prefix}_avqa_error": 1.0}
            if self.is_world_process_zero():
                print(f"[AVQA Eval] Skipping: {exc}")
            self.log(metrics)
            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
            return metrics

        if getattr(self._avqa_args, "avqa_max_samples", None) is not None:
            ann = ann[: int(self._avqa_args.avqa_max_samples)]

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1

        wrapper = getattr(self.model, "module", self.model)
        core_model = getattr(wrapper, "base_model", wrapper)
        core_model = getattr(core_model, "module", core_model)

        was_training = core_model.training
        core_model.eval()

        device = next(core_model.parameters()).device
        if device.type == "cuda":
            model_dtype = torch.bfloat16 if self.args.bf16 else (torch.float16 if self.args.fp16 else next(core_model.parameters()).dtype)
        else:
            model_dtype = torch.float32

        placeholder_id = self._tokenizer.convert_tokens_to_ids("<image>")
        if placeholder_id is None or placeholder_id == getattr(self._tokenizer, "unk_token_id", None):
            raise ValueError("Tokenizer does not recognize '<image>' as a single token.")

        local_preds: List[Dict[str, Any]] = []
        num_samples = len(ann)
        processed_global = 0
        pbar = None
        if rank == 0:
            pbar = tqdm(total=num_samples, desc="AVQA", dynamic_ncols=True, leave=False)

        micro_batch = int(getattr(self._avqa_args, "avqa_eval_batch_size", 1) or 1)
        micro_batch = max(1, micro_batch)

        for start in range(rank * micro_batch, num_samples, world_size * micro_batch):
            batch_indices = list(range(start, min(start + micro_batch, num_samples)))
            batch_rows: List[Dict[str, Any]] = []
            batch_videos: List[torch.Tensor] = []
            batch_audios: List[Image.Image] = []

            for idx in batch_indices:
                item = ann[idx]
                if not isinstance(item, dict):
                    continue
                vid = item.get("video_id") or item.get("video") or item.get("video_name") or item.get("video_id_str")
                if not isinstance(vid, (str, int)):
                    local_preds.append(
                        {
                            "question_id": item.get("id", idx),
                            "image_id": str(vid),
                            "caption": "",
                            "answer": "",
                            "error": "invalid_video_id",
                        }
                    )
                    continue
                vid = str(vid)

                qa = build_avqa_question(item)
                prompt = f"<image>\n{qa['question']}"
                messages = [{"role": "user", "content": prompt}]
                prefix_ids: List[int] = self._tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
                suffix_ids: List[int] = self._tokenizer.encode("", add_special_tokens=False)

                video_name = mapping.get(vid, vid)
                video_path = os.path.join(str(self._avqa_args.avqa_video_root), str(video_name))
                if not video_path.endswith(".mp4"):
                    video_path += ".mp4"

                if not os.path.exists(video_path):
                    local_preds.append(
                        {
                            "question_id": item.get("id", idx),
                            "image_id": vid,
                            "caption": "",
                            "answer": qa["answer"],
                            "error": f"video_not_found: {video_path}",
                        }
                    )
                    continue

                try:
                    vframes, aframes, info = load_video_frames_and_audio(
                        video_path,
                        num_frames=int(getattr(self._avqa_args, "avqa_num_frames", 20)),
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

                    audio_frames = waveform_to_imagebind_melspec_clips(
                        waveform,
                        sample_rate=self._audio_target_sr,
                        num_clips=int(vframes.shape[0]),
                        clip_duration_s=2.0,
                        clip_stride_s=0.5,
                        num_mel_bins=128,
                        target_length=204,
                        mean=-4.268,
                        std=9.138,
                    )
                    audio_frames = [log_mel_to_pil_rgb(audio_frames[i, 0]) for i in range(audio_frames.shape[0])]

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
                if pbar is not None:
                    remaining = num_samples - processed_global
                    step = min(world_size * len(batch_indices), remaining) if remaining > 0 else 0
                    processed_global += step
                    pbar.update(step)
                continue

            try:
                pv_video = self._processor.video_processor(batch_videos, return_tensors="pt")["pixel_values_videos"].to(device=device, dtype=model_dtype)
                pv_audio = self._processor.video_processor(batch_audios, return_tensors="pt")["pixel_values_videos"].to(device=device, dtype=model_dtype)

                orig_chunk = getattr(wrapper, "frame_chunk_size", None)
                try:
                    wrapper.frame_chunk_size = int(getattr(self._avqa_args, "avqa_frame_chunk_size", 4))
                    z_v, z_a = wrapper._encode_streams(pixel_values_videos=pv_video, audio_pixel_values_videos=pv_audio)
                finally:
                    if orig_chunk is not None:
                        wrapper.frame_chunk_size = orig_chunk
                z_v = z_v.to(dtype=model_dtype)
                z_a = z_a.to(dtype=model_dtype)

                z = torch.cat([z_v, z_a], dim=1)

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

                with self.autocast_smart_context_manager():
                    out = core_model(
                        inputs_embeds=inputs_embeds_padded,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        use_cache=False,
                    )
                logits = out.logits[:, -1, :]
                choice_ids = self._get_avqa_choice_token_ids(device)
                choice_logits = logits.index_select(dim=-1, index=choice_ids)
                pred_idx = int(choice_logits.argmax(dim=-1)[0].item())

                letters = ["A", "A", "B", "B", "C", "C", "D", "D"]
                pred = letters[pred_idx]

                for row in batch_rows:
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
                err = f"avqa_eval_error_batch: {type(exc).__name__}: {msg}"
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

            if pbar is not None:
                remaining = num_samples - processed_global
                step = min(world_size * micro_batch, remaining) if remaining > 0 else 0
                processed_global += step
                pbar.update(step)

        if pbar is not None:
            pbar.close()

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            gathered: List[List[Dict[str, Any]]] = [None for _ in range(world_size)]  # type: ignore[list-item]
            torch.distributed.all_gather_object(gathered, local_preds)
            merged: List[Dict[str, Any]] = []
            for chunk in gathered:
                merged.extend(chunk)
        else:
            merged = local_preds

        metrics: Optional[Dict[str, float]] = None
        if rank == 0:
            total = 0
            correct = 0
            for pred in merged:
                if pred.get("error"):
                    continue
                total += 1
                if str(pred.get("caption", "")).strip().upper() == str(pred.get("answer", "")).strip().upper():
                    correct += 1
            acc = float(correct) / float(total) if total > 0 else 0.0

            out_dir = getattr(self._avqa_args, "avqa_predictions_dir", None) or os.path.join(self.args.output_dir, "avqa_eval")
            os.makedirs(out_dir, exist_ok=True)
            if getattr(self._avqa_args, "avqa_save_predictions", True):
                pred_path = os.path.join(out_dir, f"avqa_predictions_step{self.state.global_step}.json")
                with open(pred_path, "w") as f:
                    json.dump(merged, f)

            metrics = {
                f"{metric_key_prefix}_avqa_acc": acc,
                f"{metric_key_prefix}_avqa_correct": float(correct),
                f"{metric_key_prefix}_avqa_total": float(total),
            }

            if getattr(self._avqa_args, "avqa_save_metrics", True):
                metrics_path = getattr(self._avqa_args, "avqa_metrics_path", None) or os.path.join(out_dir, "avqa_metrics.jsonl")
                record = {
                    "global_step": int(self.state.global_step),
                    "epoch": None if self.state.epoch is None else float(self.state.epoch),
                    **metrics,
                }
                with open(metrics_path, "a") as f:
                    f.write(json.dumps(record) + "\n")

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            payload: List[Optional[Dict[str, float]]] = [metrics]
            torch.distributed.broadcast_object_list(payload, src=0)
            metrics = payload[0]

        if metrics is None:
            metrics = {f"{metric_key_prefix}_avqa_error": 1.0}

        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        if was_training:
            core_model.train()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        return metrics


class LynXSFTMusicAVQATrainer(_LynXPeftSavingTrainer):
    """
    Standard SFT Trainer + optional Music-AVQA validation evaluation (substring accuracy).

    Evaluation uses all modalities (video + audio + text) and prompts the model with:
      <image>
      {question}
    """

    def __init__(
        self,
        *args: Any,
        processor: Any,
        music_avqa_args: Any,
        audio_target_sr: int,
        audio_seconds: float,
        audio_clip_duration_s: float,
        audio_clip_stride_s: float,
        mel_bins: int,
        mel_target_length: int,
        mel_mean: float,
        mel_std: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._processor = processor
        self._tokenizer = processor.tokenizer
        self._music_avqa_args = music_avqa_args

        self._audio_target_sr = int(audio_target_sr)
        self._audio_seconds = float(audio_seconds)
        self._audio_clip_duration_s = float(audio_clip_duration_s)
        self._audio_clip_stride_s = float(audio_clip_stride_s)
        self._mel_bins = int(mel_bins)
        self._mel_target_length = int(mel_target_length)
        self._mel_mean = float(mel_mean)
        self._mel_std = float(mel_std)

        self._music_avqa_ann: Optional[List[Dict[str, Any]]] = None
        self._music_avqa_mapping: Optional[Dict[str, str]] = None

    def _load_music_avqa_assets(self) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
        if self._music_avqa_ann is None:
            ann = load_json(self._music_avqa_args.music_avqa_annotation_file)
            if not isinstance(ann, list):
                raise TypeError(f"Expected a list in {self._music_avqa_args.music_avqa_annotation_file}, got {type(ann)}")
            self._music_avqa_ann = ann

        if self._music_avqa_mapping is None:
            mapping: Dict[str, str] = {}
            if self._music_avqa_args.music_avqa_video_mapping:
                raw = load_json(self._music_avqa_args.music_avqa_video_mapping)
                if isinstance(raw, dict):
                    mapping = {str(k): str(v) for k, v in raw.items()}
            self._music_avqa_mapping = mapping

        return self._music_avqa_ann, self._music_avqa_mapping

    @torch.no_grad()
    def evaluate(  # type: ignore[override]
        self,
        eval_dataset: Optional[Any] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        if not getattr(self._music_avqa_args, "enable_music_avqa_eval", False):
            metrics = {f"{metric_key_prefix}_music_avqa_skipped": 1.0}
            self.log(metrics)
            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
            return metrics

        try:
            ann, mapping = self._load_music_avqa_assets()
        except Exception as exc:
            metrics = {f"{metric_key_prefix}_music_avqa_error": 1.0}
            if self.is_world_process_zero():
                print(f"[Music-AVQA Eval] Skipping: {exc}")
            self.log(metrics)
            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
            return metrics

        if getattr(self._music_avqa_args, "music_avqa_max_samples", None) is not None:
            ann = ann[: int(self._music_avqa_args.music_avqa_max_samples)]

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1

        wrapper = getattr(self.model, "module", self.model)
        core_model = getattr(wrapper, "base_model", wrapper)
        core_model = getattr(core_model, "module", core_model)

        was_training = core_model.training
        core_model.eval()

        device = next(core_model.parameters()).device
        if device.type == "cuda":
            model_dtype = torch.bfloat16 if self.args.bf16 else (torch.float16 if self.args.fp16 else next(core_model.parameters()).dtype)
        else:
            model_dtype = torch.float32

        local_preds: List[Dict[str, Any]] = []
        num_samples = len(ann)
        processed_global = 0
        pbar = None
        if rank == 0:
            pbar = tqdm(total=num_samples, desc="Music-AVQA", dynamic_ncols=True, leave=False)

        micro_batch = int(getattr(self._music_avqa_args, "music_avqa_eval_batch_size", 1) or 1)
        micro_batch = max(1, micro_batch)

        for start in range(rank * micro_batch, num_samples, world_size * micro_batch):
            batch_indices = list(range(start, min(start + micro_batch, num_samples)))
            batch_rows: List[Dict[str, Any]] = []
            batch_videos: List[torch.Tensor] = []
            batch_audios: List[Image.Image] = []

            for idx in batch_indices:
                item = ann[idx]
                if not isinstance(item, dict):
                    continue

                vid = item.get("video_id") or item.get("video") or item.get("video_name") or item.get("video_id_str")
                if not isinstance(vid, (str, int)):
                    continue
                vid = str(vid)

                q = str(item.get("question") or item.get("question_text") or "").strip()
                if not q:
                    continue

                answer = str(item.get("answer") or item.get("answer_text") or "").strip()
                if not answer:
                    continue

                messages = [{"role": "user", "content": f"<image>\n{q}"}]
                prefix_ids: List[int] = self._tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
                suffix_ids: List[int] = self._tokenizer.encode("", add_special_tokens=False)

                video_name = mapping.get(vid, vid)
                video_path = os.path.join(str(self._music_avqa_args.music_avqa_video_root), str(video_name))
                if not video_path.endswith(".mp4"):
                    video_path += ".mp4"

                if not os.path.exists(video_path):
                    local_preds.append({"id": item.get("id", idx), "video": vid, "pred": "", "answer": answer, "error": f"video_not_found: {video_path}"})
                    continue

                try:
                    vframes, aframes, info = load_video_frames_and_audio(
                        video_path,
                        num_frames=int(getattr(self._music_avqa_args, "music_avqa_num_frames", 60)),
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
                    )
                    audio_frames = [log_mel_to_pil_rgb(audio_clips[i, 0]) for i in range(audio_clips.shape[0])]

                    batch_rows.append(
                        {
                            "id": item.get("id", idx),
                            "video": vid,
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
                    local_preds.append({"id": item.get("id", idx), "video": vid, "pred": "", "answer": answer, "error": err})

            if not batch_rows:
                if pbar is not None:
                    remaining = num_samples - processed_global
                    step = min(world_size * len(batch_indices), remaining) if remaining > 0 else 0
                    processed_global += step
                    pbar.update(step)
                continue

            try:
                pv_video = self._processor.video_processor(batch_videos, return_tensors="pt")["pixel_values_videos"].to(device=device, dtype=model_dtype)
                pv_audio = self._processor.video_processor(batch_audios, return_tensors="pt")["pixel_values_videos"].to(device=device, dtype=model_dtype)

                orig_chunk = getattr(wrapper, "frame_chunk_size", None)
                try:
                    wrapper.frame_chunk_size = int(getattr(self._music_avqa_args, "music_avqa_frame_chunk_size", 4))
                    z_v, z_a = wrapper._encode_streams(pixel_values_videos=pv_video, audio_pixel_values_videos=pv_audio)
                finally:
                    if orig_chunk is not None:
                        wrapper.frame_chunk_size = orig_chunk
                z = torch.cat([z_v.to(dtype=model_dtype), z_a.to(dtype=model_dtype)], dim=1)

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

                with self.autocast_smart_context_manager():
                    gen = core_model.generate(
                        inputs_embeds=inputs_embeds_padded,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        max_new_tokens=int(getattr(self._music_avqa_args, "music_avqa_max_new_tokens", 8)),
                        min_new_tokens=int(getattr(self._music_avqa_args, "music_avqa_min_new_tokens", 1)),
                        do_sample=False,
                    )

                for i, row in enumerate(batch_rows):
                    text = self._tokenizer.decode(gen[i], skip_special_tokens=True)
                    local_preds.append({"id": row["id"], "video": row["video"], "pred": text, "answer": row["answer"]})

            except Exception as exc:
                msg = str(exc)
                if len(msg) > 300:
                    msg = msg[:300] + "..."
                err = f"music_avqa_eval_error_batch: {type(exc).__name__}: {msg}"
                for row in batch_rows:
                    local_preds.append({"id": row["id"], "video": row["video"], "pred": "", "answer": row["answer"], "error": err})

            if pbar is not None:
                remaining = num_samples - processed_global
                step = min(world_size * micro_batch, remaining) if remaining > 0 else 0
                processed_global += step
                pbar.update(step)

        if pbar is not None:
            pbar.close()

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            gathered: List[List[Dict[str, Any]]] = [None for _ in range(world_size)]  # type: ignore[list-item]
            torch.distributed.all_gather_object(gathered, local_preds)
            merged: List[Dict[str, Any]] = []
            for chunk in gathered:
                merged.extend(chunk)
        else:
            merged = local_preds

        metrics: Optional[Dict[str, float]] = None
        if rank == 0:
            total = 0
            correct = 0
            for pred in merged:
                if pred.get("error"):
                    continue
                total += 1
                if str(pred.get("answer", "")).strip().lower() in str(pred.get("pred", "")).strip().lower():
                    correct += 1
            acc = float(correct) / float(total) if total > 0 else 0.0

            out_dir = getattr(self._music_avqa_args, "music_avqa_predictions_dir", None) or os.path.join(self.args.output_dir, "music_avqa_eval")
            os.makedirs(out_dir, exist_ok=True)
            if getattr(self._music_avqa_args, "music_avqa_save_predictions", True):
                pred_path = os.path.join(out_dir, f"music_avqa_predictions_step{self.state.global_step}.json")
                with open(pred_path, "w") as f:
                    json.dump(merged, f)

            metrics = {
                f"{metric_key_prefix}_music_avqa_acc": acc,
                f"{metric_key_prefix}_music_avqa_correct": float(correct),
                f"{metric_key_prefix}_music_avqa_total": float(total),
            }

            if getattr(self._music_avqa_args, "music_avqa_save_metrics", True):
                metrics_path = getattr(self._music_avqa_args, "music_avqa_metrics_path", None) or os.path.join(out_dir, "music_avqa_metrics.jsonl")
                record = {
                    "global_step": int(self.state.global_step),
                    "epoch": None if self.state.epoch is None else float(self.state.epoch),
                    **metrics,
                }
                with open(metrics_path, "a") as f:
                    f.write(json.dumps(record) + "\n")

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            payload: List[Optional[Dict[str, float]]] = [metrics]
            torch.distributed.broadcast_object_list(payload, src=0)
            metrics = payload[0]

        if metrics is None:
            metrics = {f"{metric_key_prefix}_music_avqa_error": 1.0}

        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        if was_training:
            core_model.train()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        return metrics


class LynXSFTAVSDTrainer(_LynXPeftSavingTrainer):
    """
    Standard SFT Trainer + optional AVSD validation evaluation (COCO caption metrics).

    Evaluation uses all modalities (video + audio + text) and prompts the model with AVSD dialog history
    (multi-turn conversation) and expects the next assistant response as caption/answer.
    """

    def __init__(
        self,
        *args: Any,
        processor: Any,
        avsd_args: Any,
        audio_target_sr: int,
        audio_seconds: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._processor = processor
        self._tokenizer = processor.tokenizer
        self._avsd_args = avsd_args

        self._audio_target_sr = int(audio_target_sr)
        self._audio_seconds = float(audio_seconds)

        self._avsd_dialogs: Optional[List[Dict[str, Any]]] = None
        self._avsd_gt: Optional[Dict[str, List[Dict[str, str]]]] = None

    def _load_avsd_assets(self) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, str]]]]:
        if self._avsd_dialogs is None:
            payload = load_json(self._avsd_args.avsd_annotation_file)
            dialogs = payload.get("dialogs") if isinstance(payload, dict) else None
            if not isinstance(dialogs, list):
                raise TypeError(
                    f"Expected AVSD annotation_file to contain a dict with list under 'dialogs', got {type(dialogs)}"
                )
            self._avsd_dialogs = dialogs

        if self._avsd_gt is None:
            gt_payload = load_json(self._avsd_args.avsd_gt_file)
            gt: Dict[str, List[Dict[str, str]]] = {}
            if isinstance(gt_payload, dict):
                gt_items = gt_payload.get("annotations")
                if isinstance(gt_items, list):
                    for item in gt_items:
                        if not isinstance(item, dict):
                            continue
                        image_id = item.get("image_id")
                        caption = item.get("caption")
                        if image_id is None or caption is None:
                            continue
                        key = str(image_id)
                        gt.setdefault(key, []).append({"caption": str(caption)})
            self._avsd_gt = gt

        return self._avsd_dialogs, self._avsd_gt

    @torch.no_grad()
    def evaluate(  # type: ignore[override]
        self,
        eval_dataset: Optional[Any] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        if not getattr(self._avsd_args, "enable_avsd_eval", True):
            metrics = {f"{metric_key_prefix}_avsd_skipped": 1.0}
            self.log(metrics)
            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
            return metrics

        try:
            dialogs, gt = self._load_avsd_assets()
        except Exception as exc:
            metrics = {f"{metric_key_prefix}_avsd_error": 1.0}
            if self.is_world_process_zero():
                print(f"[AVSD Eval] Skipping: {exc}")
            self.log(metrics)
            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
            return metrics

        if getattr(self._avsd_args, "avsd_max_samples", None) is not None:
            dialogs = dialogs[: int(self._avsd_args.avsd_max_samples)]

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1

        wrapper = getattr(self.model, "module", self.model)
        core_model = getattr(wrapper, "base_model", wrapper)
        core_model = getattr(core_model, "module", core_model)

        was_training = core_model.training
        core_model.eval()

        device = next(core_model.parameters()).device
        if device.type == "cuda":
            model_dtype = torch.bfloat16 if self.args.bf16 else (torch.float16 if self.args.fp16 else next(core_model.parameters()).dtype)
        else:
            model_dtype = torch.float32

        max_new_tokens = int(getattr(self._avsd_args, "avsd_max_new_tokens", 64))
        min_new_tokens = int(getattr(self._avsd_args, "avsd_min_new_tokens", 1))

        local_preds: List[Dict[str, Any]] = []
        num_samples = len(dialogs)
        processed_global = 0
        pbar = None
        if rank == 0:
            pbar = tqdm(total=num_samples, desc="AVSD", dynamic_ncols=True, leave=False)

        micro_batch = int(getattr(self._avsd_args, "avsd_eval_batch_size", 1) or 1)
        micro_batch = max(1, micro_batch)

        for start in range(rank * micro_batch, num_samples, world_size * micro_batch):
            batch_indices = list(range(start, min(start + micro_batch, num_samples)))
            batch_rows: List[Dict[str, Any]] = []
            batch_videos: List[torch.Tensor] = []
            batch_audios: List[Image.Image] = []

            for idx in batch_indices:
                dialog = dialogs[idx]
                if not isinstance(dialog, dict):
                    continue
                vid = dialog.get("image_id") or dialog.get("video") or dialog.get("video_name") or dialog.get("video_id")
                if vid is None:
                    continue
                vid = str(vid)
                try:
                    messages = _build_avsd_messages(dialog.get("dialog"))
                except Exception as exc:
                    local_preds.append({"image_id": vid, "caption": "", "error": f"dialog_parse_error: {exc}"})
                    continue

                prefix_ids: List[int] = self._tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)

                video_path = os.path.join(str(self._avsd_args.avsd_video_root), f"{vid}.mp4")
                if not os.path.exists(video_path):
                    local_preds.append({"image_id": vid, "caption": "", "error": f"video_not_found: {video_path}"})
                    continue

                try:
                    vframes, aframes, info = load_video_frames_and_audio(
                        video_path,
                        num_frames=int(getattr(self._avsd_args, "avsd_num_frames", 20)),
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
                        clip_duration_s=2.0,
                        clip_stride_s=0.5,
                        num_mel_bins=128,
                        target_length=204,
                        mean=-4.268,
                        std=9.138,
                    )
                    audio_frames = [log_mel_to_pil_rgb(audio_clips[i, 0]) for i in range(audio_clips.shape[0])]

                    batch_rows.append({"image_id": vid, "prefix_ids": prefix_ids})
                    batch_videos.append(vframes)
                    batch_audios.append(audio_frames)
                except Exception as exc:
                    msg = str(exc)
                    if len(msg) > 300:
                        msg = msg[:300] + "..."
                    local_preds.append({"image_id": vid, "caption": "", "error": f"avsd_eval_error: {type(exc).__name__}: {msg}"})

            if not batch_rows:
                if pbar is not None:
                    remaining = num_samples - processed_global
                    step = min(world_size * len(batch_indices), remaining) if remaining > 0 else 0
                    processed_global += step
                    pbar.update(step)
                continue

            try:
                pv_video = self._processor.video_processor(batch_videos, return_tensors="pt")["pixel_values_videos"].to(device=device, dtype=model_dtype)
                pv_audio = self._processor.video_processor(batch_audios, return_tensors="pt")["pixel_values_videos"].to(device=device, dtype=model_dtype)

                orig_chunk = getattr(wrapper, "frame_chunk_size", None)
                try:
                    wrapper.frame_chunk_size = int(getattr(self._avsd_args, "avsd_frame_chunk_size", 4))
                    z_v, z_a = wrapper._encode_streams(pixel_values_videos=pv_video, audio_pixel_values_videos=pv_audio)
                finally:
                    if orig_chunk is not None:
                        wrapper.frame_chunk_size = orig_chunk
                z = torch.cat([z_v.to(dtype=model_dtype), z_a.to(dtype=model_dtype)], dim=1)

                embed = core_model.get_input_embeddings()
                inputs_list: List[torch.Tensor] = []
                seq_lens: List[int] = []
                for i, row in enumerate(batch_rows):
                    prefix_ids = row["prefix_ids"]
                    prefix_emb = embed(prefix_ids.to(device)).unsqueeze(0)
                    inputs_embeds = torch.cat([prefix_emb, z[i : i + 1]], dim=1)
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

                with self.autocast_smart_context_manager():
                    gen = core_model.generate(
                        inputs_embeds=inputs_embeds_padded,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        max_new_tokens=max_new_tokens,
                        min_new_tokens=min_new_tokens,
                        do_sample=False,
                    )

                for i, row in enumerate(batch_rows):
                    text = self._tokenizer.decode(gen[i], skip_special_tokens=True)
                    local_preds.append({"image_id": row["image_id"], "caption": text})

            except Exception as exc:
                msg = str(exc)
                if len(msg) > 300:
                    msg = msg[:300] + "..."
                err = f"avsd_eval_error_batch: {type(exc).__name__}: {msg}"
                for row in batch_rows:
                    local_preds.append({"image_id": row["image_id"], "caption": "", "error": err})

            if pbar is not None:
                remaining = num_samples - processed_global
                step = min(world_size * micro_batch, remaining) if remaining > 0 else 0
                processed_global += step
                pbar.update(step)

        if pbar is not None:
            pbar.close()

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            gathered: List[List[Dict[str, Any]]] = [None for _ in range(world_size)]  # type: ignore[list-item]
            torch.distributed.all_gather_object(gathered, local_preds)
            merged: List[Dict[str, Any]] = []
            for chunk in gathered:
                merged.extend(chunk)
        else:
            merged = local_preds

        metrics: Optional[Dict[str, float]] = None
        if rank == 0:
            try:
                from pycocoevalcap.eval import COCOEvalCap  # type: ignore[import-not-found]
            except Exception as exc:
                metrics = {f"{metric_key_prefix}_avsd_error": 1.0}
                if self.is_world_process_zero():
                    print(f"[AVSD Eval] Missing pycocoevalcap: {exc}")
            else:
                coco_res = {"annotations": []}
                for pred in merged:
                    if pred.get("error"):
                        continue
                    image_id = str(pred.get("image_id"))
                    coco_res["annotations"].append({"image_id": image_id, "caption": str(pred.get("caption", ""))})

                coco_gt = {"annotations": []}
                for image_id, captions in gt.items():
                    for cap in captions:
                        coco_gt["annotations"].append({"image_id": str(image_id), "caption": str(cap.get("caption", ""))})

                evaluator = COCOEvalCap(coco_gt, coco_res)  # type: ignore[call-arg]
                evaluator.evaluate()

                raw = getattr(evaluator, "eval", {}) or {}
                metrics = {f"{metric_key_prefix}_avsd_{k.lower()}": float(v) for k, v in raw.items() if isinstance(v, (int, float))}

                out_dir = getattr(self._avsd_args, "avsd_predictions_dir", None) or os.path.join(self.args.output_dir, "avsd_eval")
                os.makedirs(out_dir, exist_ok=True)
                if getattr(self._avsd_args, "avsd_save_predictions", True):
                    pred_path = os.path.join(out_dir, f"avsd_predictions_step{self.state.global_step}.json")
                    with open(pred_path, "w") as f:
                        json.dump(merged, f)

                if getattr(self._avsd_args, "avsd_save_metrics", True):
                    metrics_path = getattr(self._avsd_args, "avsd_metrics_path", None) or os.path.join(out_dir, "avsd_metrics.jsonl")
                    record = {
                        "global_step": int(self.state.global_step),
                        "epoch": None if self.state.epoch is None else float(self.state.epoch),
                        **metrics,
                    }
                    with open(metrics_path, "a") as f:
                        f.write(json.dumps(record) + "\n")

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            payload: List[Optional[Dict[str, float]]] = [metrics]
            torch.distributed.broadcast_object_list(payload, src=0)
            metrics = payload[0]

        if metrics is None:
            metrics = {f"{metric_key_prefix}_avsd_error": 1.0}

        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        if was_training:
            core_model.train()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        return metrics


class LynXSFTScanQATrainer(_LynXPeftSavingTrainer):
    def __init__(
        self,
        *args: Any,
        processor: Any,
        scanqa_args: Any,
        data_args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._processor = processor
        self._tokenizer = processor.tokenizer
        self._scanqa_args = scanqa_args
        self._data_args = data_args

        self._scanqa_questions: Optional[List[Dict[str, Any]]] = None
        self._pairs_cache: Dict[str, List[Tuple[str, str]]] = {}
        self._selected_cache: Dict[str, List[Tuple[str, str]]] = {}
        self._pose_cache: Dict[Tuple[str, int], Optional[np.ndarray]] = {}

    def _load_questions(self) -> List[Dict[str, Any]]:
        if self._scanqa_questions is not None:
            return self._scanqa_questions
        payload = load_json(self._scanqa_args.scanqa_question_file)
        if not isinstance(payload, list):
            raise TypeError(f"Expected a list in {self._scanqa_args.scanqa_question_file}, got {type(payload)}")
        self._scanqa_questions = payload
        return payload

    @torch.no_grad()
    def evaluate(  # type: ignore[override]
        self,
        eval_dataset: Optional[Any] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        if not getattr(self._scanqa_args, "enable_scanqa_eval", True):
            metrics = {f"{metric_key_prefix}_scanqa_skipped": 1.0}
            self.log(metrics)
            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
            return metrics

        try:
            questions = self._load_questions()
        except Exception as exc:
            metrics = {f"{metric_key_prefix}_scanqa_error": 1.0}
            if self.is_world_process_zero():
                print(f"[ScanQA Eval] Skipping: {exc}")
            self.log(metrics)
            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
            return metrics

        if getattr(self._scanqa_args, "scanqa_max_samples", None) is not None:
            questions = questions[: int(self._scanqa_args.scanqa_max_samples)]

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1

        wrapper = getattr(self.model, "module", self.model)
        core_model = getattr(wrapper, "base_model", wrapper)
        core_model = getattr(core_model, "module", core_model)

        was_training = core_model.training
        core_model.eval()

        device = next(core_model.parameters()).device
        if device.type == "cuda":
            model_dtype = torch.bfloat16 if self.args.bf16 else (torch.float16 if self.args.fp16 else next(core_model.parameters()).dtype)
        else:
            model_dtype = torch.float32

        _warn_optional_eval_deps(
            str(getattr(self._data_args, "dataset", "scanqa")),
            scanqa_args=self._scanqa_args,
            sqa3d_args=getattr(self, "_sqa3d_args", None),
        )

        local_preds: List[Dict[str, Any]] = []
        num_samples = len(questions)
        processed_global = 0
        pbar = None
        if rank == 0:
            pbar = tqdm(total=num_samples, desc="ScanQA", dynamic_ncols=True, leave=False)

        micro_batch = int(getattr(self._scanqa_args, "scanqa_eval_batch_size", 1) or 1)
        micro_batch = max(1, micro_batch)

        for start in range(rank * micro_batch, num_samples, world_size * micro_batch):
            batch_indices = list(range(start, min(start + micro_batch, num_samples)))
            batch_rows: List[Dict[str, Any]] = []
            batch_videos: List[torch.Tensor] = []
            batch_depths: List[torch.Tensor] = []

            for idx in batch_indices:
                item = questions[idx]
                if not isinstance(item, dict):
                    continue
                vid = item.get("scene_id") or item.get("scene") or item.get("video") or item.get("video_name") or item.get("video_id")
                if not isinstance(vid, str) or not vid:
                    continue
                q = str(item.get("question") or "").strip()
                if not q:
                    continue

                messages = [{"role": "user", "content": f"<image>\n{q}"}]
                prefix_ids: List[int] = self._tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
                suffix_ids: List[int] = self._tokenizer.encode("", add_special_tokens=False)

                try:
                    rgb_t, depth_t = _load_video_rgb_depth_tensors(
                        vid,
                        frames_root=str(getattr(self._data_args, "frames_root")),
                        color_subdir=str(getattr(self._data_args, "color_subdir")),
                        depth_subdir=str(getattr(self._data_args, "depth_subdir")),
                        pose_subdir=str(getattr(self._data_args, "pose_subdir", "pose")),
                        pose_matrix_type=str(getattr(self._data_args, "pose_matrix_type", "c2w")),
                        num_frames=int(getattr(self._scanqa_args, "scanqa_num_frames", getattr(self._data_args, "num_frames", 20))),
                        depth_encoding=str(getattr(self._data_args, "depth_encoding")),
                        depth_clip_min_mm=float(getattr(self._data_args, "depth_clip_min_mm")),
                        depth_clip_max_mm=float(getattr(self._data_args, "depth_clip_max_mm")),
                        depth_normals_frame=str(getattr(self._data_args, "depth_normals_frame")),
                        depth_intrinsics_filename=str(getattr(self._data_args, "depth_intrinsics_filename")),
                        depth_auto_scale_intrinsics=bool(getattr(self._data_args, "depth_auto_scale_intrinsics")),
                        pairs_cache=self._pairs_cache,
                        selected_cache=self._selected_cache,
                        pose_cache=self._pose_cache,
                        frame_sampling=str(getattr(self._data_args, "frame_sampling", "pose")),
                    )

                    batch_rows.append(
                        {
                            "question_id": item.get("question_id") or item.get("questionId") or item.get("id") or idx,
                            "video": vid,
                            "prefix_ids": prefix_ids,
                            "suffix_ids": suffix_ids,
                        }
                    )
                    batch_videos.append(rgb_t)
                    batch_depths.append(depth_t)
                except Exception as exc:
                    msg = str(exc)
                    if len(msg) > 300:
                        msg = msg[:300] + "..."
                    local_preds.append(
                        {
                            "question_id": item.get("question_id") or item.get("questionId") or item.get("id") or idx,
                            "scene_id": vid,
                            "answer": "",
                            "error": f"scanqa_eval_error: {type(exc).__name__}: {msg}",
                        }
                    )

            if not batch_rows:
                if pbar is not None:
                    remaining = num_samples - processed_global
                    step = min(world_size * len(batch_indices), remaining) if remaining > 0 else 0
                    processed_global += step
                    pbar.update(step)
                continue

            try:
                pv_video = self._processor.video_processor(batch_videos, return_tensors="pt")["pixel_values_videos"].to(device=device, dtype=model_dtype)
                pv_depth = self._processor.video_processor(batch_depths, return_tensors="pt")["pixel_values_videos"].to(device=device, dtype=model_dtype)

                orig_chunk = getattr(wrapper, "frame_chunk_size", None)
                orig_merge = getattr(wrapper, "merge_modality_frame_pairs", None)
                try:
                    wrapper.frame_chunk_size = int(getattr(self._data_args, "frame_chunk_size", 4))
                    wrapper.merge_modality_frame_pairs = bool(getattr(self._data_args, "depth_merge_frame_pairs", False))
                    z_v, z_d = wrapper._encode_streams(pixel_values_videos=pv_video, depth_pixel_values_videos=pv_depth)
                finally:
                    if orig_chunk is not None:
                        wrapper.frame_chunk_size = orig_chunk
                    if orig_merge is not None:
                        wrapper.merge_modality_frame_pairs = orig_merge

                z = torch.cat([z_v.to(dtype=model_dtype), z_d.to(dtype=model_dtype)], dim=1)

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

                with self.autocast_smart_context_manager():
                    gen = core_model.generate(
                        inputs_embeds=inputs_embeds_padded,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        max_new_tokens=int(getattr(self._scanqa_args, "scanqa_max_new_tokens", 32)),
                        min_new_tokens=int(getattr(self._scanqa_args, "scanqa_min_new_tokens", 1)),
                        do_sample=False,
                    )

                for i, row in enumerate(batch_rows):
                    text = self._tokenizer.decode(gen[i], skip_special_tokens=True)
                    local_preds.append({"question_id": row["question_id"], "scene_id": row["video"], "answer": text})

            except Exception as exc:
                msg = str(exc)
                if len(msg) > 300:
                    msg = msg[:300] + "..."
                err = f"scanqa_eval_error_batch: {type(exc).__name__}: {msg}"
                for row in batch_rows:
                    local_preds.append({"question_id": row["question_id"], "scene_id": row["video"], "answer": "", "error": err})

            if pbar is not None:
                remaining = num_samples - processed_global
                step = min(world_size * micro_batch, remaining) if remaining > 0 else 0
                processed_global += step
                pbar.update(step)

        if pbar is not None:
            pbar.close()

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            gathered: List[List[Dict[str, Any]]] = [None for _ in range(world_size)]  # type: ignore[list-item]
            torch.distributed.all_gather_object(gathered, local_preds)
            merged: List[Dict[str, Any]] = []
            for chunk in gathered:
                merged.extend(chunk)
        else:
            merged = local_preds

        metrics: Optional[Dict[str, float]] = None
        if rank == 0:
            try:
                evaluator = _load_module_from_file("scanqa_evaluator", os.path.join("tools", "3d", "scanqa", "scanqa_evaluator.py"))
                eval_fn = getattr(evaluator, "eval_caption", None)
            except Exception as exc:
                metrics = {f"{metric_key_prefix}_scanqa_error": 1.0}
                if self.is_world_process_zero():
                    print(f"[ScanQA Eval] Missing evaluator: {exc}")
                eval_fn = None

            if callable(eval_fn):
                pred_path = None
                out_dir = getattr(self._scanqa_args, "scanqa_predictions_dir", None) or os.path.join(self.args.output_dir, "scanqa_eval")
                os.makedirs(out_dir, exist_ok=True)
                if getattr(self._scanqa_args, "scanqa_save_predictions", True):
                    pred_path = os.path.join(out_dir, f"scanqa_predictions_step{self.state.global_step}.json")
                    with open(pred_path, "w") as f:
                        json.dump(merged, f)

                try:
                    raw_metrics = eval_fn(
                        pred_path or merged,
                        self._scanqa_args.scanqa_gt_file,
                    )
                except Exception as exc:
                    metrics = {f"{metric_key_prefix}_scanqa_error": 1.0}
                    if self.is_world_process_zero():
                        print(f"[ScanQA Eval] eval_caption failed: {exc}")
                else:
                    metrics = {}
                    for k, v in (raw_metrics or {}).items():
                        metrics[f"{metric_key_prefix}_scanqa_{_sanitize_metric_key('[scanqa]', k)}"] = float(v)

                    if getattr(self._scanqa_args, "scanqa_save_metrics", True):
                        metrics_path = getattr(self._scanqa_args, "scanqa_metrics_path", None) or os.path.join(out_dir, "scanqa_metrics.jsonl")
                        record = {
                            "global_step": int(self.state.global_step),
                            "epoch": None if self.state.epoch is None else float(self.state.epoch),
                            **metrics,
                        }
                        with open(metrics_path, "a") as f:
                            f.write(json.dumps(record) + "\n")

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            payload: List[Optional[Dict[str, float]]] = [metrics]
            torch.distributed.broadcast_object_list(payload, src=0)
            metrics = payload[0]

        if metrics is None:
            metrics = {f"{metric_key_prefix}_scanqa_error": 1.0}

        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        if was_training:
            core_model.train()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        return metrics


class LynXSFTSQA3DTrainer(_LynXPeftSavingTrainer):
    def __init__(
        self,
        *args: Any,
        processor: Any,
        sqa3d_args: Any,
        data_args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._processor = processor
        self._tokenizer = processor.tokenizer
        self._sqa3d_args = sqa3d_args
        self._data_args = data_args

        self._sqa3d_questions: Optional[List[Dict[str, Any]]] = None
        self._pairs_cache: Dict[str, List[Tuple[str, str]]] = {}
        self._selected_cache: Dict[str, List[Tuple[str, str]]] = {}
        self._pose_cache: Dict[Tuple[str, int], Optional[np.ndarray]] = {}

    def _load_questions(self) -> List[Dict[str, Any]]:
        if self._sqa3d_questions is not None:
            return self._sqa3d_questions
        payload = load_json(self._sqa3d_args.sqa3d_question_file)
        if not isinstance(payload, list):
            raise TypeError(f"Expected a list in {self._sqa3d_args.sqa3d_question_file}, got {type(payload)}")
        self._sqa3d_questions = payload
        return payload

    @torch.no_grad()
    def evaluate(  # type: ignore[override]
        self,
        eval_dataset: Optional[Any] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        if not getattr(self._sqa3d_args, "enable_sqa3d_eval", True):
            metrics = {f"{metric_key_prefix}_sqa3d_skipped": 1.0}
            self.log(metrics)
            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
            return metrics

        try:
            questions = self._load_questions()
        except Exception as exc:
            metrics = {f"{metric_key_prefix}_sqa3d_error": 1.0}
            if self.is_world_process_zero():
                print(f"[SQA3D Eval] Skipping: {exc}")
            self.log(metrics)
            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
            return metrics

        if getattr(self._sqa3d_args, "sqa3d_max_samples", None) is not None:
            questions = questions[: int(self._sqa3d_args.sqa3d_max_samples)]

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1

        wrapper = getattr(self.model, "module", self.model)
        core_model = getattr(wrapper, "base_model", wrapper)
        core_model = getattr(core_model, "module", core_model)

        was_training = core_model.training
        core_model.eval()

        device = next(core_model.parameters()).device
        if device.type == "cuda":
            model_dtype = torch.bfloat16 if self.args.bf16 else (torch.float16 if self.args.fp16 else next(core_model.parameters()).dtype)
        else:
            model_dtype = torch.float32

        local_preds: List[Dict[str, Any]] = []
        num_samples = len(questions)
        processed_global = 0
        pbar = None
        if rank == 0:
            pbar = tqdm(total=num_samples, desc="SQA3D", dynamic_ncols=True, leave=False)

        micro_batch = int(getattr(self._sqa3d_args, "sqa3d_eval_batch_size", 1) or 1)
        micro_batch = max(1, micro_batch)

        for start in range(rank * micro_batch, num_samples, world_size * micro_batch):
            batch_indices = list(range(start, min(start + micro_batch, num_samples)))
            batch_rows: List[Dict[str, Any]] = []
            batch_videos: List[torch.Tensor] = []
            batch_depths: List[torch.Tensor] = []

            for idx in batch_indices:
                item = questions[idx]
                if not isinstance(item, dict):
                    continue
                vid = item.get("scene_id") or item.get("scene") or item.get("video") or item.get("video_name") or item.get("video_id")
                if not isinstance(vid, str) or not vid:
                    continue

                q = str(item.get("question") or "").strip()
                if not q:
                    continue

                choices = item.get("choices") or item.get("multi_choice") or item.get("options")
                if isinstance(choices, dict):
                    options = [choices.get("A"), choices.get("B"), choices.get("C"), choices.get("D")]
                elif isinstance(choices, list):
                    options = choices
                else:
                    options = None

                if isinstance(options, list) and len(options) >= 4:
                    prompt = (
                        "Answer the following multiple-choice question based on the 3D scene.\n"
                        f"Q: {q}\n"
                        f"A. {options[0]}\nB. {options[1]}\nC. {options[2]}\nD. {options[3]}\n"
                        "Respond with the option's letter."
                    )
                else:
                    prompt = q

                messages = [{"role": "user", "content": f"<image>\n{prompt}"}]
                prefix_ids: List[int] = self._tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
                suffix_ids: List[int] = self._tokenizer.encode("", add_special_tokens=False)

                try:
                    rgb_t, depth_t = _load_video_rgb_depth_tensors(
                        vid,
                        frames_root=str(getattr(self._data_args, "frames_root")),
                        color_subdir=str(getattr(self._data_args, "color_subdir")),
                        depth_subdir=str(getattr(self._data_args, "depth_subdir")),
                        pose_subdir=str(getattr(self._data_args, "pose_subdir", "pose")),
                        pose_matrix_type=str(getattr(self._data_args, "pose_matrix_type", "c2w")),
                        num_frames=int(getattr(self._sqa3d_args, "sqa3d_num_frames", getattr(self._data_args, "num_frames", 20))),
                        depth_encoding=str(getattr(self._data_args, "depth_encoding")),
                        depth_clip_min_mm=float(getattr(self._data_args, "depth_clip_min_mm")),
                        depth_clip_max_mm=float(getattr(self._data_args, "depth_clip_max_mm")),
                        depth_normals_frame=str(getattr(self._data_args, "depth_normals_frame")),
                        depth_intrinsics_filename=str(getattr(self._data_args, "depth_intrinsics_filename")),
                        depth_auto_scale_intrinsics=bool(getattr(self._data_args, "depth_auto_scale_intrinsics")),
                        pairs_cache=self._pairs_cache,
                        selected_cache=self._selected_cache,
                        pose_cache=self._pose_cache,
                        frame_sampling=str(getattr(self._data_args, "frame_sampling", "pose")),
                    )

                    batch_rows.append(
                        {
                            "question_id": item.get("question_id") or item.get("questionId") or item.get("id") or idx,
                            "video": vid,
                            "answer": item.get("answer") or item.get("answer_text") or "",
                            "prefix_ids": prefix_ids,
                            "suffix_ids": suffix_ids,
                        }
                    )
                    batch_videos.append(rgb_t)
                    batch_depths.append(depth_t)
                except Exception as exc:
                    msg = str(exc)
                    if len(msg) > 300:
                        msg = msg[:300] + "..."
                    local_preds.append(
                        {
                            "question_id": item.get("question_id") or item.get("questionId") or item.get("id") or idx,
                            "scene_id": vid,
                            "answer": "",
                            "gt": item.get("answer") or item.get("answer_text") or "",
                            "error": f"sqa3d_eval_error: {type(exc).__name__}: {msg}",
                        }
                    )

            if not batch_rows:
                if pbar is not None:
                    remaining = num_samples - processed_global
                    step = min(world_size * len(batch_indices), remaining) if remaining > 0 else 0
                    processed_global += step
                    pbar.update(step)
                continue

            try:
                pv_video = self._processor.video_processor(batch_videos, return_tensors="pt")["pixel_values_videos"].to(device=device, dtype=model_dtype)
                pv_depth = self._processor.video_processor(batch_depths, return_tensors="pt")["pixel_values_videos"].to(device=device, dtype=model_dtype)

                orig_chunk = getattr(wrapper, "frame_chunk_size", None)
                orig_merge = getattr(wrapper, "merge_modality_frame_pairs", None)
                try:
                    wrapper.frame_chunk_size = int(getattr(self._data_args, "frame_chunk_size", 4))
                    wrapper.merge_modality_frame_pairs = bool(getattr(self._data_args, "depth_merge_frame_pairs", False))
                    z_v, z_d = wrapper._encode_streams(pixel_values_videos=pv_video, depth_pixel_values_videos=pv_depth)
                finally:
                    if orig_chunk is not None:
                        wrapper.frame_chunk_size = orig_chunk
                    if orig_merge is not None:
                        wrapper.merge_modality_frame_pairs = orig_merge

                z = torch.cat([z_v.to(dtype=model_dtype), z_d.to(dtype=model_dtype)], dim=1)

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

                with self.autocast_smart_context_manager():
                    gen = core_model.generate(
                        inputs_embeds=inputs_embeds_padded,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        max_new_tokens=int(getattr(self._sqa3d_args, "sqa3d_max_new_tokens", 32)),
                        min_new_tokens=1,
                        do_sample=False,
                    )

                for i, row in enumerate(batch_rows):
                    text = self._tokenizer.decode(gen[i], skip_special_tokens=True)
                    local_preds.append({"question_id": row["question_id"], "scene_id": row["video"], "answer": text, "gt": row.get("answer", "")})

            except Exception as exc:
                msg = str(exc)
                if len(msg) > 300:
                    msg = msg[:300] + "..."
                err = f"sqa3d_eval_error_batch: {type(exc).__name__}: {msg}"
                for row in batch_rows:
                    local_preds.append({"question_id": row["question_id"], "scene_id": row["video"], "answer": "", "gt": row.get("answer", ""), "error": err})

            if pbar is not None:
                remaining = num_samples - processed_global
                step = min(world_size * micro_batch, remaining) if remaining > 0 else 0
                processed_global += step
                pbar.update(step)

        if pbar is not None:
            pbar.close()

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            gathered: List[List[Dict[str, Any]]] = [None for _ in range(world_size)]  # type: ignore[list-item]
            torch.distributed.all_gather_object(gathered, local_preds)
            merged: List[Dict[str, Any]] = []
            for chunk in gathered:
                merged.extend(chunk)
        else:
            merged = local_preds

        metrics: Optional[Dict[str, float]] = None
        if rank == 0:
            correct = 0
            total = 0
            for pred in merged:
                if pred.get("error"):
                    continue
                total += 1
                if str(pred.get("gt", "")).strip().lower() in str(pred.get("answer", "")).strip().lower():
                    correct += 1
            acc = float(correct) / float(total) if total > 0 else 0.0
            metrics = {
                f"{metric_key_prefix}_sqa3d_acc": acc,
                f"{metric_key_prefix}_sqa3d_correct": float(correct),
                f"{metric_key_prefix}_sqa3d_total": float(total),
            }

            out_dir = getattr(self._sqa3d_args, "sqa3d_predictions_dir", None) or os.path.join(self.args.output_dir, "sqa3d_eval")
            os.makedirs(out_dir, exist_ok=True)
            if getattr(self._sqa3d_args, "sqa3d_save_predictions", True):
                pred_path = os.path.join(out_dir, f"sqa3d_predictions_step{self.state.global_step}.json")
                with open(pred_path, "w") as f:
                    json.dump(merged, f)

            if getattr(self._sqa3d_args, "sqa3d_save_metrics", True):
                metrics_path = getattr(self._sqa3d_args, "sqa3d_metrics_path", None) or os.path.join(out_dir, "sqa3d_metrics.jsonl")
                record = {
                    "global_step": int(self.state.global_step),
                    "epoch": None if self.state.epoch is None else float(self.state.epoch),
                    **metrics,
                }
                with open(metrics_path, "a") as f:
                    f.write(json.dumps(record) + "\n")

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            payload: List[Optional[Dict[str, float]]] = [metrics]
            torch.distributed.broadcast_object_list(payload, src=0)
            metrics = payload[0]

        if metrics is None:
            metrics = {f"{metric_key_prefix}_sqa3d_error": 1.0}

        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        if was_training:
            core_model.train()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        return metrics
