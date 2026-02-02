import logging
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from packaging import version
from tqdm import tqdm
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

from libs.model.lynx_onevision import LynXOnevisionWrapper
from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lynx_utils import encode_llava_onevision_video_tokens, load_video_frames_and_audio, tile_pixel_values_videos_spatially


eval_logger = logging.getLogger("lmms-eval")

if version.parse(torch.__version__) >= version.parse("2.1.2"):
    best_fit_attn_implementation = "sdpa"
else:
    best_fit_attn_implementation = "eager"


def _maybe_disable_adapter(model: Any):
    peft_model = getattr(model, "module", model)
    disable = getattr(peft_model, "disable_adapter", None)
    if callable(disable):
        return peft_model.disable_adapter()
    return None


def _maybe_load_adapter(model: Any, *, path: Optional[str], adapter_name: str) -> Tuple[Any, Optional[str]]:
    if not path:
        return model, None
    try:
        from peft import PeftModel  # type: ignore[import-not-found]
    except Exception as exc:
        raise RuntimeError("peft is required to load adapter checkpoints for LynX evaluation.") from exc

    if isinstance(model, PeftModel):
        loader = getattr(model, "load_adapter", None)
        if not callable(loader):
            raise AttributeError("PeftModel does not expose load_adapter().")
        loader(path, adapter_name=adapter_name, is_trainable=False)
        return model, adapter_name

    model = PeftModel.from_pretrained(model, path, adapter_name=adapter_name, is_trainable=False)
    return model, adapter_name


@register_model("lynx_fastvideo_onevision")
class LynXFastVideoOnevision(lmms):
    """
    LynX Fast/Slow Video evaluation model for lmms-eval.

    - Loads a HF LLaVA-OneVision model (Qwen2-OV HF format).
    - Optionally loads:
      - vision adapter (LoRA) for the fast stream
      - llm adapter (LoRA) for text generation
    - Builds slow stream tokens with adapters disabled.
    - Builds fast stream tokens with vision adapter enabled, using tiled-frame inputs.
    """

    def __init__(
        self,
        pretrained: str = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
        *,
        vision_adapter_path: Optional[str] = None,
        llm_adapter_path: Optional[str] = None,
        vision_adapter_name: str = "vision",
        llm_adapter_name: str = "llm",
        slow_num_frames: int = 20,
        fast_num_frames: Optional[int] = None,
        fast_frame_multiplier: int = 4,
        fast_tile_size: int = 4,
        frame_chunk_size: int = 4,
        pool_fast_tokens: bool = False,
        device: Optional[str] = None,
        batch_size: Optional[Union[int, str]] = 1,
        attn_implementation: Optional[str] = best_fit_attn_implementation,
        device_map: Optional[str] = "cuda:0",
        use_cache: bool = True,
        local_files_only: bool = True,
        cache_dir: Optional[str] = "/mnt/hdd1/",
        video_backend: str = "torchvision",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device or "cuda:0")
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        self.batch_size_per_gpu = int(batch_size) if isinstance(batch_size, int) else 1
        self.use_cache = bool(use_cache)
        self._video_backend = str(video_backend)

        self._slow_num_frames = int(slow_num_frames)
        if fast_num_frames is None:
            fast_num_frames = int(fast_frame_multiplier) * int(slow_num_frames)
        self._fast_num_frames = int(fast_num_frames)
        self._fast_tile_size = int(fast_tile_size)
        self._frame_chunk_size = int(frame_chunk_size)
        self._pool_fast_tokens = bool(pool_fast_tokens)

        self._processor = AutoProcessor.from_pretrained(
            pretrained,
            cache_dir=cache_dir,
            local_files_only=bool(local_files_only),
        )
        self._tokenizer = self._processor.tokenizer
        self._video_processor = self._processor.video_processor

        torch_dtype = torch.bfloat16 if self._device.type == "cuda" else torch.float32
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            pretrained,
            cache_dir=cache_dir,
            local_files_only=bool(local_files_only),
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            device_map=self.device_map,
            attn_implementation=attn_implementation,
        )

        model, loaded_vision_name = _maybe_load_adapter(model, path=vision_adapter_path, adapter_name=str(vision_adapter_name))
        model, loaded_llm_name = _maybe_load_adapter(model, path=llm_adapter_path, adapter_name=str(llm_adapter_name))

        self._vision_adapter_name = loaded_vision_name
        self._llm_adapter_name = loaded_llm_name

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
                DistributedType.DEEPSPEED,
            ], "Unsupported distributed type for LynX evaluation."

            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs_ds = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs_ds)
                eval_logger.info("Detected DeepSpeed in Accelerate; run `accelerate config` and set ZeRO stage to 0 for eval.")

            self._model = accelerator.prepare_model(model, evaluation_mode=True)
            self.accelerator = accelerator
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._model = model
            self._rank = 0
            self._world_size = 1

        self._config = getattr(self.model, "config", None)
        self._max_length = int(getattr(self._config, "max_position_embeddings", 0) or 0)

        placeholder_id = self._tokenizer.convert_tokens_to_ids("<image>")
        if placeholder_id is None or placeholder_id == getattr(self._tokenizer, "unk_token_id", None):
            raise ValueError("Tokenizer does not recognize '<image>' as a single token.")
        self._placeholder_token_id = int(placeholder_id)

        # We use the wrapper only for its adapter toggling semantics.
        self._wrapper = LynXOnevisionWrapper(
            self.model,
            placeholder_token_id=self._placeholder_token_id,
            pool_video_tokens=False,
            frame_chunk_size=self._frame_chunk_size,
            vision_adapter_name=self._vision_adapter_name,
            llm_adapter_name=self._llm_adapter_name,
            detach_modality_tokens=True,
        )

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        return self._model

    @property
    def device(self):
        return self._device

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError

    def _build_prompt_ids(self, context: str) -> torch.Tensor:
        # Force the single <image> placeholder for LLaVA-OneVision models.
        user_text = f"<image>\n{context}"
        messages = [{"role": "user", "content": user_text}]
        ids: List[int] = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        if not isinstance(ids, list):
            raise TypeError("Tokenizer chat template did not return token ids as a list.")
        return torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)

    def _encode_two_stream_tokens(self, *, pv_slow: torch.Tensor, pv_fast: torch.Tensor) -> torch.Tensor:
        core = getattr(self.model, "module", self.model)

        with torch.no_grad():
            # Slow stream: adapters disabled.
            ctx = _maybe_disable_adapter(core)
            if ctx is not None:
                with ctx:
                    z_slow, z_fast = self._wrapper._encode_streams(pixel_values_videos=pv_slow, fast_pixel_values_videos=pv_fast)
            else:
                # If adapters cannot be toggled, fall back to explicit encoding with current adapter state.
                z_slow = encode_llava_onevision_video_tokens(
                    core,
                    pixel_values_videos=pv_slow,
                    vision_feature_layer=core.config.vision_feature_layer,
                    vision_feature_select_strategy=core.config.vision_feature_select_strategy,
                    frame_chunk_size=self._frame_chunk_size,
                )
                z_fast = encode_llava_onevision_video_tokens(
                    core,
                    pixel_values_videos=pv_fast,
                    vision_feature_layer=core.config.vision_feature_layer,
                    vision_feature_select_strategy=core.config.vision_feature_select_strategy,
                    frame_chunk_size=self._frame_chunk_size,
                )

        if self._pool_fast_tokens:
            # Pool fast tokens spatially (keeps per-frame tokens small) by reusing wrapper's helper.
            from libs.model.lynx_onevision import spatial_mean_pool_video_tokens

            z_fast = spatial_mean_pool_video_tokens(z_fast, frames=int(pv_fast.shape[1]))

        return torch.cat([z_slow, z_fast], dim=1)

    def _prepare_inputs_embeds(self, *, input_ids: torch.Tensor, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Single-sample for simplicity.
        if input_ids.shape[0] != 1:
            raise ValueError("Expected batch size 1 in _prepare_inputs_embeds.")

        ids = input_ids[0]
        pos = (ids == self._placeholder_token_id).nonzero()
        if pos.numel() != 1:
            raise ValueError(f"Expected exactly 1 <image> token, got {pos.numel()}")
        ph = int(pos.item())

        prefix_ids = ids[:ph]
        suffix_ids = ids[ph + 1 :]

        embed = self.model.get_input_embeddings()
        prefix_emb = embed(prefix_ids).to(dtype=tokens.dtype)
        suffix_emb = embed(suffix_ids).to(dtype=tokens.dtype)

        inputs_embeds = torch.cat([prefix_emb.unsqueeze(0), tokens, suffix_emb.unsqueeze(0)], dim=1)
        attention_mask = torch.ones((1, inputs_embeds.shape[1]), device=inputs_embeds.device, dtype=torch.long)
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device, dtype=torch.long).unsqueeze(0)
        return inputs_embeds, attention_mask, position_ids

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res: List[str] = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0], add_special_tokens=False)
            return -len(toks), x[0]

        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=1, batch_fn=None)  # force per-sample decode to keep it robust

        num_iters = len(requests)
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        for chunk in chunks:
            (context, gen_kwargs, doc_to_visual, doc_id, task, split) = chunk[0]
            doc = self.task_dict[task][split][doc_id]
            visuals = doc_to_visual(doc)
            if not visuals:
                res.append("")
                pbar.update(1)
                continue
            video_path = visuals[0]

            gen_kwargs = dict(gen_kwargs)
            if "until" in gen_kwargs:
                gen_kwargs.pop("until")
            if "image_aspect_ratio" in gen_kwargs:
                gen_kwargs.pop("image_aspect_ratio")

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 64
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0.0
            if "do_sample" not in gen_kwargs:
                gen_kwargs["do_sample"] = False
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = 1.0
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            try:
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

                pv_slow = self._video_processor([slow_frames], return_tensors="pt")["pixel_values_videos"].to(device=self.device)
                pv_fast = self._video_processor([fast_frames], return_tensors="pt")["pixel_values_videos"].to(device=self.device)
                pv_fast = tile_pixel_values_videos_spatially(pv_fast, tile_size=self._fast_tile_size)

                input_ids = self._build_prompt_ids(context)
                tokens = self._encode_two_stream_tokens(pv_slow=pv_slow, pv_fast=pv_fast)
                inputs_embeds, attention_mask, position_ids = self._prepare_inputs_embeds(input_ids=input_ids, tokens=tokens)

                pad_id = int(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
                with torch.inference_mode():
                    out = self.model.generate(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        pad_token_id=pad_id,
                        use_cache=self.use_cache,
                        **gen_kwargs,
                    )
                text = self.tokenizer.decode(out[0], skip_special_tokens=True).strip()
            except Exception as exc:
                eval_logger.error(f"[lynx_fastvideo_onevision] failed on doc_id={doc_id}: {exc}")
                text = ""

            res.append(text)
            pbar.update(1)

        pbar.close()
        return re_ords.get_original(res)

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        raise NotImplementedError
