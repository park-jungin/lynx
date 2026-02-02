import ast
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset
from transformers import AutoProcessor, HfArgumentParser, LlavaOnevisionForConditionalGeneration, TrainingArguments, set_seed
from transformers.trainer_utils import IntervalStrategy

from libs.dataset.lynx_sft import (
    AVQAInstructionCollator,
    AVQAInstructionDataset,
    AVSDInstructionDataset,
    EgoExoInstructionCollator,
    EgoExoProficiencyInstructionDataset,
    LlavaVideo178KFastInstructionCollator,
    LlavaVideo178KFastInstructionDataset,
    ThreeDInstructionCollator,
    ThreeDInstructionDataset,
)
from libs.model.lynx_onevision import LynXOnevisionWrapper
from libs.utils.hf_utils import maybe_patch_torch_autograd_graph_for_deepspeed, resolve_local_model_path
from libs.utils.lynx_trainer import (
    _LynXPeftSavingTrainer,
    LynXSFTAVQATrainer,
    LynXSFTAVSDTrainer,
    LynXSFTMusicAVQATrainer,
    LynXSFTScanQATrainer,
    LynXSFTSQA3DTrainer,
)
from lynx_utils import resolve_attn_implementation


def _split_csv(value: Optional[str]) -> List[str]:
    if value is None:
        return []
    items = [v.strip() for v in str(value).split(",")]
    return [v for v in items if v]


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


@dataclass
class LynXModelArguments:
    model_name_or_path: str = field(default="llava-hf/llava-onevision-qwen2-7b-ov-hf")
    cache_dir: Optional[str] = field(default='/mnt/hdd1/')
    local_files_only: bool = field(default=True)
    use_flash_attn: bool = field(default=True)

    vision_adapter_path: Optional[str] = field(default=None, metadata={"help": "Pretrained LynX vision-LoRA adapter (Stage-1/2)."})
    vision_adapter_name: str = field(default="vision")


@dataclass
class LynXSFTDataArguments:
    dataset: str = field(default="avqa", metadata={"help": "SFT dataset name: avqa, avsd, or music_avqa."})
    train_instruct_file: str = field(default="./data/video_instruction_tuning/avqa/avqa_train_qa_instruct.json")
    video_root: str = field(default="./data/video_instruction_tuning/avqa/videos")
    video_mapping: Optional[str] = field(default="./data/video_instruction_tuning/avqa/avqa_from_vid_to_video_name.json")
    max_samples: Optional[int] = field(default=None)

    num_frames: int = field(default=32)
    frame_chunk_size: int = field(default=4)
    pool_video_tokens: bool = field(default=False)

    audio_target_sr: int = field(default=16000)
    audio_seconds: float = field(default=8.0)
    audio_clip_duration_s: float = field(default=2.0)
    audio_clip_stride_s: float = field(default=0.5)
    mel_bins: int = field(default=128)
    mel_target_length: int = field(default=204)
    mel_mean: float = field(default=-4.268)
    mel_std: float = field(default=9.138)


@dataclass
class EgoExoSFTDataArguments:
    train_instruct_file: str = field(default="./data/video_instruction_tuning/egoexo/proficiency_demonstrator_train_instruct.json")
    video_root: str = field(default="./data/video_instruction_tuning/egoexo")
    take_id_to_video_mapping: str = field(default="./data/video_instruction_tuning/egoexo/from_take_id_to_video.json")
    max_samples: Optional[int] = field(default=None)
    num_frames: int = field(default=20)
    frame_chunk_size: int = field(default=4)
    pool_video_tokens: bool = field(default=False)
    video_backend: str = field(default="torchvision")


@dataclass
class LynXFastVideoSFTDataArguments:
    train_instruct_files: str = field(default="", metadata={"help": "Comma-separated list of LLaVA-Video-178K instruct json files."})
    video_roots: str = field(
        default="",
        metadata={"help": "Comma-separated list of video roots aligned with train_instruct_files (root is the parent of academic_source/liwei_youtube_videos)."},
    )
    max_samples: Optional[int] = field(default=None)

    slow_num_frames: int = field(default=32)
    frame_chunk_size: int = field(default=4)
    pool_video_tokens: bool = field(default=False)

    fast_num_frames: Optional[int] = field(
        default=None, metadata={"help": "Frames sampled from each video for the fast stream before tiling. Defaults to 4 * slow_num_frames."}
    )
    fast_frame_multiplier: int = field(default=4)
    fast_tile_size: int = field(default=4)


@dataclass
class LynX3DSFTDataArguments:
    dataset: str = field(default="scanqa", metadata={"help": "SFT dataset name: scanqa or sqa."})
    train_instruct_file: str = field(default="./data/video_instruction_tuning/scannet/scanqa_train_instruct.json")
    frames_root: str = field(default="./data/video_instruction_tuning/3d/frames_square")
    color_subdir: str = field(default="color")
    depth_subdir: str = field(default="depth")
    pose_subdir: str = field(default="pose")
    pose_matrix_type: str = field(default="c2w", metadata={"help": "Pose matrix type in ./pose/*.txt: c2w (ScanNet default) or w2c."})
    max_samples: Optional[int] = field(default=None)

    frame_sampling: str = field(default="pose", metadata={"help": "Frame sampling strategy: uniform or pose (pose-aware for ScanNet)."})

    num_frames: int = field(default=20)
    frame_chunk_size: int = field(default=4)
    pool_video_tokens: bool = field(default=False)

    depth_clip_min_mm: float = field(default=200.0)
    depth_clip_max_mm: float = field(default=10000.0)
    depth_encoding: str = field(default="turbo", metadata={"help": "Depth preprocessing: gray, turbo, normals, turbo+normals."})
    depth_normals_frame: str = field(default="camera", metadata={"help": "Normal map coordinate frame: camera or world (requires pose)."})
    depth_intrinsics_filename: str = field(default="intrinsic_depth.txt")
    depth_auto_scale_intrinsics: bool = field(default=True)
    depth_merge_frame_pairs: bool = field(
        default=True,
        metadata={
            "help": (
                "If depth_encoding=turbo+normals, encode 2 frames per timestep (turbo + normals) but merge them at the token level "
                "so the LLM sees ~T timesteps instead of 2T (helps avoid OOM)."
            )
        },
    )


@dataclass
class LynXScanQAEvalArguments:
    enable_scanqa_eval: bool = field(default=True)
    scanqa_question_file: str = field(default="./data/video_instruction_tuning/scannet/llava-3d-scanqa_val_question.json")
    scanqa_gt_file: str = field(default="./data/video_instruction_tuning/scannet/llava3d_scanqa_val_answer.json")
    scanqa_num_frames: int = field(default=20)
    scanqa_max_samples: Optional[int] = field(default=None)
    scanqa_eval_before_train: bool = field(default=False)
    scanqa_eval_batch_size: int = field(default=1, metadata={"help": "Micro-batch size per GPU for ScanQA evaluation."})
    scanqa_max_new_tokens: int = field(default=32)
    scanqa_min_new_tokens: int = field(default=1)
    scanqa_save_predictions: bool = field(default=True)
    scanqa_predictions_dir: Optional[str] = field(default=None)
    scanqa_save_metrics: bool = field(default=True)
    scanqa_metrics_path: Optional[str] = field(default=None)


@dataclass
class LynXSQA3DEvalArguments:
    enable_sqa3d_eval: bool = field(default=True)
    sqa3d_question_file: str = field(default="./data/video_instruction_tuning/sqa/llava3d_sqa3d_test_question.json")
    sqa3d_gt_file: str = field(default="./data/video_instruction_tuning/sqa/llava3d_sqa3d_test_answer.json")
    sqa3d_num_frames: int = field(default=20)
    sqa3d_max_samples: Optional[int] = field(default=None)
    sqa3d_eval_before_train: bool = field(default=False)
    sqa3d_eval_batch_size: int = field(default=1, metadata={"help": "Micro-batch size per GPU for SQA3D evaluation."})
    sqa3d_max_new_tokens: int = field(default=32)
    sqa3d_min_new_tokens: int = field(default=1)
    sqa3d_save_predictions: bool = field(default=True)
    sqa3d_predictions_dir: Optional[str] = field(default=None)
    sqa3d_save_metrics: bool = field(default=True)
    sqa3d_metrics_path: Optional[str] = field(default=None)


@dataclass
class LlmLoraArguments:
    enable_llm_lora: bool = field(default=True)
    llm_adapter_name: str = field(default="llm")
    llm_lora_r: int = field(default=64)
    llm_lora_alpha: int = field(default=16)
    llm_lora_dropout: float = field(default=0.05)
    llm_lora_bias: str = field(default="none")


@dataclass
class LynXAVQAEvalArguments:
    enable_avqa_eval: bool = field(default=True)
    avqa_annotation_file: str = field(default="./data/video_instruction_tuning/avqa/val_qa.json")
    avqa_video_root: str = field(default="./data/video_instruction_tuning/avqa/videos")
    avqa_video_mapping: Optional[str] = field(default="./data/video_instruction_tuning/avqa/avqa_from_vid_to_video_name.json")
    avqa_num_frames: int = field(default=20)
    avqa_max_samples: Optional[int] = field(default=1500)
    avqa_eval_before_train: bool = field(default=False)
    avqa_eval_batch_size: int = field(default=1, metadata={"help": "Micro-batch size per GPU for AVQA evaluation."})

    avqa_pool_video_tokens: bool = field(default=False)
    avqa_frame_chunk_size: int = field(default=4)

    avqa_save_predictions: bool = field(default=True)
    avqa_predictions_dir: Optional[str] = field(default=None)
    avqa_save_metrics: bool = field(default=True)
    avqa_metrics_path: Optional[str] = field(default=None)


@dataclass
class LynXAVSDEvalArguments:
    enable_avsd_eval: bool = field(default=True)
    avsd_annotation_file: str = field(default="./data/video_instruction_tuning/avsd/mock_test_set4DSTC10-AVSD_from_DSTC7_singref.json")
    avsd_gt_file: str = field(default="./data/video_instruction_tuning/avsd/avsd_coco_version_test_gt.json")
    avsd_video_root: str = field(default="./data/video_instruction_tuning/avsd/Charades_vu17_test")
    avsd_num_frames: int = field(default=20)
    avsd_max_samples: Optional[int] = field(default=None)
    avsd_eval_before_train: bool = field(default=False)
    avsd_eval_batch_size: int = field(default=1, metadata={"help": "Micro-batch size per GPU for AVSD evaluation."})

    # avsd_prompt: str = field(default="<image>\nDescribe the video.")
    avsd_max_new_tokens: int = field(default=64)
    avsd_min_new_tokens: int = field(
        default=1,
        metadata={"help": "Minimum number of tokens to generate (helps avoid empty outputs like immediate <|im_end|>)."},
    )
    avsd_pool_video_tokens: bool = field(default=False)
    avsd_frame_chunk_size: int = field(default=4)

    avsd_save_predictions: bool = field(default=True)
    avsd_predictions_dir: Optional[str] = field(default=None)
    avsd_save_metrics: bool = field(default=True)
    avsd_metrics_path: Optional[str] = field(default=None)


@dataclass
class LynXMusicAVQAEvalArguments:
    enable_music_avqa_eval: bool = field(default=False)
    music_avqa_annotation_file: str = field(default="./data/video_instruction_tuning/music_avqa/music_avqa_updated_avqa-test.json")
    music_avqa_video_root: str = field(default="./data/video_instruction_tuning/music_avqa")
    music_avqa_video_mapping: Optional[str] = field(default="./data/video_instruction_tuning/music_avqa/music_avqa_all_videos_mapping.json")
    music_avqa_num_frames: int = field(default=60)
    music_avqa_max_samples: Optional[int] = field(default=None)
    music_avqa_eval_before_train: bool = field(default=False)
    music_avqa_eval_batch_size: int = field(default=1, metadata={"help": "Micro-batch size per GPU for Music-AVQA evaluation."})

    music_avqa_max_new_tokens: int = field(default=8)
    music_avqa_min_new_tokens: int = field(
        default=1,
        metadata={"help": "Minimum number of tokens to generate (helps avoid empty outputs like immediate <|im_end|>)."},
    )
    music_avqa_frame_chunk_size: int = field(default=4)

    music_avqa_save_predictions: bool = field(default=True)
    music_avqa_predictions_dir: Optional[str] = field(default=None)
    music_avqa_save_metrics: bool = field(default=True)
    music_avqa_metrics_path: Optional[str] = field(default=None)


def _main_audio(argv: Optional[Sequence[str]] = None) -> None:
    parser = HfArgumentParser(
        (
            LynXModelArguments,
            LynXSFTDataArguments,
            LlmLoraArguments,
            LynXAVQAEvalArguments,
            LynXAVSDEvalArguments,
            LynXMusicAVQAEvalArguments,
            TrainingArguments,
        )
    )
    if argv is None:
        model_args, data_args, lora_args, avqa_args, avsd_args, music_avqa_args, training_args = parser.parse_args_into_dataclasses()
    else:
        model_args, data_args, lora_args, avqa_args, avsd_args, music_avqa_args, training_args = parser.parse_args_into_dataclasses(args=list(argv))

    if training_args.deepspeed:
        maybe_patch_torch_autograd_graph_for_deepspeed()

    training_args.remove_unused_columns = False
    set_seed(training_args.seed)

    resolved_model_path = resolve_local_model_path(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        local_files_only=model_args.local_files_only,
    )

    processor = AutoProcessor.from_pretrained(resolved_model_path, cache_dir=model_args.cache_dir, local_files_only=model_args.local_files_only)
    tokenizer = processor.tokenizer
    video_processor = processor.video_processor

    device = torch.device(training_args.device)
    torch_dtype = torch.bfloat16 if (device.type == "cuda" and training_args.bf16) else (torch.float16 if device.type == "cuda" else torch.float32)

    attn_impl = resolve_attn_implementation(bool(model_args.use_flash_attn), device=device)
    device_map = {"": str(device)} if device.type == "cuda" else None

    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        resolved_model_path,
        cache_dir=model_args.cache_dir,
        local_files_only=model_args.local_files_only,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map=device_map,
        attn_implementation=attn_impl,
    )
    if device_map is None:
        model.to(device)

    vision_adapter_name: Optional[str] = None
    if model_args.vision_adapter_path:
        from peft import PeftModel

        vision_adapter_name = str(model_args.vision_adapter_name or "vision")
        model = PeftModel.from_pretrained(
            model,
            model_args.vision_adapter_path,
            adapter_name=vision_adapter_name,
            is_trainable=False,
        )

    if lora_args.enable_llm_lora:
        from peft import LoraConfig, PeftModel, get_peft_model

        llm_target_modules = _find_llm_linear_modules(model)
        if not llm_target_modules:
            raise RuntimeError("No LLM linear layers found for LoRA injection.")

        llm_adapter_name = str(lora_args.llm_adapter_name or "llm")
        llm_cfg = LoraConfig(
            r=int(lora_args.llm_lora_r),
            lora_alpha=int(lora_args.llm_lora_alpha),
            target_modules=llm_target_modules,
            lora_dropout=float(lora_args.llm_lora_dropout),
            bias=str(lora_args.llm_lora_bias),
            task_type="CAUSAL_LM",
        )

        if isinstance(model, PeftModel):
            model.add_adapter(llm_adapter_name, llm_cfg)
        else:
            model = get_peft_model(model, llm_cfg, adapter_name=llm_adapter_name)
        model.set_adapter(llm_adapter_name)

        for n, p in model.named_parameters():
            p.requires_grad = ("lora_" in n) and (f".{llm_adapter_name}." in n)
    else:
        llm_adapter_name = None
        for p in model.parameters():
            p.requires_grad = False

    placeholder_id = tokenizer.convert_tokens_to_ids("<image>")
    if placeholder_id is None or placeholder_id == getattr(tokenizer, "unk_token_id", None):
        raise ValueError("Tokenizer does not recognize '<image>' as a single token.")

    lynx_model = LynXOnevisionWrapper(
        model,
        placeholder_token_id=int(placeholder_id),
        pool_video_tokens=bool(data_args.pool_video_tokens),
        frame_chunk_size=int(data_args.frame_chunk_size),
        vision_adapter_name=vision_adapter_name,
        llm_adapter_name=llm_adapter_name,
        detach_modality_tokens=True,
    )

    dataset = str(getattr(data_args, "dataset", "avqa") or "avqa").lower().strip()
    if dataset not in ("avqa", "avsd", "music_avqa"):
        raise ValueError(f"Unsupported --dataset {dataset!r}. Expected one of: avqa, avsd, music_avqa.")

    if dataset == "avsd":
        train_dataset = AVSDInstructionDataset(
            data_args.train_instruct_file,
            video_root=data_args.video_root,
            video_mapping=data_args.video_mapping,
            tokenizer=tokenizer,
            video_processor=video_processor,
            num_frames=data_args.num_frames,
            audio_target_sr=data_args.audio_target_sr,
            audio_seconds=data_args.audio_seconds,
            audio_clip_duration_s=data_args.audio_clip_duration_s,
            audio_clip_stride_s=data_args.audio_clip_stride_s,
            mel_bins=data_args.mel_bins,
            mel_target_length=data_args.mel_target_length,
            mel_mean=data_args.mel_mean,
            mel_std=data_args.mel_std,
            max_samples=data_args.max_samples,
        )

        eval_dataset: Optional[Dataset] = train_dataset if training_args.eval_strategy != IntervalStrategy.NO else None
        trainer = LynXSFTAVSDTrainer(
            model=lynx_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=AVQAInstructionCollator(tokenizer),
            tokenizer=tokenizer,
            processor=processor,
            avsd_args=avsd_args,
            audio_target_sr=data_args.audio_target_sr,
            audio_seconds=data_args.audio_seconds,
            audio_clip_duration_s=data_args.audio_clip_duration_s,
            audio_clip_stride_s=data_args.audio_clip_stride_s,
            mel_bins=data_args.mel_bins,
            mel_target_length=data_args.mel_target_length,
            mel_mean=data_args.mel_mean,
            mel_std=data_args.mel_std,
        )

        if getattr(avsd_args, "avsd_eval_before_train", False):
            trainer.evaluate()
    elif dataset == "music_avqa":
        train_dataset = AVQAInstructionDataset(
            data_args.train_instruct_file,
            video_root=data_args.video_root,
            video_mapping=data_args.video_mapping,
            tokenizer=tokenizer,
            video_processor=video_processor,
            num_frames=data_args.num_frames,
            audio_target_sr=data_args.audio_target_sr,
            audio_seconds=data_args.audio_seconds,
            audio_clip_duration_s=data_args.audio_clip_duration_s,
            audio_clip_stride_s=data_args.audio_clip_stride_s,
            mel_bins=data_args.mel_bins,
            mel_target_length=data_args.mel_target_length,
            mel_mean=data_args.mel_mean,
            mel_std=data_args.mel_std,
            max_samples=data_args.max_samples,
        )

        eval_dataset = train_dataset if training_args.eval_strategy != IntervalStrategy.NO else None
        trainer = LynXSFTMusicAVQATrainer(
            model=lynx_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=AVQAInstructionCollator(tokenizer),
            tokenizer=tokenizer,
            processor=processor,
            music_avqa_args=music_avqa_args,
            audio_target_sr=data_args.audio_target_sr,
            audio_seconds=data_args.audio_seconds,
            audio_clip_duration_s=data_args.audio_clip_duration_s,
            audio_clip_stride_s=data_args.audio_clip_stride_s,
            mel_bins=data_args.mel_bins,
            mel_target_length=data_args.mel_target_length,
            mel_mean=data_args.mel_mean,
            mel_std=data_args.mel_std,
        )

        if getattr(music_avqa_args, "music_avqa_eval_before_train", False):
            trainer.evaluate()
    else:
        train_dataset = AVQAInstructionDataset(
            data_args.train_instruct_file,
            video_root=data_args.video_root,
            video_mapping=data_args.video_mapping,
            tokenizer=tokenizer,
            video_processor=video_processor,
            num_frames=data_args.num_frames,
            audio_target_sr=data_args.audio_target_sr,
            audio_seconds=data_args.audio_seconds,
            audio_clip_duration_s=data_args.audio_clip_duration_s,
            audio_clip_stride_s=data_args.audio_clip_stride_s,
            mel_bins=data_args.mel_bins,
            mel_target_length=data_args.mel_target_length,
            mel_mean=data_args.mel_mean,
            mel_std=data_args.mel_std,
            max_samples=data_args.max_samples,
        )

        eval_dataset = train_dataset if training_args.eval_strategy != IntervalStrategy.NO else None
        trainer = LynXSFTAVQATrainer(
            model=lynx_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=AVQAInstructionCollator(tokenizer),
            tokenizer=tokenizer,
            processor=processor,
            avqa_args=avqa_args,
            audio_target_sr=data_args.audio_target_sr,
            audio_seconds=data_args.audio_seconds,
        )

        if avqa_args.avqa_eval_before_train:
            trainer.evaluate()

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)


def _main_egoexo(argv: Optional[Sequence[str]] = None) -> None:
    parser = HfArgumentParser((LynXModelArguments, EgoExoSFTDataArguments, LlmLoraArguments, TrainingArguments))
    if argv is None:
        model_args, data_args, lora_args, training_args = parser.parse_args_into_dataclasses()
    else:
        model_args, data_args, lora_args, training_args = parser.parse_args_into_dataclasses(args=list(argv))

    if training_args.deepspeed:
        maybe_patch_torch_autograd_graph_for_deepspeed()

    if training_args.output_dir is None:
        raise ValueError("--output_dir is required")
    os.makedirs(training_args.output_dir, exist_ok=True)
    training_args.remove_unused_columns = False
    set_seed(training_args.seed)

    resolved_model_path = resolve_local_model_path(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        local_files_only=model_args.local_files_only,
    )

    processor = AutoProcessor.from_pretrained(
        resolved_model_path,
        cache_dir=model_args.cache_dir,
        local_files_only=model_args.local_files_only,
    )
    tokenizer = processor.tokenizer
    video_processor = processor.video_processor

    device = torch.device(training_args.device)
    torch_dtype = torch.bfloat16 if (device.type == "cuda" and training_args.bf16) else (torch.float16 if device.type == "cuda" else torch.float32)
    attn_impl = resolve_attn_implementation(bool(model_args.use_flash_attn), device=device)
    device_map = {"": str(device)} if device.type == "cuda" else None

    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        resolved_model_path,
        cache_dir=model_args.cache_dir,
        local_files_only=model_args.local_files_only,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map=device_map,
        attn_implementation=attn_impl,
    )
    if device_map is None:
        model.to(device)

    vision_adapter_name: Optional[str] = None
    if model_args.vision_adapter_path:
        from peft import PeftModel  # type: ignore[import-not-found]

        vision_adapter_name = str(model_args.vision_adapter_name or "vision")
        model = PeftModel.from_pretrained(
            model,
            model_args.vision_adapter_path,
            adapter_name=vision_adapter_name,
            is_trainable=False,
        )

    llm_adapter_name: Optional[str] = None
    if lora_args.enable_llm_lora:
        from peft import LoraConfig, PeftModel, get_peft_model  # type: ignore[import-not-found]

        llm_target_modules = _find_llm_linear_modules(model)
        if not llm_target_modules:
            raise RuntimeError("No LLM linear layers found for LoRA injection.")

        llm_adapter_name = str(lora_args.llm_adapter_name or "llm")
        llm_cfg = LoraConfig(
            r=int(lora_args.llm_lora_r),
            lora_alpha=int(lora_args.llm_lora_alpha),
            target_modules=llm_target_modules,
            lora_dropout=float(lora_args.llm_lora_dropout),
            bias=str(lora_args.llm_lora_bias),
            task_type="CAUSAL_LM",
        )

        if isinstance(model, PeftModel):
            model.add_adapter(llm_adapter_name, llm_cfg)
        else:
            model = get_peft_model(model, llm_cfg, adapter_name=llm_adapter_name)
        model.set_adapter(llm_adapter_name)

        for name, param in model.named_parameters():
            param.requires_grad = ("lora_" in name) and (f".{llm_adapter_name}." in name)
    else:
        for param in model.parameters():
            param.requires_grad = False

    placeholder_id = tokenizer.convert_tokens_to_ids("<image>")
    if placeholder_id is None or placeholder_id == getattr(tokenizer, "unk_token_id", None):
        raise ValueError("Tokenizer does not recognize '<image>' as a single token.")

    lynx_model = LynXOnevisionWrapper(
        model,
        placeholder_token_id=int(placeholder_id),
        pool_video_tokens=bool(data_args.pool_video_tokens),
        frame_chunk_size=int(data_args.frame_chunk_size),
        vision_adapter_name=vision_adapter_name,
        llm_adapter_name=llm_adapter_name,
        detach_modality_tokens=True,
    )

    train_dataset = EgoExoProficiencyInstructionDataset(
        data_args.train_instruct_file,
        video_root=data_args.video_root,
        take_id_to_video_mapping=data_args.take_id_to_video_mapping,
        tokenizer=tokenizer,
        video_processor=video_processor,
        num_frames=int(data_args.num_frames),
        max_samples=data_args.max_samples,
        video_backend=str(data_args.video_backend),
    )
    if len(train_dataset) == 0:
        raise RuntimeError("EgoExo SFT dataset is empty. Check --video_root / --take_id_to_video_mapping paths and download the videos.")

    collator = EgoExoInstructionCollator(tokenizer)
    trainer = _LynXPeftSavingTrainer(
        model=lynx_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_state()
    trainer.save_model(training_args.output_dir)


def _main_fastvideo(argv: Optional[Sequence[str]] = None) -> None:
    parser = HfArgumentParser((LynXModelArguments, LynXFastVideoSFTDataArguments, LlmLoraArguments, TrainingArguments))
    if argv is None:
        model_args, data_args, lora_args, training_args = parser.parse_args_into_dataclasses()
    else:
        model_args, data_args, lora_args, training_args = parser.parse_args_into_dataclasses(args=list(argv))

    if training_args.deepspeed:
        maybe_patch_torch_autograd_graph_for_deepspeed()

    if training_args.output_dir is None:
        raise ValueError("--output_dir is required")
    os.makedirs(training_args.output_dir, exist_ok=True)
    training_args.remove_unused_columns = False
    set_seed(training_args.seed)

    resolved_model_path = resolve_local_model_path(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        local_files_only=model_args.local_files_only,
    )

    processor = AutoProcessor.from_pretrained(
        resolved_model_path,
        cache_dir=model_args.cache_dir,
        local_files_only=model_args.local_files_only,
    )
    tokenizer = processor.tokenizer
    video_processor = processor.video_processor

    device = torch.device(training_args.device)
    torch_dtype = torch.bfloat16 if (device.type == "cuda" and training_args.bf16) else (torch.float16 if device.type == "cuda" else torch.float32)
    attn_impl = resolve_attn_implementation(bool(model_args.use_flash_attn), device=device)
    device_map = {"": str(device)} if device.type == "cuda" else None

    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        resolved_model_path,
        cache_dir=model_args.cache_dir,
        local_files_only=model_args.local_files_only,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map=device_map,
        attn_implementation=attn_impl,
    )
    if device_map is None:
        model.to(device)

    vision_adapter_name: Optional[str] = None
    if model_args.vision_adapter_path:
        from peft import PeftModel  # type: ignore[import-not-found]

        vision_adapter_name = str(model_args.vision_adapter_name or "vision")
        model = PeftModel.from_pretrained(
            model,
            model_args.vision_adapter_path,
            adapter_name=vision_adapter_name,
            is_trainable=False,
        )

    llm_adapter_name: Optional[str] = None
    if lora_args.enable_llm_lora:
        from peft import LoraConfig, PeftModel, get_peft_model  # type: ignore[import-not-found]

        llm_target_modules = _find_llm_linear_modules(model)
        if not llm_target_modules:
            raise RuntimeError("No LLM linear layers found for LoRA injection.")

        llm_adapter_name = str(lora_args.llm_adapter_name or "llm")
        llm_cfg = LoraConfig(
            r=int(lora_args.llm_lora_r),
            lora_alpha=int(lora_args.llm_lora_alpha),
            target_modules=llm_target_modules,
            lora_dropout=float(lora_args.llm_lora_dropout),
            bias=str(lora_args.llm_lora_bias),
            task_type="CAUSAL_LM",
        )

        if isinstance(model, PeftModel):
            model.add_adapter(llm_adapter_name, llm_cfg)
        else:
            model = get_peft_model(model, llm_cfg, adapter_name=llm_adapter_name)
        model.set_adapter(llm_adapter_name)

        for name, param in model.named_parameters():
            param.requires_grad = ("lora_" in name) and (f".{llm_adapter_name}." in name)
    else:
        for param in model.parameters():
            param.requires_grad = False

    placeholder_id = tokenizer.convert_tokens_to_ids("<image>")
    if placeholder_id is None or placeholder_id == getattr(tokenizer, "unk_token_id", None):
        raise ValueError("Tokenizer does not recognize '<image>' as a single token.")

    lynx_model = LynXOnevisionWrapper(
        model,
        placeholder_token_id=int(placeholder_id),
        pool_video_tokens=bool(data_args.pool_video_tokens),
        frame_chunk_size=int(data_args.frame_chunk_size),
        vision_adapter_name=vision_adapter_name,
        llm_adapter_name=llm_adapter_name,
        detach_modality_tokens=True,
    )

    train_instruct_files = _split_csv(data_args.train_instruct_files)
    video_roots = _split_csv(data_args.video_roots)
    if not train_instruct_files or not video_roots:
        raise ValueError("--train_instruct_files and --video_roots are required.")
    if len(train_instruct_files) != len(video_roots):
        raise ValueError(f"--train_instruct_files and --video_roots must have the same length, got {len(train_instruct_files)} vs {len(video_roots)}")

    fast_num_frames = int(data_args.fast_num_frames) if data_args.fast_num_frames is not None else int(data_args.fast_frame_multiplier) * int(data_args.slow_num_frames)
    train_dataset = LlavaVideo178KFastInstructionDataset(
        train_instruct_files,
        video_roots=video_roots,
        tokenizer=tokenizer,
        video_processor=video_processor,
        slow_num_frames=int(data_args.slow_num_frames),
        fast_num_frames=fast_num_frames,
        fast_tile_size=int(data_args.fast_tile_size),
        max_samples=data_args.max_samples,
        video_backend="torchvision",
    )
    collator = LlavaVideo178KFastInstructionCollator(tokenizer)

    trainer = _LynXPeftSavingTrainer(
        model=lynx_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_state()
    trainer.save_model(training_args.output_dir)


def _main_3d(argv: Optional[Sequence[str]] = None) -> None:
    parser = HfArgumentParser(
        (
            LynXModelArguments,
            LynX3DSFTDataArguments,
            LlmLoraArguments,
            LynXScanQAEvalArguments,
            LynXSQA3DEvalArguments,
            TrainingArguments,
        )
    )
    if argv is None:
        model_args, data_args, lora_args, scanqa_args, sqa3d_args, training_args = parser.parse_args_into_dataclasses()
    else:
        model_args, data_args, lora_args, scanqa_args, sqa3d_args, training_args = parser.parse_args_into_dataclasses(args=list(argv))

    if training_args.deepspeed:
        maybe_patch_torch_autograd_graph_for_deepspeed()

    if training_args.output_dir is None:
        raise ValueError("--output_dir is required")
    os.makedirs(training_args.output_dir, exist_ok=True)
    training_args.remove_unused_columns = False
    set_seed(training_args.seed)

    resolved_model_path = resolve_local_model_path(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        local_files_only=model_args.local_files_only,
    )

    processor = AutoProcessor.from_pretrained(
        resolved_model_path,
        cache_dir=model_args.cache_dir,
        local_files_only=model_args.local_files_only,
    )
    tokenizer = processor.tokenizer
    video_processor = processor.video_processor

    device = torch.device(training_args.device)
    torch_dtype = torch.bfloat16 if (device.type == "cuda" and training_args.bf16) else (torch.float16 if device.type == "cuda" else torch.float32)
    attn_impl = resolve_attn_implementation(bool(model_args.use_flash_attn), device=device)
    device_map = {"": str(device)} if device.type == "cuda" else None

    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        resolved_model_path,
        cache_dir=model_args.cache_dir,
        local_files_only=model_args.local_files_only,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map=device_map,
        attn_implementation=attn_impl,
    )
    if device_map is None:
        model.to(device)

    vision_adapter_name: Optional[str] = None
    if model_args.vision_adapter_path:
        from peft import PeftModel  # type: ignore[import-not-found]

        vision_adapter_name = str(model_args.vision_adapter_name or "vision")
        model = PeftModel.from_pretrained(
            model,
            model_args.vision_adapter_path,
            adapter_name=vision_adapter_name,
            is_trainable=False,
        )

    llm_adapter_name: Optional[str] = None
    if lora_args.enable_llm_lora:
        from peft import LoraConfig, PeftModel, get_peft_model  # type: ignore[import-not-found]

        llm_target_modules = _find_llm_linear_modules(model)
        if not llm_target_modules:
            raise RuntimeError("No LLM linear layers found for LoRA injection.")

        llm_adapter_name = str(lora_args.llm_adapter_name or "llm")
        llm_cfg = LoraConfig(
            r=int(lora_args.llm_lora_r),
            lora_alpha=int(lora_args.llm_lora_alpha),
            target_modules=llm_target_modules,
            lora_dropout=float(lora_args.llm_lora_dropout),
            bias=str(lora_args.llm_lora_bias),
            task_type="CAUSAL_LM",
        )

        if isinstance(model, PeftModel):
            model.add_adapter(llm_adapter_name, llm_cfg)
        else:
            model = get_peft_model(model, llm_cfg, adapter_name=llm_adapter_name)
        model.set_adapter(llm_adapter_name)

        for name, param in model.named_parameters():
            param.requires_grad = ("lora_" in name) and (f".{llm_adapter_name}." in name)
    else:
        for param in model.parameters():
            param.requires_grad = False

    placeholder_id = tokenizer.convert_tokens_to_ids("<image>")
    if placeholder_id is None or placeholder_id == getattr(tokenizer, "unk_token_id", None):
        raise ValueError("Tokenizer does not recognize '<image>' as a single token.")

    depth_encoding = str(getattr(data_args, "depth_encoding", "") or "").lower().strip()
    merge_modality_frame_pairs = bool(getattr(data_args, "depth_merge_frame_pairs", False)) and ("+" in depth_encoding)
    lynx_model = LynXOnevisionWrapper(
        model,
        placeholder_token_id=int(placeholder_id),
        pool_video_tokens=bool(data_args.pool_video_tokens),
        frame_chunk_size=int(data_args.frame_chunk_size),
        vision_adapter_name=vision_adapter_name,
        llm_adapter_name=llm_adapter_name,
        detach_modality_tokens=True,
        merge_modality_frame_pairs=merge_modality_frame_pairs,
    )

    train_dataset = ThreeDInstructionDataset(
        data_args.train_instruct_file,
        frames_root=data_args.frames_root,
        color_subdir=data_args.color_subdir,
        depth_subdir=data_args.depth_subdir,
        pose_subdir=data_args.pose_subdir,
        pose_matrix_type=data_args.pose_matrix_type,
        tokenizer=tokenizer,
        video_processor=video_processor,
        frame_sampling=data_args.frame_sampling,
        num_frames=int(data_args.num_frames),
        depth_clip_min_mm=float(data_args.depth_clip_min_mm),
        depth_clip_max_mm=float(data_args.depth_clip_max_mm),
        depth_encoding=str(data_args.depth_encoding),
        depth_normals_frame=str(data_args.depth_normals_frame),
        depth_intrinsics_filename=str(data_args.depth_intrinsics_filename),
        depth_auto_scale_intrinsics=bool(data_args.depth_auto_scale_intrinsics),
        max_samples=data_args.max_samples,
    )
    if len(train_dataset) == 0:
        raise RuntimeError("3D SFT dataset is empty. Check --frames_root/--train_instruct_file and ensure frames exist on disk.")

    collator = ThreeDInstructionCollator(tokenizer)
    eval_dataset: Optional[Dataset] = train_dataset if training_args.eval_strategy != IntervalStrategy.NO else None

    dataset = str(getattr(data_args, "dataset", "scanqa") or "scanqa").lower().strip()
    if dataset not in ("scanqa", "sqa", "sqa3d"):
        raise ValueError(f"Unsupported --dataset {dataset!r}. Expected one of: scanqa, sqa.")

    if dataset == "scanqa":
        trainer = LynXSFTScanQATrainer(
            model=lynx_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collator,
            tokenizer=tokenizer,
            processor=processor,
            scanqa_args=scanqa_args,
            data_args=data_args,
        )
        if getattr(scanqa_args, "scanqa_eval_before_train", False):
            trainer.evaluate()
    else:
        trainer = LynXSFTSQA3DTrainer(
            model=lynx_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collator,
            tokenizer=tokenizer,
            processor=processor,
            sqa3d_args=sqa3d_args,
            data_args=data_args,
        )
        if getattr(sqa3d_args, "sqa3d_eval_before_train", False):
            trainer.evaluate()

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)


def _normalize_task_name(task: str) -> Optional[str]:
    normalized = str(task).lower().replace("-", "").replace("_", "")
    if normalized in ("audio", "avqa", "sft"):
        return "audio"
    if normalized in ("egoexo",):
        return "egoexo"
    if normalized in ("fastvideo", "fast", "llavavideo", "video"):
        return "fastvideo"
    if normalized in ("3d", "depth", "scanqa", "sqa"):
        return "3d"
    return None


def _split_task_from_argv(argv: Sequence[str]) -> Tuple[str, List[str]]:
    args = list(argv)
    task: Optional[str] = None

    for idx, arg in enumerate(args):
        if arg == "--task":
            if idx + 1 >= len(args):
                raise ValueError("--task requires a value (audio|3d|egoexo|fastvideo).")
            task = args[idx + 1]
            del args[idx : idx + 2]
            break
        if arg.startswith("--task="):
            task = arg.split("=", 1)[1]
            del args[idx]
            break

    if task is not None:
        normalized = _normalize_task_name(task)
        if normalized is None:
            raise ValueError(f"Unknown --task={task!r}. Expected one of: audio, 3d, egoexo, fastvideo.")
        return normalized, args

    if args and not args[0].startswith("-"):
        normalized = _normalize_task_name(args[0])
        if normalized is not None:
            args.pop(0)
            return normalized, args

    return "audio", args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    task, task_argv = _split_task_from_argv(args)
    if task == "audio":
        _main_audio(task_argv)
        return
    if task == "egoexo":
        _main_egoexo(task_argv)
        return
    if task == "fastvideo":
        _main_fastvideo(task_argv)
        return
    if task == "3d":
        _main_3d(task_argv)
        return
    raise RuntimeError(f"Unhandled task {task!r}")


if __name__ == "__main__":
    main()
