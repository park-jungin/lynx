# AVQA (Audio-Visual Question Answering)

Upstream reference: https://github.com/dragonlzm/PAVE/blob/main/doc/avqa_dataset_prep.md

## Expected files (this repo)

Place the following under `data/video_instruction_tuning/avqa/`:

- `videos/` (AVQA video subset, `.mp4`)
- `avqa_from_vid_to_video_name.json` (video-id → filename mapping)
- `train_qa.json` (raw train annotations, used by `scripts/*/avqa_train.sh`)
- `val_qa.json` (val/test annotations, used by `scripts/*/avqa_eval.sh`)
- `avqa_train_qa_instruct.json` (instruction-tuning json, used by `scripts/*/avqa_sft.sh`)

## How to prepare

- Download AVQA annotations from the official AVQA page, and videos via VGGSound (as described in the upstream PAVE doc).
- Either:
  - Download the processed `avqa_train_qa_instruct.json` + `avqa_from_vid_to_video_name.json` from the PAVE dataset release (recommended), or
  - Generate them using the conversion scripts from the upstream PAVE repo.

