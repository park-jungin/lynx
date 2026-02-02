# ScanQA (3D)

Upstream reference: https://github.com/dragonlzm/PAVE/blob/main/doc/scanqa_dataset_prep.md

## Expected files (this repo)

Place the following under `data/video_instruction_tuning/`:

- Frames:
  - `3d/frames_square/<scene_id>/{color,depth,pose}/...` (32-frame RGB/depth + poses)
- Annotations:
  - `scannet/scanqa_train_instruct.json`
  - `scannet/llava-3d-scanqa_val_question.json`
  - `scannet/llava3d_scanqa_val_answer.json`

## How to prepare

- Download ScanQA and extract RGB/depth frames as in the upstream PAVE doc.
- Either download the processed jsons from the PAVE dataset release, or generate them using the upstream PAVE tools.

