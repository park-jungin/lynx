# SQA3D (3D)

Upstream reference: https://github.com/dragonlzm/PAVE/blob/main/doc/sqa3d_dataset_prep.md

## Expected files (this repo)

Place the following under `data/video_instruction_tuning/`:

- Frames:
  - `3d/frames_square/<scene_id>/{color,depth,pose}/...`
- Annotations:
  - `sqa/SQA_train_instruct.json`
  - `sqa/llava3d_sqa3d_test_question.json`
  - `sqa/llava3d_sqa3d_test_answer.json`

## How to prepare

- Download SQA3D and extract RGB/depth frames as in the upstream PAVE doc.
- Either download the processed jsons from the PAVE dataset release, or generate them using the upstream PAVE tools.

