# Data preparation

This repo follows the same dataset layout and preparation flow as PAVE. For the original, detailed walkthroughs, see the PAVE docs:

- AVQA: https://github.com/dragonlzm/PAVE/blob/main/doc/avqa_dataset_prep.md
- Music-AVQA: https://github.com/dragonlzm/PAVE/blob/main/doc/music_avqa_dataset_prep.md
- AVSD: https://github.com/dragonlzm/PAVE/blob/main/doc/avsd_dataset_prep.md
- Ego-Exo4D (Proficiency Demonstrator): https://github.com/dragonlzm/PAVE/blob/main/doc/egoexo4d_dp_data_prep.md
- LLaVA-Video-178K: https://github.com/dragonlzm/PAVE/blob/main/doc/llava_video_dataset_prep.md
- ScanQA: https://github.com/dragonlzm/PAVE/blob/main/doc/scanqa_dataset_prep.md
- SQA3D: https://github.com/dragonlzm/PAVE/blob/main/doc/sqa3d_dataset_prep.md

Local notes in this repo (paths match the training scripts):

- Audio benchmarks expect files under `data/video_instruction_tuning/{avqa,music_avqa,avsd}/...`.
- Ego-Exo expects files under `data/video_instruction_tuning/egoexo/...`.
- 3D benchmarks expect frames under `data/video_instruction_tuning/3d/frames_square/` and annotations under:
  - `data/video_instruction_tuning/scannet/` (ScanQA)
  - `data/video_instruction_tuning/sqa/` (SQA3D)

