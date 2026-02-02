# Data directory layout

This repository does not ship datasets. Place all datasets / converted instruction files under this folder.

Expected subdirectories (referenced by `scripts/**/*.sh`):

- `avqa/` (AVQA audio-visual QA)
- `music_avqa/` (Music-AVQA)
- `avsd/` (DSTC7 AVSD / Charades)
- `egoexo/` (Ego-Exo4D Proficiency Demonstrator)
- `3d/frames_square/` (RGB/Depth/Pose frame dumps for 3D benchmarks)
- `scannet/` (ScanQA converted annotations)
- `sqa/` (SQA3D converted annotations)

See `docs/data_preparation/README.md` for upstream preparation instructions (PAVE).

