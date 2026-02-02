# AVSD (DSTC7) / Charades

Upstream reference: https://github.com/dragonlzm/PAVE/blob/main/doc/avsd_dataset_prep.md

## Expected files (this repo)

Place the following under `data/video_instruction_tuning/avsd/`:

- `avsd_train_instruct.json`
- `avsd_all_videos_mapping.json`
- `Charades_v1/` (training videos)
- `Charades_vu17_test/` (test videos)
- `avsd_coco_version_test_gt.json` (COCO-format GT for evaluation)
- `mock_test_set4DSTC10-AVSD_from_DSTC7_singref.json` (evaluation annotations)

## How to prepare

- Download AVSD annotations (DSTC7) and Charades videos (train + test) as in the upstream PAVE doc.
- Either download the processed jsons from the PAVE dataset release or generate them using the upstream PAVE conversion scripts.

