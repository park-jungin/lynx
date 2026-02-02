# Music-AVQA

Upstream reference: https://github.com/dragonlzm/PAVE/blob/main/doc/music_avqa_dataset_prep.md

## Expected files (this repo)

Place the following under `data/video_instruction_tuning/music_avqa/`:

- `music_avqa_train_instruct_duplicate_audio.json` (train instruct json)
- `music_avqa_all_videos_mapping.json` (video-id → filename mapping)
- `music_avqa_updated_avqa-test.json` (eval annotation json)
- the corresponding videos (paths resolved from the mapping), under the same root directory

## How to prepare

- Download the dataset from the official Music-AVQA site.
- Either download the processed jsons from the PAVE dataset release, or reproduce them following the upstream PAVE doc.

