# LLaVA-Video-178K (FastVideo)

Upstream reference: https://github.com/dragonlzm/PAVE/blob/main/doc/llava_video_dataset_prep.md

## Expected files (this repo)

Place the processed annotation + mapping jsons under `data/video_instruction_tuning/llava_video_178k/` (see the defaults in `scripts/*/llava_video.sh`):

- `*_videos_mapping_updated.json` (comma-separated list passed via `VIDEO_MAPPING_LIST`)
- `*_qa_processed_2pv.json` (comma-separated list passed via `INSTRUCT_FILES`)

Videos themselves are not expected to live inside this repo; `scripts/*/llava_video.sh` defaults `VIDEO_ROOT_LIST`/`INSTRUCT_VIDEO_ROOTS` to paths like `/mnt/hdd1/datasets/...` which you should adjust for your machine.

## How to prepare

- Download LLaVA-Video-178K from the official Hugging Face dataset page.
- Generate (or download) the processed annotation/mapping files as described in the upstream PAVE doc.

