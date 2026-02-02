# Ego-Exo4D (Proficiency Demonstrator)

Upstream reference: https://github.com/dragonlzm/PAVE/blob/main/doc/egoexo4d_dp_data_prep.md

## Expected files (this repo)

Place the following under `data/video_instruction_tuning/egoexo/`:

- `from_take_id_to_video.json` (take-id → video path)
- `proficiency_demonstrator_train_instruct.json` (instruction-tuning json)
- the (downscaled) take videos referenced by the mapping

## How to prepare

- Obtain Ego-Exo4D access/license from the official site.
- Use the `egoexo` downloader CLI (as described in the upstream doc) to fetch `annotations` + `downscaled_takes/448`.
- Either download the processed jsons from the PAVE dataset release, or convert them using the upstream PAVE tools.

