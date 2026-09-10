# v1.1.0 Dataset Guide

The v1.1.0 dataset is intentionally **small and curated**. The project-wide hard ceiling is **4,000 images after deduplication**. This is a maximum, not a target: a clean 2,500–3,500 image dataset is preferable to filling the budget with repetitive frames.

## Target composition

Use multiple independent sources and preserve a stable `source_id` for each complete video/session. Aim for roughly 15–30 varied sources across broadcast footage, fixed cameras, stands/handheld footage and smartphone recordings. Do not force equal source counts if the available footage is not equally useful.

Normal video sampling should stay sparse. The default collector uses 1 frame/second and at most 120 normal frames per source, with an absolute normal-sampling cap of 200 frames per source. Hard cases and manually selected examples are added separately.

Prioritize data that was weak in v1.0.0:

- small/distant basketballs;
- motion blur;
- partial occlusion;
- passes, shots, rebounds and fast transitions;
- ball near hands, heads, rim or backboard;
- frames with no visible basketball, especially real false positives from v1.0.0;
- players and rims at different scales and viewpoints.

The five labels remain `basketball`, `player`, `referee`, `backboard`, and `rim`. Product priorities are basketball precision, player recall and rim recall.

## Collect normal frames from a local video

The repository does not download remote videos. Obtain footage locally only when you have the necessary rights/permission, then run:

```bash
python3 scripts/collect_video_frames.py \
  videos/game_01.mp4 \
  --source-id game_01 \
  --kind broadcast
```

Output:

```text
local_data/collected/game_01/
├── images/
├── manifest.csv
└── source.json
```

The default is approximately 1 sampled frame per second, capped at 120 normal frames for that source. For a deliberately shorter/larger sample within policy:

```bash
python3 scripts/collect_video_frames.py \
  videos/game_02.mp4 \
  --source-id game_02 \
  --kind smartphone \
  --fps 0.5 \
  --max-frames 80
```

Do not compensate for poor diversity by increasing `--max-frames`. Add another independent video/source instead.

## Gold test collection

`OwiQgAMqyUs` is protected in `configs/data_sources.toml`. If you have an authorized local copy, collect its registered first ~60 seconds only for evaluation:

```bash
python3 scripts/collect_video_frames.py \
  videos/gold_test_01.mp4 \
  --source-id OwiQgAMqyUs \
  --kind broadcast \
  --purpose evaluation
```

The registered segment is applied automatically unless `--start`/`--end` are supplied. The protected source cannot be collected for training.

## Label and review

For a collected source, Grounding DINO can create initial labels in place:

```bash
python3 scripts/autolabel.py \
  --root local_data/collected/game_01 \
  --skip-copy
```

Then prepare selected frames for manual review. Review all gold-test frames, but only selected/high-risk training frames. Keep `skip` and `uncertain` out of curated training data.

Use v1.0.0 plus the tracker to add hard cases from the same training-eligible sources. The normal collector is not intended to replace hard-case mining.

## Build budget

Approved sources eventually live under:

```text
local_data/sources/<source_id>/
├── images/
├── labels/
└── manifest.csv
```

Then build:

```bash
python3 scripts/build_dataset.py
```

The builder removes exact/near duplicates first and then enforces the **4,000-image hard cap**. If more than 4,000 curated images remain, the build fails deliberately. Reduce repetitive/easy sources or review the source mix; do not raise the cap for v1.1.0.

The split remains approximately:

```text
70% train
15% validation
15% test
```

Because complete sources are assigned together, exact frame counts can differ from 2,800 / 600 / 600. Never split one video across train/validation/test merely to hit exact percentages.

## Practical stopping rule

Stop collecting when the dataset already covers the important camera types and failure modes. Do not wait until 4,000 images just because budget remains. After the two controlled v1.1.0 candidates are trained and evaluated, add more data only if the evaluation shows a specific gap worth fixing.
