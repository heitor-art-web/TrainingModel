# TrainingModel

Basketball object-detection project built around Ultralytics YOLO, Grounding DINO-assisted labelling, temporal ball tracking and a curated-data workflow for future model iterations.

## Current model

The promoted model remains **v1.0.0**. Training a new candidate does not automatically replace it.

Default local checkpoint:

```text
runs/detect/yolo26b/weights/model_v1.0.0.pt
```

Weights, datasets, local videos, review data and generated runs are intentionally excluded from Git.

## Classes and product priorities

| ID | Class | Priority |
| --- | --- | --- |
| 0 | basketball | highest; prefer fewer false positives |
| 1 | player | high; prefer higher recall |
| 2 | referee | secondary |
| 3 | backboard | secondary |
| 4 | rim | high; prefer higher recall |

`rim` is the canonical class name.

## v1.1.0 data policy

The next dataset is intentionally compact and curated:

- **maximum 4,000 images after deduplication**; this is a hard ceiling, not a target;
- 2,500–3,500 strong images are preferable to filling the cap with repetitive frames;
- target split is **70% train / 15% validation / 15% test**;
- splitting is by complete **video/source**, never by individual random frames;
- one source can exist in only one split;
- exact and near-duplicate frames are removed before the split;
- hard negatives where no basketball is visible are valid and important training examples;
- Grounding DINO is a pre-labelling assistant, not ground truth;
- selected hard cases are manually reviewed;
- `skip` and `uncertain` review rows are excluded from curated training data;
- historical `dataset/`, `dataset_1/` and `dataset_1.1/` are reference sources only;
- the gold test is fully manually reviewed and never used for training or training-oriented hard-case mining.

Detailed collection guidance is in [`DATASET_GUIDE.md`](DATASET_GUIDE.md).

The source registry and collection limits live in `configs/data_sources.toml`.

The first protected gold-test source is:

```text
source id: OwiQgAMqyUs
role: gold_test
segment: approximately the first 60 seconds
training: forbidden
```

Only use footage that you have the necessary rights/permission to use. This repository does not download remote videos.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

## Use v1.0.0 without training

```bash
python3 scripts/inference.py videos/game.mp4
python3 scripts/ball_tracker.py videos/game.mp4
```

Extract representative ball failures:

```bash
python3 scripts/extract_ball_failures.py videos/game.mp4
```

The extractor records representative `predicted`, `lost`, `low_confidence` and `large_jump` frames under `runs/hard_cases/<video_name>/`.

## 1. Collect a compact sample from each video

Do not extract every frame. Use a stable source ID and classify the camera type:

```bash
python3 scripts/collect_video_frames.py \
  videos/game_01.mp4 \
  --source-id game_01 \
  --kind broadcast
```

Supported source kinds are `broadcast`, `fixed_camera`, `stands`, `smartphone` and `other`.

Default collection policy:

```text
normal sampling: 1 frame/second
normal frames/source: 120 by default
absolute normal frames/source: 200
final dataset maximum: 4000 images
```

The output is kept local:

```text
local_data/collected/game_01/
├── images/
├── manifest.csv
└── source.json
```

If you need more diversity, add another independent source instead of increasing the frame count of the same video.

For the protected gold test, use an authorized local copy and evaluation purpose:

```bash
python3 scripts/collect_video_frames.py \
  videos/gold_test_01.mp4 \
  --source-id OwiQgAMqyUs \
  --kind broadcast \
  --purpose evaluation
```

Its registered ~0–60 second segment is used automatically unless explicitly overridden.

## 2. Prepare labels with Grounding DINO

Auto-label a collected source in place:

```bash
python3 scripts/autolabel.py \
  --root local_data/collected/game_01 \
  --skip-copy
```

Configuration lives in `configs/autolabel.toml`. Auto-label output is only a starting point, especially for small, blurred or occluded basketballs.

The legacy default layout is still supported with:

```bash
python3 scripts/autolabel.py
```

## 3. Prepare a review batch

For hard cases:

```bash
python3 scripts/prepare_review.py \
  --batch game_01_ball_failures \
  --source-id game_01 \
  --hard-cases runs/hard_cases/game_01
```

For arbitrary image/YOLO-label folders:

```bash
python3 scripts/prepare_review.py \
  --batch game_01_sample \
  --source-id game_01 \
  --images /path/to/images \
  --labels /path/to/labels
```

A review batch is non-destructive:

```text
runs/review/<batch>/
├── images/
├── labels_original/
├── labels_reviewed/
└── manifest.csv
```

For a protected gold-test source, preparation must be evaluation-only.

## 4. Review locally

```bash
python3 scripts/review_labels.py game_01_ball_failures
```

The reviewer supports all five classes and lets you add, move/resize, delete and reclassify boxes. It also supports filters and the statuses:

```text
corrected
already_correct
skip
uncertain
```

Navigation saves automatically. Original labels are never overwritten.

## 5. Export approved reviewed data

```bash
python3 scripts/export_reviewed.py game_01_ball_failures
```

Only `corrected` and `already_correct` rows are exported into:

```text
local_data/sources/<source_id>/
├── images/
├── labels/
└── manifest.csv
```

Review every selected gold-test frame before export. Training sources only need selective review of hard/high-risk cases plus quality-control samples.

## 6. Build the dataset

```bash
python3 scripts/build_dataset.py
```

To intentionally replace an existing local dataset:

```bash
python3 scripts/build_dataset.py --force
```

The builder:

1. reads complete source folders under `local_data/sources/`;
2. honors explicit/protected source roles;
3. removes exact and near-duplicate images globally;
4. **refuses to build if more than 4,000 images remain after deduplication**;
5. assigns every complete source to exactly one split;
6. targets 70/15/15 while respecting source boundaries;
7. writes `dataset/manifest.csv` and `dataset/split_report.json`.

Result:

```text
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
├── manifest.csv
└── split_report.json
```

Exact proportions can vary slightly because a video is never split across subsets.

`python3 scripts/split_dataset.py` remains as a compatibility command and now routes to the source-aware builder; the old random 80/20 frame split is retired.

## 7. Audit before training

```bash
python3 scripts/audit_dataset.py
```

Fix structural/label issues before training. Empty YOLO labels are allowed when the image deliberately contains none of the five annotated classes.

## 8. Train controlled v1.1.0 candidates

Candidate A — initialize from promoted v1.0.0:

```bash
python3 training/train.py --version 1.1.0 --init promoted
```

Candidate B — initialize from the generic pretrained YOLO base:

```bash
python3 training/train.py --version 1.1.0 --init pretrained
```

The training command independently counts dataset images and **refuses to start above the 4,000-image policy cap**, including datasets prepared manually outside the builder.

Outputs are separated:

```text
runs/detect/
├── model_v1.1.0_promoted/weights/best.pt
└── model_v1.1.0_pretrained/weights/best.pt
```

Keep dataset, epochs, image size, learning rate and other parameters identical between the A/B runs unless deliberately changing the experiment.

## 9. Evaluate candidates

Development validation:

```bash
python3 evaluation/evaluate.py \
  --weights runs/detect/model_v1.1.0_promoted/weights/best.pt \
  --split val
```

Held-out test:

```bash
python3 evaluation/evaluate.py \
  --weights runs/detect/model_v1.1.0_promoted/weights/best.pt \
  --split test
```

Compare summaries:

```bash
python3 evaluation/compare_models.py \
  runs/evaluation/<baseline>/metrics_summary.json \
  runs/evaluation/<candidate>/metrics_summary.json
```

Promotion priorities remain:

- `basketball`: precision should improve;
- `player`: recall must not worsen;
- `rim`: recall must not worsen.

Also compare temporal tracking continuity and behaviour across representative real videos. The test set must not be used to tune training parameters or thresholds.

## End-to-end v1.1.0 flow

```text
varied local videos / images
        ↓
collect sparse normal frames with stable source IDs
        ↓
v1.0.0 hard-case mining + Grounding DINO pre-labels
        ↓
selective manual review + gold-test full review
        ↓
approved curated sources + hard negatives
        ↓
deduplication + hard 4000-image ceiling
        ↓
70% train / 15% val / 15% test by source
        ↓
dataset audit
        ↓
A: v1.0.0 initialization
B: pretrained initialization
        ↓
validation + held-out test + tracking review
        ↓
promote only if agreed priorities improve
```

## Tests and CI

```bash
python3 -m unittest discover -s tests -v
```

GitHub Actions runs syntax and lightweight unit tests on pushes to `main` and pull requests. Training itself is never run in CI.

## Apple Silicon

Automatic device selection prefers CUDA, then Apple MPS, then CPU. Check MPS with:

```bash
python3 -c "import torch; print(torch.backends.mps.is_available())"
```
