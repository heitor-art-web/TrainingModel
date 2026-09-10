# TrainingModel

Basketball object-detection project built around Ultralytics YOLO, with Grounding DINO helpers, temporal ball tracking and a curated-data workflow for the next model iteration.

## Current model

The current promoted model remains **v1.0.0**. Training a new model does not automatically replace it.

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

## Data policy for v1.1.0

The next dataset follows these rules:

- target split is **70% train / 15% validation / 15% test**;
- splitting is by complete **video/source**, never by individual random frames;
- a source can exist in only one split;
- exact and near-duplicate frames are filtered before the split;
- hard negatives where no basketball is visible are valid training examples;
- Grounding DINO is a pre-labelling assistant, not ground truth;
- selected hard cases are manually reviewed;
- `skip` and `uncertain` review rows are not exported into curated training data;
- historical `dataset/`, `dataset_1/` and `dataset_1.1/` are reference sources only, not automatically merged into v1.1.0;
- the gold test is fully manually reviewed and never used for training or training-oriented hard-case mining.

The source registry is `configs/data_sources.toml`. The first protected gold-test source is:

```text
source id: OwiQgAMqyUs
role: gold_test
segment: approximately the first 60 seconds
training: forbidden
```

Only use locally obtained footage when you have the rights/permission needed for that use. This repository does not download YouTube videos.

## Local data layout

Use stable source IDs so origin is never lost:

```text
local_data/
└── sources/
    ├── game_or_video_A/
    │   ├── images/
    │   ├── labels/
    │   └── manifest.csv       # optional but recommended
    ├── game_or_video_B/
    │   ├── images/
    │   └── labels/
    └── OwiQgAMqyUs/
        ├── images/
        ├── labels/
        └── manifest.csv       # gold-test rows must all be manually approved
```

`local_data/` is ignored by Git.

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

Ball tracking output:

```text
runs/ball_tracking/<video_name>/
├── tracked.mp4
├── tracking_report.json
└── trajectory.csv
```

Extract representative ball failures:

```bash
python3 scripts/extract_ball_failures.py videos/game.mp4
```

This produces `runs/hard_cases/<video_name>/frames/` plus `manifest.csv` with reasons such as `predicted`, `lost`, `low_confidence` and `large_jump`.

## Prepare labels with Grounding DINO

The existing auto-label helper remains available:

```bash
python3 scripts/autolabel.py --root local_data/work/game_A
```

or with the legacy layout:

```bash
python3 scripts/autolabel.py
```

Configuration lives in `configs/autolabel.toml`. Auto-label output must be treated as a starting point, especially for small/blurred/occluded basketballs.

## Prepare a review batch

For hard cases:

```bash
python3 scripts/prepare_review.py \
  --batch game_A_ball_failures \
  --source-id game_A \
  --hard-cases runs/hard_cases/game_A
```

For an arbitrary image/YOLO-label folder:

```bash
python3 scripts/prepare_review.py \
  --batch game_A_sample \
  --source-id game_A \
  --images /path/to/images \
  --labels /path/to/labels
```

A batch is non-destructive:

```text
runs/review/<batch>/
├── images/
├── labels_original/
├── labels_reviewed/
└── manifest.csv
```

For a protected gold-test source, preparation must explicitly be evaluation-only:

```bash
python3 scripts/prepare_review.py \
  --batch gold_test_01 \
  --source-id OwiQgAMqyUs \
  --images /path/to/gold/images \
  --labels /path/to/gold/labels \
  --purpose evaluation
```

## Review locally in the browser

```bash
python3 scripts/review_labels.py game_A_ball_failures
```

The reviewer supports all five classes and allows you to:

- add a bounding box;
- move/resize a box;
- delete an incorrect box;
- change a box class;
- filter by hard-case reason, class and review state;
- see original confidence/metadata when it exists;
- mark each frame `corrected`, `already_correct`, `skip` or `uncertain`.

Navigation saves automatically. On restart, the default view resumes with unreviewed frames. Original labels are never overwritten.

## Export approved reviewed data

```bash
python3 scripts/export_reviewed.py game_A_ball_failures
```

Only `corrected` and `already_correct` rows are copied into:

```text
local_data/sources/<source_id>/
├── images/
├── labels/
└── manifest.csv
```

`skip` and `uncertain` rows remain in the review batch but are excluded from the curated source.

For the gold test, review **every selected frame** before export. The dataset builder refuses unreviewed gold-test rows.

## Build the 70/15/15 dataset

Once curated source directories exist:

```bash
python3 scripts/build_dataset.py
```

If `dataset/` already exists and you intentionally want to rebuild it:

```bash
python3 scripts/build_dataset.py --force
```

The builder:

1. reads complete source folders under `local_data/sources/`;
2. honors protected/explicit roles from `configs/data_sources.toml`;
3. removes exact and near-duplicate images globally, preserving protected test and hard-case examples first;
4. assigns each remaining source to exactly one split;
5. targets 70/15/15 by frame count while respecting whole-source boundaries;
6. prefixes output filenames with the source ID;
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

The exact proportions can vary slightly because a video is never split across subsets.

`python3 scripts/split_dataset.py` remains as a compatibility command but now invokes this source-aware builder; the former random 80/20 frame split is retired.

## Audit before training

```bash
python3 scripts/audit_dataset.py
```

Fix structural/label issues before starting a run. Empty YOLO labels are allowed and are useful for deliberate negative examples when the corresponding image truly contains none of the five annotated classes.

## Train the v1.1.0 A/B candidates

The agreed experiment changes only the initialization while keeping dataset and training configuration identical.

Candidate A — start from promoted v1.0.0:

```bash
python3 training/train.py --version 1.1.0 --init promoted
```

Candidate B — start from the generic pretrained YOLO base:

```bash
python3 training/train.py --version 1.1.0 --init pretrained
```

The pretrained base is configured in `configs/train.toml` (`yolo26b.pt` by default). Ultralytics may download that public base checkpoint when it is not already cached.

Outputs are separated:

```text
runs/detect/
├── model_v1.1.0_promoted/weights/best.pt
└── model_v1.1.0_pretrained/weights/best.pt
```

A custom local start is still possible with `--weights /path/to/model.pt`, which creates a `_custom` run.

Do not change epochs, image size, learning rate or other training settings between the two A/B candidates unless the experiment is intentionally being changed. Current defaults remain in `configs/train.toml`.

## Evaluate candidates

Validation during development:

```bash
python3 evaluation/evaluate.py \
  --weights runs/detect/model_v1.1.0_promoted/weights/best.pt \
  --split val
```

Final held-out test evaluation:

```bash
python3 evaluation/evaluate.py \
  --weights runs/detect/model_v1.1.0_promoted/weights/best.pt \
  --split test
```

`metrics_summary.json` now includes per-class precision, recall and mAP metrics.

Compare a baseline summary against a candidate summary:

```bash
python3 evaluation/compare_models.py \
  runs/evaluation/<baseline>/metrics_summary.json \
  runs/evaluation/<candidate>/metrics_summary.json
```

Detection promotion priorities are:

- `basketball`: precision must improve (false positives are the primary concern);
- `player`: recall must not worsen;
- `rim`: recall must not worsen.

Passing those checks is **not sufficient by itself**. Before promotion, compare tracking continuity and behaviour on representative real videos. The test set must not be used to tune thresholds or choose training settings.

## Model promotion

`configs/model.toml` remains the source of truth for the promoted model. Keep v1.0.0 promoted until a candidate has passed validation, held-out testing and real-video tracking review.

Only then:

1. copy/rename the selected checkpoint using the model version convention;
2. update `configs/model.toml`;
3. record the promotion in `CHANGELOG.md`.

## End-to-end v1.1.0 flow

```text
real videos / images from varied camera types
        ↓
source IDs preserved
        ↓
normal-frame sampling + hard-case mining
        ↓
Grounding DINO pre-labels
        ↓
selective manual review
        ↓
approved curated sources + hard negatives
        ↓
source-aware deduplication
        ↓
70% train / 15% val / 15% test by source
        ↓
dataset audit
        ↓
A: v1.0.0 initialization
B: pretrained initialization
        ↓
validation comparison
        ↓
held-out test + tracking/product-video review
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
