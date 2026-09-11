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

## INSTRUCTIONS

### Beginner workflow

Use this section if you want to add a new dataset and train a new model version.

```text
get data
  ↓
convert/group it into sources
  ↓
create or import labels
  ↓
manually review that specific dataset/source
  ↓
export approved data
  ↓
build + audit dataset
  ↓
train
  ↓
compare with previous model
```

The project always uses these class IDs:

| ID | Class |
| --- | --- |
| 0 | basketball |
| 1 | player |
| 2 | referee |
| 3 | backboard |
| 4 | rim |

### 1. Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On future terminal sessions:

```bash
source .venv/bin/activate
```

### 2. Add new data

You can use your own videos, an existing YOLO dataset, or a dataset downloaded from Hugging Face.

Each independent game/video should remain a separate **source**. TrainingModel keeps a complete source in only one of `train`, `val` or `test`, which prevents frames from the same game leaking into both training and evaluation.

#### Your own video

```bash
python3 scripts/collect_video_frames.py \
  videos/game_01.mp4 \
  --source-id game_01 \
  --kind broadcast
```

Create initial Grounding DINO labels:

```bash
python3 scripts/autolabel.py \
  --root local_data/collected/game_01 \
  --skip-copy
```

Grounding DINO output is a pre-annotation, not guaranteed ground truth.

### 3. Manual review — one specific dataset/source at a time

Manual review is not performed blindly across the whole repository. You create a review batch for the specific dataset/source you are currently adding.

For example, to review `game_01`:

```bash
python3 scripts/prepare_review.py \
  --batch game_01_review \
  --source-id game_01 \
  --images local_data/collected/game_01/images \
  --labels local_data/collected/game_01/labels
```

This creates:

```text
runs/review/game_01_review/
├── images/
├── labels_original/
├── labels_reviewed/
└── manifest.csv
```

Open that exact review batch:

```bash
python3 scripts/review_labels.py game_01_review
```

The browser reviewer lets you add, delete, move/resize and reclassify boxes. Normal status is automatic:

```text
no edits  -> already_correct
edited    -> corrected
```

Use `skip` or `uncertain` only when needed. Original labels under `labels_original/` are preserved.

When that dataset/source review is complete:

```bash
python3 scripts/export_reviewed.py game_01_review
```

Approved data is exported to:

```text
local_data/sources/game_01/
```

Repeat this process separately for every new source you want to add.

### 4. Hugging Face datasets

Check/install the Hugging Face CLI:

```bash
hf --help
```

If needed:

```bash
pip install -U huggingface_hub
```

Generic download pattern:

```bash
hf download \
  OWNER/DATASET_NAME \
  --repo-type dataset \
  --local-dir local_data/external/DATASET_NAME
```

Different Hugging Face datasets may use YOLO, COCO JSON, Parquet, XML or another format. Before training, they must ultimately be converted into source folders such as:

```text
local_data/sources/<source_id>/
├── images/
└── labels/
```

Labels must use the TrainingModel class IDs listed above. Never assume an external dataset uses the same numeric class mapping. Preserve original game/video/source grouping whenever possible, and check the dataset license before using it.

#### E-BARD example

E-BARD already has dedicated support in this repository.

```bash
mkdir -p local_data/external/e_bard

hf download \
  GabrieleGiudici/E-BARD-detection \
  all.zip \
  --repo-type dataset \
  --local-dir local_data/external/e_bard

cd local_data/external/e_bard
unzip all.zip
cd ../../..

python3 scripts/import_e_bard.py
```

A healthy import is approximately 1,800 frames from 60 games/sources.

E-BARD has no native `backboard` class, so add Grounding DINO backboard pre-labels:

```bash
python3 scripts/enrich_ebard_backboard.py
```

For E-BARD, manual review uses a QA sample instead of reviewing all 1,800 frames:

```bash
python3 scripts/prepare_ebard_review.py
python3 scripts/review_labels.py ebard_backboard_qa
```

By default this is about 4 images per game, or roughly 240 images total. If the sample looks good, finalize E-BARD:

```bash
python3 scripts/finalize_ebard.py
```

If the sample shows systematic errors, increase the review sample before finalizing. See [`EBARD_GUIDE.md`](EBARD_GUIDE.md) for the full E-BARD workflow.

#### Basketball Game Detections (planned v1.2.0 source)

TrainingModel plans to use the **Basketball Game Detections Dataset** published by **FirstTime** on Roboflow Universe:

- Source: https://universe.roboflow.com/firsttime/basketball-game-detections
- Publisher/host: Roboflow Universe
- Dataset author/attribution: FirstTime
- License declared by the dataset page: **CC BY 4.0 (Creative Commons Attribution 4.0 International)**
- License text: https://creativecommons.org/licenses/by/4.0/

Under CC BY 4.0, appropriate attribution must be retained, the license must be identified/linked, and modifications to the licensed material must be indicated when the material is shared. This repository does not claim that the dataset license resolves every possible third-party right in the underlying imagery; users remain responsible for checking whether their intended use requires additional permissions.

TrainingModel adapts the source annotations to its canonical class schema as follows:

| Basketball Game Detections | TrainingModel | Treatment |
| --- | --- | --- |
| `Ball` | `basketball` (0) | remapped |
| `Player` | `player` (1) | remapped |
| `Ref` | `referee` (2) | remapped |
| `Hoop` | `rim` (4) | remapped; `rim` is the canonical project class |
| `FG Attempt` | — | ignored |
| `FG Made` | — | ignored |

`backboard` (class 3) is not provided by this source and must not be invented from the `Hoop` annotation. Any later backboard enrichment must be documented separately as a TrainingModel modification/pre-annotation step.

When this dataset is imported, provenance is preserved in source metadata/manifests, including the dataset name, author, source URL, CC BY 4.0 license, class remapping, ignored classes, and the fact that annotations were converted into TrainingModel's YOLO/source structure.

Do not automatically ingest every exported image. Prefer original/non-augmented images where available, preserve source/game grouping when recoverable, remove exact/near duplicates, and run a dataset-specific QA sample before admitting the selected subset to `local_data/sources/`.

##### Basketball Game Detections qualification workflow

Export/download the dataset in YOLO format and unpack it under:

```text
local_data/external/basketball_game_detections/
```

First analyze the whole external dataset without modifying the curated training sources:

```bash
python3 scripts/analyze_basketball_game_detections.py
```

By default the analyzer uses the v1.1.0 candidate checkpoint at:

```text
runs/detect/model_v1.1.0_promoted/weights/best.pt
```

and selects up to 3,500 useful images. It reports exact/near duplicates, ball-size buckets (`tiny`, `small`, `medium`, `large`), frames where v1.1.0 misses a labelled ball, frames where it predicts a ball but the source label has none, no-ball hard negatives, `ball + rim` context and general diversity.

Reports are written to:

```text
runs/dataset_analysis/basketball_game_detections/
├── analysis_manifest.csv
├── selected_manifest.csv
└── analysis_summary.json
```

To change the selection size:

```bash
python3 scripts/analyze_basketball_game_detections.py \
  --max-select 3000 \
  --force
```

Roboflow filenames are not treated as proof of original game identity. If the downloaded filenames/metadata contain a reliable game/video identifier, explicitly provide a regex with a `source` capture group, for example:

```bash
python3 scripts/analyze_basketball_game_detections.py \
  --source-regex '^(?P<source>game_[0-9]+)_frame_[0-9]+' \
  --force
```

Only use such a regex when the captured value really identifies an original independent game/video. Otherwise leave grouping unknown. Unknown grouping is deliberately imported as a **training-only** source so this external dataset cannot leak into validation/test.

After inspecting `analysis_summary.json` and `selected_manifest.csv`, stage only the selected images:

```bash
python3 scripts/import_basketball_game_detections.py
```

Default staging output:

```text
local_data/bgd_import/sources/bgd_training_pool/
```

If reliable source groups were proven during analysis, import them separately with:

```bash
python3 scripts/import_basketball_game_detections.py \
  --use-proven-source-groups
```

Before the staged data becomes eligible for training, create a stratified QA sample. The default is 320 frames spread across current-model false negatives, false positives, tiny/small-ball cases, no-ball hard negatives, `ball + rim` and general examples:

```bash
python3 scripts/prepare_bgd_review.py
python3 scripts/review_labels.py bgd_qa
```

Every sampled QA row must finish as `corrected` or `already_correct`. If the sample reveals systematic label problems, do not finalize; increase the QA sample and review again:

```bash
python3 scripts/prepare_bgd_review.py --sample-size 600 --force
python3 scripts/review_labels.py bgd_qa
```

When the QA sample is clean:

```bash
python3 scripts/finalize_bgd.py
```

This copies the staged selected dataset into `local_data/sources/`, applying the manually corrected labels to sampled frames and retaining provenance for all imported frames. `bgd_training_pool` is explicitly registered with `role = "train"` in `configs/data_sources.toml` while original game grouping remains unknown.

The current 4,000-image hard cap belongs to the v1.1.0 policy. **Do not silently raise it.** Decide the v1.2.0 dataset budget explicitly after inspecting the Basketball Game Detections analysis report, then update the configured cap before building/training v1.2.0.

### 5. Build and audit the final dataset

Once approved sources are under `local_data/sources/`:

```bash
python3 scripts/build_dataset.py --force
python3 scripts/audit_dataset.py
```

The builder keeps sources together, removes duplicates, targets a 70/15/15 split and enforces the current 4,000-image maximum. Do not train until the audit passes.

### 6. Train a new version

To continue from the currently promoted TrainingModel:

```bash
python3 training/train.py \
  --version NEW_VERSION \
  --init promoted
```

Example:

```bash
python3 training/train.py --version 1.1.1 --init promoted
```

Training output is written under `runs/detect/model_vNEW_VERSION_promoted/`. Normally use `weights/best.pt` for evaluation.

### 7. Evaluate the candidate

Validation:

```bash
python3 evaluation/evaluate.py \
  --weights runs/detect/model_vNEW_VERSION_promoted/weights/best.pt \
  --split val
```

Final held-out test:

```bash
python3 evaluation/evaluate.py \
  --weights runs/detect/model_vNEW_VERSION_promoted/weights/best.pt \
  --split test
```

Evaluate the previous model on the same test split and compare the generated summaries:

```bash
python3 evaluation/compare_models.py \
  runs/evaluation/<previous>/metrics_summary.json \
  runs/evaluation/<candidate>/metrics_summary.json
```

Pay particular attention to basketball precision/recall, player recall and rim recall. Do not judge a candidate only by global mAP.

### Recommended improvement loop

```text
current model
  ↓
real basketball videos
  ↓
find false positives / missed detections
  ↓
collect those hard frames
  ↓
review that specific batch
  ↓
add approved data
  ↓
fine-tune
  ↓
compare old vs new model
```

Useful hard cases include small/distant basketballs, motion blur, partial occlusion, ball near player hands/rim, and frames where no basketball is visible.

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
