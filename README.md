# TrainingModel

Basketball object-detection project built around Ultralytics YOLO, with Grounding DINO helpers for preparing new labelled data.

## Current model

The current promoted model is **v1.0.0**.

Its default local checkpoint path is configured in `configs/model.toml`:

```text
runs/detect/yolo26b/weights/model_v1.0.0.pt
```

Model weights, datasets, local source videos and generated runs are intentionally excluded from Git. A cloned repository therefore contains the code/configuration, but you must copy the required `.pt` file, dataset and any source footage to the machine where they are needed.

## Classes

| ID | Class |
| --- | --- |
| 0 | basketball |
| 1 | player |
| 2 | referee |
| 3 | backboard |
| 4 | rim |

`rim` is the canonical class name across dataset and auto-label configuration.

## Project structure

```text
TrainingModel/
├── configs/
│   ├── autolabel.toml
│   ├── ball_tracking.toml
│   ├── model.toml
│   └── train.toml
├── evaluation/
│   └── evaluate.py
├── scripts/
│   ├── audit_dataset.py
│   ├── autolabel.py
│   ├── ball_tracker.py
│   ├── CVAT_analysis.py
│   ├── extract_ball_failures.py
│   ├── flatten_dataset.py
│   ├── inference.py
│   ├── rename.py
│   └── split_dataset.py
├── tests/
├── tracking/
├── training/
│   ├── __init__.py
│   ├── common.py
│   └── train.py
├── videos/                 # local only; ignored by Git
├── CHANGELOG.md
├── dataset.yaml
├── requirements.txt
└── README.md
```

The `videos/` directory is the standard location for local game footage used during inference, tracking and hard-case extraction. It is ignored by Git so large video files are never pushed to the repository.

Create it locally after cloning:

```bash
mkdir -p videos
```

Example:

```text
videos/
└── game.mp4
```

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

## 1. Use the existing model

Training is **not required** to use `model_v1.0.0.pt`.

Run inference on an image, video, directory or supported URL/source:

```bash
python3 scripts/inference.py videos/game.mp4
```

Useful options:

```bash
python3 scripts/inference.py /path/to/image.jpg --conf 0.35
python3 scripts/inference.py videos/game.mp4 --no-save
python3 scripts/inference.py videos/game.mp4 --weights /path/to/another_model.pt
```

Default confidence, IoU and image size are defined in `configs/model.toml`. Annotated outputs are saved under `runs/inference/` unless `--no-save` is used.

## 2. Hybrid ball tracking

For basketball-specific video analysis, `scripts/ball_tracker.py` combines the current YOLO detector with temporal tracking. The detector remains unchanged; the tracker associates ball detections between frames, estimates velocity and predicts short gaps when the ball is temporarily missed because of blur or occlusion.

Place local game footage under `videos/` and run:

```bash
python3 scripts/ball_tracker.py videos/game.mp4
```

Useful options:

```bash
python3 scripts/ball_tracker.py videos/game.mp4 --no-video
python3 scripts/ball_tracker.py videos/game.mp4 --max-gap 8
python3 scripts/ball_tracker.py videos/game.mp4 --weights /path/to/another_model.pt
```

Defaults live in `configs/ball_tracking.toml`. Output is written under:

```text
runs/ball_tracking/<video_name>/
├── tracked.mp4
├── tracking_report.json
└── trajectory.csv
```

Each trajectory row is marked as `detected`, `predicted` or `lost`. Predicted positions are intentionally limited to a short configurable gap and are not treated as real detector outputs. This gives a way to measure how much temporal information improves ball continuity before deciding whether a new detector version is necessary.

### Extract hard cases

After running the tracker, extract representative problem frames with:

```bash
python3 scripts/extract_ball_failures.py videos/game.mp4
```

The extractor samples frames such as predicted gaps, lost tracking, low-confidence detections and large trajectory jumps. Outputs are written under `runs/hard_cases/<video_name>/` for later review or dataset construction.

## 3. Model configuration and versioning

`configs/model.toml` is the source of truth for the **promoted/current** model:

- model version;
- local checkpoint path;
- class names;
- default inference settings.

Project convention for model versions:

- **PATCH** (`1.0.0` → `1.0.1`): a small corrective model iteration with no material change to the task/classes or overall training strategy;
- **MINOR** (`1.0.0` → `1.1.0`): new training data or a meaningful improvement to the training/data pipeline while keeping the same task/classes;
- **MAJOR** (`1.x` → `2.0.0`): a breaking change to the model contract, classes or intended task.

A candidate checkpoint is not promoted merely because training completes. Evaluate it first, then update `configs/model.toml` only after deciding that it replaces the current model.

## 4. Download or collect a new training set

New training sets are intentionally kept **outside Git**. Do not commit large image collections or exported datasets to the repository.

Recommended workflow:

1. Download or export the new image set to a local folder.
2. Keep the raw copy unchanged so it can be reproduced later.
3. Move/copy the images into the local preparation structure expected by the helper scripts.
4. Label or auto-label the images.
5. Review the labels, build the train/validation dataset, audit it, then train a new model candidate.

A simple local layout can be:

```text
TrainingModel/
├── modelv0.2/              # local only; ignored by Git
│   ├── images_input/       # newly downloaded/exported images
│   ├── images/
│   ├── frames/
│   └── labels/
└── dataset/                # generated train/validation dataset; ignored by Git
```

For example, after downloading a new image set:

```bash
mkdir -p modelv0.2/images_input
cp /path/to/downloaded/images/* modelv0.2/images_input/
```

If the downloaded set is arranged in nested folders (for example by team/player), the existing flattening helper can be used when its input layout matches your source:

```bash
python3 scripts/flatten_dataset.py
```

The project does not currently include a provider-specific dataset downloader because the source of future data may change. If a dataset is downloaded from Kaggle, Hugging Face, CVAT, cloud storage or another provider, export/download it locally first and then feed the images into the preparation flow above.

### Using hard cases as a future training set

The hybrid tracker can help create the next targeted dataset from real product footage:

```bash
python3 scripts/ball_tracker.py videos/game.mp4
python3 scripts/extract_ball_failures.py videos/game.mp4
```

Review the frames under:

```text
runs/hard_cases/<video_name>/frames/
```

Useful confirmed failures can then be copied into the next local image set and manually labelled or auto-labelled before training. This is the preferred path for improving the `basketball` class because it focuses new data on actual model failures rather than simply adding more easy frames.

## 5. Prepare and label the new data

### Auto-label with Grounding DINO

The auto-label configuration is in `configs/autolabel.toml`.

Current defaults are:

```text
model: IDEA-Research/grounding-dino-base
box threshold: 0.35
text threshold: 0.25
```

Place images under:

```text
modelv0.2/images_input/
```

Then run:

```bash
python3 scripts/autolabel.py
```

Existing label files are skipped by default. Use `--overwrite` only when you intentionally want to regenerate them.

The script keeps a local manifest so repeated runs do not copy the same source images again. Images with no retained detections receive an empty YOLO label file; the treatment of negative samples can therefore be decided later without silently deleting them.

After auto-labeling, manually review the important cases before using them for training, especially the `basketball` class and any frames that originated from hard-case extraction.

### Legacy CVAT helper

```bash
python3 scripts/CVAT_analysis.py
```

This is intentionally a different workflow: frames with no retained detections are moved to `discarded_frames`. Use it only when that destructive behaviour is desired.

## 6. Build the train/validation dataset

Once the images and labels are ready, run:

```bash
python3 scripts/split_dataset.py
```

The current implementation uses the existing **80/20 random split with seed 42**.

Expected generated structure:

```text
dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

`dataset/` is ignored by Git and remains local.

## 7. Audit the dataset before training

Always audit a newly prepared set before starting a training run:

```bash
python3 scripts/audit_dataset.py
```

The audit checks:

- image/label pairing;
- orphan label files;
- corrupt images;
- empty label files;
- invalid class IDs;
- YOLO coordinate ranges;
- bounding boxes crossing image boundaries;
- object counts by class.

The report is written to:

```text
runs/audit/dataset_audit.json
```

The command exits with a non-zero status when structural/label issues are found. Fix those issues before training.

## 8. Fine-tune a new model version

`training/train.py` creates a **new model candidate**. It is not needed to use v1.0.0.

Before training, make sure the machine has:

1. the baseline checkpoint configured in `configs/model.toml` (currently `model_v1.0.0.pt`);
2. the prepared local `dataset/`;
3. a clean audit result;
4. a chosen target version.

Review training parameters in:

```text
configs/train.toml
```

Then start a new version explicitly, for example:

```bash
python3 training/train.py --version 1.1.0
```

The target version is required deliberately to prevent accidental retraining.

The output run is versioned automatically:

```text
runs/detect/model_v1.1.0/weights/best.pt
```

That `best.pt` is only a **candidate** at this point.

## 9. Evaluate and promote the candidate

Evaluate the new checkpoint before replacing the current model:

```bash
python3 evaluation/evaluate.py \
  --weights runs/detect/model_v1.1.0/weights/best.pt \
  --split val
```

Evaluation stores plots plus `metrics_summary.json` under `runs/evaluation/`.

You should also run the candidate on representative product footage:

```bash
python3 scripts/ball_tracker.py \
  videos/game.mp4 \
  --weights runs/detect/model_v1.1.0/weights/best.pt
```

Compare its detection/tracking behaviour with v1.0.0 before promotion.

Only after deciding that the candidate is better should you:

1. rename/copy the selected checkpoint using the project version convention, for example `model_v1.1.0.pt`;
2. update `configs/model.toml` so `version` and `weights` point to the promoted model;
3. record the change in `CHANGELOG.md`.

## Complete new-data → training flow

The intended end-to-end workflow is:

```text
Download/export new images
        ↓
modelv0.2/images_input/
        ↓
auto-label / manual label review
        ↓
scripts/split_dataset.py
        ↓
scripts/audit_dataset.py
        ↓
training/train.py --version X.Y.Z
        ↓
evaluation/evaluate.py
        ↓
ball_tracker.py on representative videos
        ↓
promote checkpoint only if it improves the product
```

For ball-specific development, new data should preferably come from verified hard cases discovered with the current model:

```text
v1.0.0 → real games → tracking/hard-case extraction → reviewed labels → future training set
```

## Tests and CI

Run local unit tests with:

```bash
python3 -m unittest discover -s tests -v
```

GitHub Actions runs lightweight syntax and unit-test checks on pushes to `main` and on pull requests. It does not run training or require GPU/MPS.

## Apple Silicon

Automatic device selection prefers CUDA, then Apple MPS, then CPU. To check MPS manually:

```bash
python3 -c "import torch; print(torch.backends.mps.is_available())"
```

You can override device selection in inference/evaluation/tracking with `--device`, and training defaults can be changed in `configs/train.toml`.

## Decisions intentionally still open

The repository is prepared for continued product development with v1.0.0. These methodological choices remain open until real product evidence justifies a new model iteration:

- whether temporal tracking is sufficient for ball continuity;
- train/validation split strategy;
- whether/how to add a fixed test set;
- handling and proportion of negative samples;
- frame deduplication;
- Grounding DINO threshold tuning;
- dataset composition for a future v1.0.1 or v1.1.0.
