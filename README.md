# TrainingModel

Basketball object-detection project built around Ultralytics YOLO, with Grounding DINO helpers for preparing new labelled data.

## Current model

The current promoted model is **v1.0.0**.

Its default local checkpoint path is configured in `configs/model.toml`:

```text
runs/detect/yolo26b/weights/model_v1.0.0.pt
```

Model weights, datasets and generated runs are intentionally excluded from Git. A cloned repository therefore contains the code/configuration, but you must copy the required `.pt` file and dataset to the machine where they are needed.

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
│   ├── model.toml
│   └── train.toml
├── evaluation/
│   └── evaluate.py
├── scripts/
│   ├── audit_dataset.py
│   ├── autolabel.py
│   ├── CVAT_analysis.py
│   ├── flatten_dataset.py
│   ├── inference.py
│   ├── rename.py
│   └── split_dataset.py
├── training/
│   ├── __init__.py
│   ├── common.py
│   └── train.py
├── dataset.yaml
├── requirements.txt
└── README.md
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
python3 scripts/inference.py /path/to/video.mp4
```

Useful options:

```bash
python3 scripts/inference.py /path/to/image.jpg --conf 0.35
python3 scripts/inference.py /path/to/video.mp4 --no-save
python3 scripts/inference.py /path/to/video.mp4 --weights /path/to/another_model.pt
```

Default confidence, IoU and image size are defined in `configs/model.toml`. Annotated outputs are saved under `runs/inference/` unless `--no-save` is used.

## 2. Model configuration and versioning

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

## 3. Prepare new data

The existing helper scripts are kept for the current data-preparation workflow.

If source images are nested by team/player:

```bash
python3 scripts/flatten_dataset.py
```

### Auto-label with Grounding DINO

The auto-label configuration is in `configs/autolabel.toml`.

Current defaults are:

```text
model: IDEA-Research/grounding-dino-base
box threshold: 0.35
text threshold: 0.25
```

Run:

```bash
python3 scripts/autolabel.py
```

Existing label files are skipped by default. Use `--overwrite` only when you intentionally want to regenerate them.

The script keeps a local manifest so repeated runs do not copy the same source images again. Images with no retained detections receive an empty YOLO label file; the treatment of negative samples can therefore be decided later without silently deleting them.

### Legacy CVAT helper

```bash
python3 scripts/CVAT_analysis.py
```

This is intentionally a different workflow: frames with no retained detections are moved to `discarded_frames`. Use it only when that destructive behaviour is desired.

## 4. Build the current train/validation dataset

```bash
python3 scripts/split_dataset.py
```

The current implementation still uses the existing **80/20 random split with seed 42**. No change to the split methodology has been made yet.

Expected local structure:

```text
dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

## 5. Audit the dataset before training

Run:

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

The command exits with a non-zero status when structural/label issues are found.

## 6. Evaluate a model

Evaluate the current promoted model on validation data:

```bash
python3 evaluation/evaluate.py --split val
```

Evaluate another checkpoint without promoting it:

```bash
python3 evaluation/evaluate.py --weights /path/to/candidate.pt --split val
```

Evaluation uses Ultralytics validation and stores plots plus `metrics_summary.json` under `runs/evaluation/`.

A `test` split can be used later if/when `dataset.yaml` defines one:

```bash
python3 evaluation/evaluate.py --split test
```

## 7. Fine-tune a new version

`training/train.py` exists only for creating a **new model iteration**. It is not needed for inference with v1.0.0.

Before training, you need locally:

1. baseline weights (by default `model_v1.0.0.pt`);
2. a prepared dataset matching `dataset.yaml`;
3. a chosen target version.

Example:

```bash
python3 training/train.py --version 1.1.0
```

The target version is required deliberately to prevent accidental retraining. Training parameters live in `configs/train.toml`.

The output run is versioned automatically:

```text
runs/detect/model_v1.1.0/weights/best.pt
```

That `best.pt` is a **candidate**. Evaluate it before renaming/promoting it as the new current model.

The code materializes an absolute dataset path at runtime, so a relative `path: dataset` in `dataset.yaml` is resolved against this repository rather than the global Ultralytics datasets directory.

## Apple Silicon

Automatic device selection prefers CUDA, then Apple MPS, then CPU. To check MPS manually:

```bash
python3 -c "import torch; print(torch.backends.mps.is_available())"
```

You can override device selection in inference/evaluation with `--device`, and training defaults can be changed in `configs/train.toml`.

## Decisions intentionally still open

The repository is now prepared for the next model iteration, but these methodological choices have **not** been changed yet:

- train/validation split strategy;
- whether/how to add a fixed test set;
- handling and proportion of negative samples;
- frame deduplication;
- Grounding DINO threshold tuning;
- dataset composition for v1.0.1 vs v1.1.0;
- fine-tuning hyperparameters/augmentations/freezing strategy.

These should be decided before the next training run rather than mixed into the repository cleanup.
