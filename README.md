# TrainingModel

Training pipeline for basketball object detection with YOLO.

## Classes

The dataset currently uses five classes:

| ID | Class |
| --- | --- |
| 0 | basketball |
| 1 | player |
| 2 | referee |
| 3 | backboard |
| 4 | rim |

The class name `rim` is used consistently across the dataset configuration and auto-labeling scripts.

## Project structure

```text
TrainingModel/
├── README.md
├── dataset.yaml
├── scripts/
│   ├── autolabel.py
│   ├── CVAT_analysis.py
│   ├── flatten_dataset.py
│   ├── rename.py
│   └── split_dataset.py
└── training/
    └── train.py
```

Large datasets, model weights and training outputs are intentionally excluded from Git.

## Current workflow

### 1. Prepare images

If the source images are nested by team/player, flatten them with:

```bash
python3 scripts/flatten_dataset.py
```

### 2. Auto-label images

Grounding DINO is used to create YOLO-format labels for:

- basketball
- basketball player
- referee
- backboard
- rim

Run:

```bash
python3 scripts/autolabel.py
```

The alternative `scripts/CVAT_analysis.py` processes frames in `modelv0.2/frames`, skips frames that already have labels, and moves frames with no retained detections to `modelv0.2/discarded_frames`.

### 3. Build train/validation folders

Run:

```bash
python3 scripts/split_dataset.py
```

This currently creates an 80/20 train/validation split using a fixed random seed.

The generated structure is:

```text
dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

### 4. Verify Apple MPS support

Before training on Apple Silicon:

```bash
python3 -c "import torch; print(torch.backends.mps.is_available())"
```

If this prints `True`, the current training configuration can use `device="mps"`.

### 5. Train / fine-tune

The current training entry point fine-tunes from:

```text
runs/detect/yolo26b/weights/best.pt
```

Run:

```bash
python3 training/train.py
```

Current settings include 50 epochs, image size 768, batch size 8, learning rate 0.0005, MPS, disk cache and early stopping with patience 15.

The resulting best checkpoint is expected at:

```text
runs/detect/yolo26b/weights/best.pt
```

## Notes

The current workflow intentionally preserves the existing training and dataset methodology. Further changes to dataset cleaning, splitting strategy, negative samples, deduplication and training experiments should be discussed and introduced separately.
