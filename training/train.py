from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREVIOUS_WEIGHTS = PROJECT_ROOT / "runs" / "detect" / "yolo26b" / "weights" / "best.pt"
DATASET_CONFIG = PROJECT_ROOT / "dataset.yaml"
RUNS_DIR = PROJECT_ROOT / "runs" / "detect"
RUN_NAME = "yolo26b"

if __name__ == "__main__":
    if not PREVIOUS_WEIGHTS.exists():
        raise FileNotFoundError(
            f"Could not find baseline weights at {PREVIOUS_WEIGHTS}. "
            "Verify the path before fine-tuning."
        )

    print(f"Loading baseline weights from: {PREVIOUS_WEIGHTS}")
    model = YOLO(str(PREVIOUS_WEIGHTS))

    model.train(
        data=str(DATASET_CONFIG),
        epochs=50,
        imgsz=768,
        batch=8,
        lr0=0.0005,
        device="mps",
        workers=4,
        cache="disk",
        plots=True,
        save=True,
        patience=15,
        project=str(RUNS_DIR),
        name=RUN_NAME,
        exist_ok=True,
    )

    output_weights = RUNS_DIR / RUN_NAME / "weights" / "best.pt"
    print("Fine-tuning complete!")
    print(f"Best weights saved to: {output_weights}")
