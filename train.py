import os
from ultralytics import YOLO

# Path to original model weights (your initial basketball_model run)
PREVIOUS_WEIGHTS = "runs/detect/yolo26b/weights/best.pt"

if __name__ == '__main__':
    if not os.path.exists(PREVIOUS_WEIGHTS):
        raise FileNotFoundError(
            f"Could not find baseline weights at {PREVIOUS_WEIGHTS}. "
            "Verify the path before fine-tuning!"
        )

    print(f"Loading baseline weights from: {PREVIOUS_WEIGHTS}")
    model = YOLO(PREVIOUS_WEIGHTS)

    # Train and output results under the new run name 'yolo26b'
    results = model.train(
        data="dataset.yaml",
        epochs=50,          # 20–30 is enough for a second pass
        imgsz=768,
        batch=8,           # or 8/16 if you know what fits
        lr0=0.0005,         # lower than the first fine-tune
        device="mps",
        workers=4,
        cache="disk",
        plots=True,
        save=True,
        patience=15,        # stop early if validation stops improving
        name="yolo26b",
        exist_ok=True,
    )

    print("Fine-tuning complete!")
    print("New weights saved to: /weights/yolo26b")