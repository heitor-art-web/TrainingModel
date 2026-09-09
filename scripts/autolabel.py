from pathlib import Path
import shutil

import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "modelv0.2"
SOURCE_FOLDERS = ["images_input"]
IMAGES_FOLDER = DATASET_ROOT / "images"
LABELS_FOLDER = DATASET_ROOT / "labels"

IMAGES_FOLDER.mkdir(parents=True, exist_ok=True)
LABELS_FOLDER.mkdir(parents=True, exist_ok=True)

CLASS_MAPPING = {
    "basketball": 0,
    "basketball player": 1,
    "referee": 2,
    "backboard": 3,
    "rim": 4,
}
TEXT_PROMPT = " . ".join(CLASS_MAPPING.keys()) + " ."

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print(f"Running on {DEVICE.upper()}")

processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
model = AutoModelForZeroShotObjectDetection.from_pretrained(
    "IDEA-Research/grounding-dino-base"
).to(DEVICE)

extensions = (".jpg", ".jpeg", ".png", ".webp")
counter = 1

print("\nMerging images...")
for folder in SOURCE_FOLDERS:
    src = DATASET_ROOT / folder
    if not src.exists():
        print(f"Missing folder: {src}")
        continue

    for file in sorted(src.iterdir()):
        if file.suffix.lower() not in extensions:
            continue

        new_name = f"{counter:06d}{file.suffix.lower()}"
        shutil.copy2(file, IMAGES_FOLDER / new_name)
        counter += 1

print(f"Copied {counter - 1} images.")

images = sorted(
    file for file in IMAGES_FOLDER.iterdir() if file.suffix.lower() in extensions
)
print(f"\nLabelling {len(images)} images...\n")

for idx, img_path in enumerate(images, start=1):
    txt_path = LABELS_FOLDER / f"{img_path.stem}.txt"
    image = Image.open(img_path).convert("RGB")
    width, height = image.size

    inputs = processor(images=image, text=TEXT_PROMPT, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        threshold=0.35,
        text_threshold=0.25,
        target_sizes=[(height, width)],
    )

    with txt_path.open("w") as label_file:
        for box, label in zip(results[0]["boxes"], results[0]["labels"]):
            detected = label.lower().strip()
            if detected not in CLASS_MAPPING:
                continue

            xmin, ymin, xmax, ymax = box.tolist()
            xc = ((xmin + xmax) / 2) / width
            yc = ((ymin + ymax) / 2) / height
            bw = (xmax - xmin) / width
            bh = (ymax - ymin) / height

            label_file.write(
                f"{CLASS_MAPPING[detected]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n"
            )

    if idx % 100 == 0:
        print(f"{idx}/{len(images)} images labelled...")

print("\nDataset complete.")
