from pathlib import Path
import shutil
import time

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT / "modelv0.2"
FRAME_DIR = ROOT / "frames"
LABEL_DIR = ROOT / "labels"
DISCARDED_DIR = ROOT / "discarded_frames"

LABEL_DIR.mkdir(parents=True, exist_ok=True)
DISCARDED_DIR.mkdir(parents=True, exist_ok=True)

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

CLASS_MAPPING = {
    "basketball": 0,
    "basketball player": 1,
    "referee": 2,
    "backboard": 3,
    "rim": 4,
}
TEXT_PROMPT = " . ".join(CLASS_MAPPING.keys()) + " ."

processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
model = AutoModelForZeroShotObjectDetection.from_pretrained(
    "IDEA-Research/grounding-dino-base"
).to(DEVICE).eval()

images = [
    file for file in FRAME_DIR.iterdir()
    if file.suffix.lower() in (".jpg", ".jpeg", ".png")
]

start = time.time()

for i, img_path in enumerate(tqdm(images), start=1):
    txt_path = LABEL_DIR / f"{img_path.stem}.txt"

    if txt_path.exists():
        continue

    img = Image.open(img_path).convert("RGB")
    width, height = img.size
    inputs = processor(images=img, text=TEXT_PROMPT, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        threshold=0.35,
        text_threshold=0.25,
        target_sizes=[(height, width)],
    )[0]

    kept = False
    with txt_path.open("w") as label_file:
        for box, label in zip(results["boxes"], results["labels"]):
            label = label.lower().strip()
            if label not in CLASS_MAPPING:
                continue

            kept = True
            xmin, ymin, xmax, ymax = box.tolist()
            xc = ((xmin + xmax) / 2) / width
            yc = ((ymin + ymax) / 2) / height
            bw = (xmax - xmin) / width
            bh = (ymax - ymin) / height
            label_file.write(
                f"{CLASS_MAPPING[label]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n"
            )

    if not kept:
        txt_path.unlink()
        shutil.move(str(img_path), str(DISCARDED_DIR / img_path.name))

    if i % 100 == 0:
        avg = (time.time() - start) / i
        eta = (len(images) - i) * avg / 60
        print(f"ETA: {eta:.1f} min")
