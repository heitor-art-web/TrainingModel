import os
import shutil
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------

dataset_root = "modelv0.2"

source_folders = [
    "images_input",
]

images_folder = os.path.join(dataset_root, "images")
labels_folder = os.path.join(dataset_root, "labels")

os.makedirs(images_folder, exist_ok=True)
os.makedirs(labels_folder, exist_ok=True)

# -------------------------------------------------------
# Grounding DINO
# -------------------------------------------------------

CLASS_MAPPING = {
    "basketball": 0,
    "basketball player": 1,
    "referee": 2,
    "backboard": 3,
    "rim": 4,
}

TEXT_PROMPT = " . ".join(CLASS_MAPPING.keys()) + " ."

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Running on {device.upper()}")

processor = AutoProcessor.from_pretrained(
    "IDEA-Research/grounding-dino-base"
)

model = AutoModelForZeroShotObjectDetection.from_pretrained(
    "IDEA-Research/grounding-dino-base"
).to(device)

# -------------------------------------------------------
# Merge all datasets
# -------------------------------------------------------

print("\nMerging images...")

extensions = (".jpg", ".jpeg", ".png", ".webp")

counter = 1

for folder in source_folders:

    src = os.path.join(dataset_root, folder)

    if not os.path.exists(src):
        print(f"Missing folder: {src}")
        continue

    files = sorted(os.listdir(src))

    for file in files:

        if not file.lower().endswith(extensions):
            continue

        ext = os.path.splitext(file)[1].lower()

        new_name = f"{counter:06d}{ext}"

        shutil.copy2(
            os.path.join(src, file),
            os.path.join(images_folder, new_name)
        )

        counter += 1

print(f"Copied {counter-1} images.")

# -------------------------------------------------------
# Auto-label
# -------------------------------------------------------

images = sorted([
    f for f in os.listdir(images_folder)
    if f.lower().endswith(extensions)
])

print(f"\nLabelling {len(images)} images...\n")

for idx, filename in enumerate(images, start=1):

    img_path = os.path.join(images_folder, filename)

    txt_path = os.path.join(
        labels_folder,
        os.path.splitext(filename)[0] + ".txt"
    )

    image = Image.open(img_path).convert("RGB")

    w, h = image.size

    inputs = processor(
        images=image,
        text=TEXT_PROMPT,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        threshold=0.35,
        text_threshold=0.25,
        target_sizes=[(h, w)],
    )

    with open(txt_path, "w") as f:

        for box, label in zip(
            results[0]["boxes"],
            results[0]["labels"]
        ):

            detected = label.lower().strip()

            if detected not in CLASS_MAPPING:
                continue

            class_id = CLASS_MAPPING[detected]

            xmin, ymin, xmax, ymax = box.tolist()

            xc = ((xmin + xmax) / 2) / w
            yc = ((ymin + ymax) / 2) / h
            bw = (xmax - xmin) / w
            bh = (ymax - ymin) / h

            f.write(
                f"{class_id} "
                f"{xc:.6f} "
                f"{yc:.6f} "
                f"{bw:.6f} "
                f"{bh:.6f}\n"
            )

    if idx % 100 == 0:
        print(f"{idx}/{len(images)} images labelled...")

print("\nDataset complete.")