from pathlib import Path
import random
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = PROJECT_ROOT / "modelv0.2" / "frames"
LBL_DIR = PROJECT_ROOT / "modelv0.2" / "labels"
BASE_DIR = PROJECT_ROOT / "dataset"

if BASE_DIR.exists():
    shutil.rmtree(BASE_DIR)

for split in ("train", "val"):
    (BASE_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

images = [
    file for file in IMG_DIR.iterdir()
    if file.is_file()
    and file.suffix.lower() in (".jpg", ".jpeg", ".png")
    and (LBL_DIR / f"{file.stem}.txt").exists()
]

random.seed(42)
random.shuffle(images)

split_idx = int(len(images) * 0.8)
train_imgs = images[:split_idx]
val_imgs = images[split_idx:]


def copy_files(file_list, split_type):
    for img_path in file_list:
        lbl_path = LBL_DIR / f"{img_path.stem}.txt"
        shutil.copy(img_path, BASE_DIR / "images" / split_type / img_path.name)
        shutil.copy(lbl_path, BASE_DIR / "labels" / split_type / lbl_path.name)


copy_files(train_imgs, "train")
copy_files(val_imgs, "val")

print(f"Split complete! Train: {len(train_imgs)} images | Val: {len(val_imgs)} images.")
