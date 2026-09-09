from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "liga_betclic_players"
TARGET_IMG_DIR = PROJECT_ROOT / "images"

TARGET_IMG_DIR.mkdir(parents=True, exist_ok=True)

count = 0
for src_path in SOURCE_ROOT.rglob("*"):
    if not src_path.is_file() or src_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
        continue

    rel_path = src_path.relative_to(SOURCE_ROOT)
    new_filename = "_".join(rel_path.parts)
    shutil.copy(src_path, TARGET_IMG_DIR / new_filename)
    count += 1

print(f"Successfully flattened {count} images into {TARGET_IMG_DIR}.")
