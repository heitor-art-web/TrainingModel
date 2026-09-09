from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FOLDER = PROJECT_ROOT / "modelv0.2" / "shots_jpg"
PREFIX = "basket"
EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

files = sorted(
    file for file in FOLDER.iterdir()
    if file.is_file() and file.suffix.lower() in EXTENSIONS
)

for i, file in enumerate(files, start=1):
    new_name = FOLDER / f"{PREFIX}_{i:06d}{file.suffix.lower()}"
    file.rename(new_name)

print(f"Renamed {len(files)} files.")
