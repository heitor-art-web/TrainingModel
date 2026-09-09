import os

folder = "modelv0.2/shots_jpg"
prefix = "basket"

extensions = (".jpg", ".jpeg", ".png", ".webp")

files = sorted(
    f for f in os.listdir(folder)
    if f.lower().endswith(extensions)
)

for i, filename in enumerate(files, start=1):
    ext = os.path.splitext(filename)[1].lower()
    new_name = f"{prefix}_{i:06d}{ext}"

    os.rename(
        os.path.join(folder, filename),
        os.path.join(folder, new_name)
    )

print(f"Renamed {len(files)} files.")