import os
import shutil
import random

# Source directories
img_dir = "modelv0.2/frames"
lbl_dir = "modelv0.2/labels"
base_dir = "dataset"

# 1. Wipe old dataset folder to ensure clean split
if os.path.exists(base_dir):
    shutil.rmtree(base_dir)

# 2. Recreate target dataset structure
for split in ['train', 'val']:
    os.makedirs(f"{base_dir}/images/{split}", exist_ok=True)
    os.makedirs(f"{base_dir}/labels/{split}", exist_ok=True)

# 3. Get all valid images that have matching labels
images = [
    f for f in os.listdir(img_dir) 
    if f.lower().endswith(('.jpg', '.jpeg', '.png')) 
    and os.path.exists(os.path.join(lbl_dir, os.path.splitext(f)[0] + ".txt"))
]

random.seed(42)  # Consistent shuffle
random.shuffle(images)

split_idx = int(len(images) * 0.8)
train_imgs = images[:split_idx]
val_imgs = images[split_idx:]

def copy_files(file_list, split_type):
    for img_name in file_list:
        base_name = os.path.splitext(img_name)[0]
        lbl_name = f"{base_name}.txt"
        
        shutil.copy(os.path.join(img_dir, img_name), f"{base_dir}/images/{split_type}/{img_name}")
        shutil.copy(os.path.join(lbl_dir, lbl_name), f"{base_dir}/labels/{split_type}/{lbl_name}")

copy_files(train_imgs, 'train')
copy_files(val_imgs, 'val')

print(f"Split complete! Train: {len(train_imgs)} images | Val: {len(val_imgs)} images.")