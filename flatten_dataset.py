import os
import shutil

# Path to the root folder where your team/player folders are stored
source_root = "liga_betclic_players"  # e.g., "liga_betclic_photos"
target_img_dir = "images"

os.makedirs(target_img_dir, exist_ok=True)

count = 0
for root, dirs, files in os.walk(source_root):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            src_path = os.path.join(root, file)
            
            # Create a safe, unique filename using the folder structure (Team_Player_Filename)
            rel_path = os.path.relpath(src_path, source_root)
            new_filename = rel_path.replace(os.sep, "_")
            
            dst_path = os.path.join(target_img_dir, new_filename)
            shutil.copy(src_path, dst_path)
            count += 1

print(f"Successfully flattened {count} images into your main 'images/' folder!")