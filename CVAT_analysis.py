# Grounding DINO Auto Label v2 (skeleton)
# Improvements:
# - CUDA/MPS/CPU auto selection
# - Single inference pass
# - Save YOLO labels
# - Move images with no detections to discarded_frames
# - ETA
# NOTE: Batch inference placeholder (BATCH_SIZE) ready for future implementation.

import os, shutil, time, torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

ROOT=os.path.expanduser("~/Documents/VideoAnalysis/Model_Dictionary/modelv0.2")
FRAME_DIR=os.path.join(ROOT,"frames")
LABEL_DIR=os.path.join(ROOT,"labels")
DISCARDED_DIR=os.path.join(ROOT,"discarded_frames")
os.makedirs(LABEL_DIR,exist_ok=True)
os.makedirs(DISCARDED_DIR,exist_ok=True)

if torch.cuda.is_available():
    DEVICE="cuda"
elif torch.backends.mps.is_available():
    DEVICE="mps"
else:
    DEVICE="cpu"

CLASS_MAPPING={
    "basketball":0,
    "basketball player":1,
    "referee":2,
    "backboard":3,
    "basketball rim":4,
}

TEXT_PROMPT=" . ".join(CLASS_MAPPING.keys())+" ."

processor=AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
model=AutoModelForZeroShotObjectDetection.from_pretrained(
    "IDEA-Research/grounding-dino-base"
).to(DEVICE).eval()

images=[f for f in os.listdir(FRAME_DIR)
        if f.lower().endswith((".jpg",".jpeg",".png"))]

start=time.time()

for i,name in enumerate(tqdm(images),1):

    img_path=os.path.join(FRAME_DIR,name)
    txt_path=os.path.join(LABEL_DIR,os.path.splitext(name)[0]+".txt")

    if os.path.exists(txt_path):
        continue

    img=Image.open(img_path).convert("RGB")
    w,h=img.size

    inputs=processor(images=img,text=TEXT_PROMPT,return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs=model(**inputs)

    results=processor.post_process_grounded_object_detection(
        outputs,
        threshold=0.35,
        text_threshold=0.25,
        target_sizes=[(h,w)]
    )[0]

    kept=False

    with open(txt_path,"w") as f:
        for box,label in zip(results["boxes"],results["labels"]):
            label=label.lower().strip()
            if label not in CLASS_MAPPING:
                continue
            kept=True
            xmin,ymin,xmax,ymax=box.tolist()
            xc=((xmin+xmax)/2)/w
            yc=((ymin+ymax)/2)/h
            bw=(xmax-xmin)/w
            bh=(ymax-ymin)/h
            f.write(f"{CLASS_MAPPING[label]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

    if not kept:
        os.remove(txt_path)
        shutil.move(img_path,os.path.join(DISCARDED_DIR,name))

    if i%100==0:
        avg=(time.time()-start)/i
        eta=(len(images)-i)*avg/60
        print(f"ETA: {eta:.1f} min")