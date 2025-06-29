import os
import json
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import DetrImageProcessor, DetrForObjectDetection
from torchvision.transforms import ToTensor
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
from itertools import chain


# --- Config ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # Thesis/
COCO_DIR = BASE_DIR / "Dataset" / "COCO"
TRAIN_IMG_DIR = COCO_DIR / "train2017" / "train2017"
VAL_IMG_DIR = COCO_DIR / "val2017" / "val2017"
VAL_ANN_FILE = COCO_DIR / "annotations_trainval2017" / "annotations" / "instances_val2017.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# --- Load Model and Processor ---
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(DEVICE)
model.eval()

# --- Load COCO Ground Truth ---
print("\n Loading COCO validation annotations...")
coco_gt = COCO(VAL_ANN_FILE)
image_ids = list(sorted(coco_gt.imgs.keys()))

# --- Build accurate label → category_id mapping using label names ---
hf_id2label = model.config.id2label 
label_to_cat_id = {}

for label_id, label_name in hf_id2label.items():
    if label_name == "__background__":
        continue
    cat_ids = coco_gt.getCatIds(catNms=[label_name])
    if cat_ids:
        label_to_cat_id[label_id] = cat_ids[0]  # Correct COCO category ID

# --- Collect Results for COCO Evaluation ---
results_list = []

print("\n Running COCO evaluation on val2017...")
for img_id in tqdm(image_ids[:500], desc="Evaluating on val images"):  # Use full list for full eval
    img_info = coco_gt.loadImgs(img_id)[0]
    img_path = VAL_IMG_DIR / img_info["file_name"]
    if not img_path.exists():
        continue

    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    target_size = torch.tensor([image.size[::-1]]).to(DEVICE)
    predictions = processor.post_process_object_detection(outputs, target_sizes=target_size, threshold=0.0)[0]

    for score, label, box in zip(predictions["scores"], predictions["labels"], predictions["boxes"]):
        category_id = label_to_cat_id.get(label.item())
        if category_id is None:
            continue  # skip labels not in COCO

        xmin, ymin, xmax, ymax = box.tolist()
        width = xmax - xmin
        height = ymax - ymin

        results_list.append({
            "image_id": img_id,
            "category_id": category_id,
            "bbox": [xmin, ymin, width, height],
            "score": score.item()
        })

# --- Save Results to JSON ---
output_json = "detr_results.json"
with open(output_json, "w") as f:
    json.dump(results_list, f)

# --- COCO Evaluation ---
print("\nEvaluating predictions using pycocotools...")
coco_dt = coco_gt.loadRes(output_json)
coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
coco_eval.evaluate()
coco_eval.accumulate()
print("\n Evaluation Results (COCO Metrics):")
coco_eval.summarize()