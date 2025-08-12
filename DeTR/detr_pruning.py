import torch
import torch.nn as nn
import logging
import csv
import sys
import time
import json
from pathlib import Path
from transformers import DetrImageProcessor, DetrForObjectDetection
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch_pruning as tp
from torch.utils.data import Dataset, DataLoader

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("temp.txt", mode="w")]
)
logger = logging.getLogger(__name__)

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "facebook/detr-resnet-50"
BASE_DIR = Path.cwd()
COCO_DIR = BASE_DIR / "Dataset" / "COCO"
VAL_IMG_DIR = COCO_DIR / "val2017"
VAL_ANN_FILE = COCO_DIR / "annotations" / "instances_val2017.json"
TRAIN_IMG_DIR = COCO_DIR / "train2017"
TRAIN_ANN_FILE = COCO_DIR / "annotations" / "instances_train2017.json"
CSV_FILE = BASE_DIR / "detr_metrics.csv"

# Validate paths
for path in [VAL_ANN_FILE, VAL_IMG_DIR, TRAIN_ANN_FILE, TRAIN_IMG_DIR]:
    if not path.exists():
        logger.error(f"Path not found: {path}")
        raise FileNotFoundError(f"Path not found: {path}")

processor = DetrImageProcessor.from_pretrained(MODEL_NAME, revision="no_timm")

# COCO Dataset
class COCODataset(Dataset):
    def __init__(self, img_dir, ann_file, processor, max_labels=64, max_images=5000):
        self.img_dir = img_dir
        self.processor = processor
        self.max_labels = max_labels
        try:
            self.coco = COCO(str(ann_file))
            self.img_ids = self.coco.getImgIds()[:max_images]
            logger.info(f"Loaded {len(self.img_ids)} images from {ann_file}")
            logger.info(f"Valid COCO category IDs: {set(self.coco.getCatIds())}")
            sample_anns = self.coco.loadAnns(self.coco.getAnnIds(self.img_ids[:5]))
            logger.info(f"Sample annotations: {[{k: v for k, v in ann.items() if k in ['image_id', 'category_id', 'bbox']} for ann in sample_anns[:5]]}")
        except Exception as e:
            logger.error(f"Failed to load COCO annotations: {str(e)}")
            raise

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_info = self.coco.loadImgs(self.img_ids[idx])[0]
        img_path = self.img_dir / img_info['file_name']
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.info(f"Error loading image {img_path}: {str(e)}")
            image = Image.new("RGB", (224, 224), color=(0, 0, 0))
        ann_ids = self.coco.getAnnIds(imgIds=img_info['id'])
        anns = self.coco.loadAnns(ann_ids)

        img_width, img_height = image.size
        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            if w <= 0 or h <= 0:
                continue
            center_x = (x + w / 2) / img_width
            center_y = (y + h / 2) / img_height
            width = w / img_width
            height = h / img_height
            boxes.append([center_x, center_y, width, height])
            labels.append(ann['category_id'])
        if not boxes:
            boxes = [[0, 0, 0, 0]]
            labels = [0]
        if len(boxes) > self.max_labels:
            boxes = boxes[:self.max_labels]
            labels = labels[:self.max_labels]
        else:
            while len(boxes) < self.max_labels:
                boxes.append([0, 0, 0, 0])
                labels.append(0)
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs['boxes'] = boxes
        inputs['class_labels'] = labels
        logger.debug(f"Sample {img_info['id']}: pixel_values shape={inputs['pixel_values'].shape}, boxes shape={boxes.shape}, boxes sample={boxes[:2].tolist()}, labels={labels[:5].tolist()}")
        return inputs

def load_image(img_info):
    image_path = VAL_IMG_DIR / img_info['file_name']
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        logger.info(f"Error loading image {image_path}: {str(e)}")
        image = Image.new("RGB", (224, 224), color=(0, 0, 0))
    return image

def evaluate_model(model, image_ids, images, coco_val):
    model.eval()
    coco_predictions = []
    start = time.time()
    predicted_labels = set()

    detr_to_coco = {
        1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11,
        13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18, 19: 19, 20: 20, 21: 21,
        22: 22, 23: 23, 24: 24, 25: 25, 27: 27, 28: 28, 31: 31, 32: 32, 33: 33,
        34: 34, 35: 35, 36: 36, 37: 37, 38: 38, 39: 39, 40: 40, 41: 41, 42: 42,
        43: 43, 44: 44, 46: 46, 47: 47, 48: 48, 49: 49, 50: 50, 51: 51, 52: 52,
        53: 53, 54: 54, 55: 55, 56: 56, 57: 57, 58: 58, 59: 59, 60: 60, 61: 61,
        62: 62, 63: 63, 64: 64, 65: 65, 67: 67, 70: 70, 72: 72, 73: 73, 74: 74,
        75: 75, 76: 76, 77: 77, 78: 78, 79: 79, 80: 80, 81: 81, 82: 82, 84: 84,
        85: 85, 86: 86, 87: 87, 88: 88, 89: 89, 90: 90
    }

    for i, img_info in enumerate(tqdm(images, desc="Evaluating", file=sys.stdout)):
        try:
            image = load_image(img_info)
            img_width, img_height = image.size
            inputs = processor(images=image, return_tensors="pt").to(DEVICE)
            logger.debug(f"Image {img_info['id']}: pixel_values shape={inputs['pixel_values'].shape}, image size=({img_width}, {img_height})")
            with torch.no_grad():
                outputs = model(**inputs)
            if torch.isnan(outputs.logits).any() or torch.isnan(outputs.pred_boxes).any():
                logger.info(f"NaN detected in logits or boxes for image {img_info['id']}")
                continue
            raw_predictions = len(outputs.logits[0])
            logger.debug(f"Image {img_info['id']}: raw predictions={raw_predictions}, sample boxes={outputs.pred_boxes[0, :2].cpu().numpy().tolist()}")
            target_sizes = torch.tensor([[img_height, img_width]], device=DEVICE)  # [h, w]
            processed = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.0)
            if not processed or len(processed[0]["scores"]) == 0:
                logger.info(f"No predictions for image {img_info['id']} after post-processing")
                continue

            results = processed[0]
            logger.debug(f"Image {img_info['id']}: post-processed predictions={len(results['scores'])}, sample scores={results['scores'][:2].cpu().numpy().tolist() if len(results['scores']) > 0 else []}, sample boxes={results['boxes'][:2].cpu().numpy().tolist() if len(results['boxes']) > 0 else []}, sample labels={results['labels'][:2].cpu().numpy().tolist() if len(results['labels']) > 0 else []}")
            ann_ids = coco_val.getAnnIds(imgIds=img_info['id'])
            gt_anns = coco_val.loadAnns(ann_ids)
            logger.debug(f"Image {img_info['id']}: ground truth annotations={[{k: v for k, v in ann.items() if k in ['category_id', 'bbox']} for ann in gt_anns[:5]]}")
            prediction_count = 0
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                predicted_labels.add(label.item())
                if label.item() == 0:
                    logger.debug(f"Skipping background label 0 for image {img_info['id']}")
                    continue
                category_id = detr_to_coco.get(label.item())
                if category_id is None:
                    logger.info(f"Skipping unmapped label {label.item()} for image {img_info['id']}")
                    continue
                x_min, y_min, x_max, y_max = box.cpu().numpy()
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(img_width, x_max)
                y_max = min(img_height, y_max)
                w = x_max - x_min
                h = y_max - y_min
                if w <= 0 or h <= 0:
                    logger.info(f"Invalid box for image {img_info['id']}: [{x_min:.2f}, {y_min:.2f}, {x_max:.2f}, {y_max:.2f}]")
                    continue
                coco_predictions.append({
                    "image_id": img_info["id"],
                    "category_id": category_id,
                    "bbox": [float(x_min), float(y_min), float(w), float(h)],
                    "score": float(score.item())
                })
                prediction_count += 1
                if prediction_count >= 100:
                    break
            logger.debug(f"Image {img_info['id']}: {prediction_count} predictions, sample={coco_predictions[-min(5, len(coco_predictions)):]}")
            if (i + 1) % 1000 == 0:
                with open(f"detr_results_partial_{i + 1}.json", "w") as f:
                    json.dump(coco_predictions, f)
                logger.info(f"Saved partial predictions to detr_results_partial_{i + 1}.json")
                torch.cuda.empty_cache()
        except Exception as e:
            logger.info(f"Error processing image {img_info['id']}: {str(e)}")
            continue

    logger.info(f"Predicted labels: {sorted(predicted_labels)}")
    inference_time = time.time() - start
    logger.info(f"Total predictions generated: {len(coco_predictions)}")

    if not coco_predictions:
        logger.info("No predictions were made. Returning 0.0 for mAP.")
        return {
            "AP": 0.0, "AP50": 0.0, "AP75": 0.0, "APs": 0.0, "APm": 0.0, "APl": 0.0,
            "inference_time": inference_time
        }

    output_json = "detr_results.json"
    with open(output_json, "w") as f:
        json.dump(coco_predictions, f)
    logger.info(f"Saved final predictions to {output_json}")

    try:
        coco_dt = coco_val.loadRes(output_json)
        coco_eval = COCOeval(coco_val, coco_dt, "bbox")
        coco_eval.params.imgIds = image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    except Exception as e:
        logger.error(f"Error in COCO evaluation: {str(e)}")
        return {
            "AP": 0.0, "AP50": 0.0, "AP75": 0.0, "APs": 0.0, "APm": 0.0, "APl": 0.0,
            "inference_time": inference_time
        }

    torch.cuda.empty_cache()
    return {
        "AP": coco_eval.stats[0], "AP50": coco_eval.stats[1], "AP75": coco_eval.stats[2],
        "APs": coco_eval.stats[3], "APm": coco_eval.stats[4], "APl": coco_eval.stats[5],
        "inference_time": inference_time
    }

def prune_model(model, pruning_ratio=0.02):
    model.eval()
    dg = tp.DependencyGraph()
    example_inputs = torch.randn(1, 3, 224, 224).to(DEVICE)
    try:
        dg.build_dependency(model, example_inputs=example_inputs)
        logger.info("Dependency graph built successfully")
    except Exception as e:
        logger.error(f"Error building dependency graph: {str(e)}")
        return model

    logger.info("Model modules for pruning:")
    prunable_layers = []
    for name, module in model.named_modules():
        logger.debug(f"Module: {name}, Type: {type(module).__name__}")
        if isinstance(module, nn.Conv2d) and ('model.backbone.conv_encoder.model.layer1' in name or 'model.backbone.conv_encoder.model.layer2' in name):
            prunable_layers.append((name, module))
    logger.info(f"Prunable layers: {[name for name, _ in prunable_layers]}")

    if not prunable_layers:
        logger.warning("No prunable layers found. Skipping pruning.")
        return model

    ignored_layers = [
        model.class_labels_classifier,
        model.bbox_predictor,
        model.model.encoder,
        model.model.decoder
    ]

    try:
        pruner = tp.pruner.MagnitudePruner(
            model,
            example_inputs=example_inputs,
            importance=tp.importance.MagnitudeImportance(p=2),
            pruning_ratio=pruning_ratio,
            iterative_steps=1,
            ignored_layers=ignored_layers
        )
        pruner.step()
        logger.info(f"Pruned layers: {[name for name, _ in prunable_layers]}")
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Pruned model parameters: {total_params}")
        for name, module in prunable_layers:
            if isinstance(module, nn.Conv2d):
                logger.info(f"Layer {name}: in_channels={module.in_channels}, out_channels={module.out_channels}")
    except Exception as e:
        logger.error(f"Error during pruning: {str(e)}")
        return model

    try:
        with torch.no_grad():
            test_output = model(pixel_values=example_inputs)
        logger.info(f"Post-pruning validation: logits shape={test_output.logits.shape}, boxes={test_output.pred_boxes.shape}")
    except Exception as e:
        logger.error(f"Post-pruning validation failed: {str(e)}")
        return model

    return model

def fine_tune_model(model, dataset, num_epochs=10, accum_steps=4):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-5, total_steps=num_epochs * len(dataset) // accum_steps, pct_start=0.1)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x)
    logger.info(f"DataLoader created with {len(dataloader)} batches")

    for epoch in range(num_epochs):
        logger.info(f"Fine-tuning epoch {epoch + 1}/{num_epochs}")
        total_loss = 0.0
        num_batches = 0
        optimizer.zero_grad()
        for i, batch in enumerate(tqdm(dataloader, desc=f"Fine-tuning epoch {epoch + 1}", file=sys.stdout)):
            batch_inputs = []
            batch_labels = []
            try:
                for item in batch:
                    pixel_values = item['pixel_values'].unsqueeze(0).to(DEVICE)
                    pixel_mask = item.get('pixel_mask', None)
                    if pixel_mask is not None:
                        pixel_mask = pixel_mask.unsqueeze(0).to(DEVICE)
                    labels = {
                        'boxes': item['boxes'].to(DEVICE),
                        'class_labels': item['class_labels'].to(DEVICE)
                    }
                    batch_inputs.append({'pixel_values': pixel_values, 'pixel_mask': pixel_mask})
                    batch_labels.append(labels)
            except Exception as e:
                logger.info(f"Error preparing batch {i}: {str(e)}")
                continue

            losses = []
            for j, (inputs, labels) in enumerate(zip(batch_inputs, batch_labels)):
                try:
                    outputs = model(**inputs, labels=[labels])
                    if outputs.loss is None or torch.isnan(outputs.loss):
                        logger.info(f"Warning: Invalid loss for batch item {j}, labels={labels['class_labels'][:5].tolist()}")
                        continue
                    losses.append(outputs.loss / accum_steps)
                except Exception as e:
                    logger.info(f"Error in forward pass for batch item {j}: {str(e)}")
                    continue
            if not losses:
                logger.info(f"No valid losses in batch {i}, skipping...")
                continue
            loss = torch.stack(losses).sum()
            total_loss += loss.item() * accum_steps
            num_batches += 1
            loss.backward()
            if (i + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            if (i + 1) % 100 == 0:
                torch.cuda.empty_cache()
                logger.info(f"Processed {i + 1} batches in epoch {epoch + 1}, current loss: {loss.item() * accum_steps:.4f}")
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            logger.info(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")
            logger.info(f"Processed {num_batches} valid batches in epoch {epoch + 1}")
        else:
            logger.error(f"Epoch {epoch + 1}: No valid batches processed.")
            return
        torch.cuda.empty_cache()
        if avg_loss < 0.5:
            logger.info(f"Early stopping at epoch {epoch + 1}: average loss {avg_loss:.4f}")
            break

def save_metrics_to_csv(metrics, model_type):
    with open(CSV_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(['Model', 'AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'Inference_Time'])
        writer.writerow([
            model_type,
            f"{metrics['AP']:.4f}",
            f"{metrics['AP50']:.4f}",
            f"{metrics['AP75']:.4f}",
            f"{metrics['APs']:.4f}",
            f"{metrics['APm']:.4f}",
            f"{metrics['APl']:.4f}",
            f"{metrics['inference_time']:.2f}"
        ])

# Main execution
try:
    print("Starting script execution...")
    logger.info("Starting script execution")
    print("Loading COCO validation annotations...")
    coco_val = COCO(str(VAL_ANN_FILE))
    image_ids = coco_val.getImgIds()[:5000]
    images = coco_val.loadImgs(image_ids)
    logger.info(f"Loaded {len(image_ids)} validation images")

    if torch.cuda.is_available():
        logger.info(f"GPU memory before evaluation: {torch.cuda.memory_allocated(DEVICE)/1e9:.2f} GB used, {torch.cuda.memory_reserved(DEVICE)/1e9:.2f} GB reserved")

    print("Loading baseline DETR model...")
    model = DetrForObjectDetection.from_pretrained(MODEL_NAME, torch_dtype=torch.float32, revision="no_timm")
    model.to(DEVICE)

    print("Evaluating baseline DETR model...")
    baseline_metrics = evaluate_model(model, image_ids, images, coco_val)

    print("Baseline DETR Metrics:")
    print(f"AP (IoU=0.5:0.95): {baseline_metrics['AP']:.4f}")
    print(f"AP50 (IoU=0.5): {baseline_metrics['AP50']:.4f}")
    print(f"AP75 (IoU=0.75): {baseline_metrics['AP75']:.4f}")
    print(f"AP small: {baseline_metrics['APs']:.4f}")
    print(f"AP medium: {baseline_metrics['APm']:.4f}")
    print(f"AP large: {baseline_metrics['APl']:.4f}")
    print(f"Inference Time: {baseline_metrics['inference_time']:.2f} seconds")
    logger.info("Baseline DETR Metrics:")
    logger.info(f"AP (IoU=0.5:0.95): {baseline_metrics['AP']:.4f}")
    logger.info(f"AP50 (IoU=0.5): {baseline_metrics['AP50']:.4f}")
    logger.info(f"AP75 (IoU=0.75): {baseline_metrics['AP75']:.4f}")
    logger.info(f"AP small: {baseline_metrics['APs']:.4f}")
    logger.info(f"AP medium: {baseline_metrics['APm']:.4f}")
    logger.info(f"AP large: {baseline_metrics['APl']:.4f}")
    logger.info(f"Inference Time: {baseline_metrics['inference_time']:.2f} seconds")
    save_metrics_to_csv(baseline_metrics, "Baseline")

    torch.cuda.empty_cache()
    logger.info(f"GPU memory after baseline evaluation: {torch.cuda.memory_allocated(DEVICE)/1e9:.2f} GB used, {torch.cuda.memory_reserved(DEVICE)/1e9:.2f} GB reserved")

    print("Pruning model with 2% pruning ratio...")
    try:
        model = prune_model(model, pruning_ratio=0.05)
    except Exception as e:
        logger.error(f"Pruning failed: {str(e)}")
        print(f"Pruning failed: {str(e)}. Continuing with unpruned model...")

    torch.cuda.empty_cache()
    logger.info(f"GPU memory after pruning: {torch.cuda.memory_allocated(DEVICE)/1e9:.2f} GB used, {torch.cuda.memory_reserved(DEVICE)/1e9:.2f} GB reserved")

    print("Fine-tuning model for 10 epochs...")
    try:
        train_dataset = COCODataset(TRAIN_IMG_DIR, TRAIN_ANN_FILE, processor, max_labels=64, max_images=5000)
        fine_tune_model(model, train_dataset, num_epochs=10, accum_steps=4)
    except Exception as e:
        logger.error(f"Fine-tuning failed: {str(e)}")
        print(f"Fine-tuning failed: {str(e)}. Continuing with current model...")

    torch.cuda.empty_cache()
    logger.info(f"GPU memory after fine-tuning: {torch.cuda.memory_allocated(DEVICE)/1e9:.2f} GB used, {torch.cuda.memory_reserved(DEVICE)/1e9:.2f} GB reserved")

    print("Evaluating final DETR model...")
    final_metrics = evaluate_model(model, image_ids, images, coco_val)

    print("Final DETR Metrics:")
    print(f"AP (IoU=0.5:0.95): {final_metrics['AP']:.4f}")
    print(f"AP50 (IoU=0.5): {final_metrics['AP50']:.4f}")
    print(f"AP75 (IoU=0.75): {final_metrics['AP75']:.4f}")
    print(f"AP small: {final_metrics['APs']:.4f}")
    print(f"AP medium: {final_metrics['APm']:.4f}")
    print(f"AP large: {final_metrics['APl']:.4f}")
    print(f"Inference Time: {final_metrics['inference_time']:.2f} seconds")
    logger.info("Final DETR Metrics:")
    logger.info(f"AP (IoU=0.5:0.95): {final_metrics['AP']:.4f}")
    logger.info(f"AP50 (IoU=0.5): {final_metrics['AP50']:.4f}")
    logger.info(f"AP75 (IoU=0.75): {final_metrics['AP75']:.4f}")
    logger.info(f"AP small: {final_metrics['APs']:.4f}")
    logger.info(f"AP medium: {final_metrics['APm']:.4f}")
    logger.info(f"AP large: {final_metrics['APl']:.4f}")
    logger.info(f"Inference Time: {final_metrics['inference_time']:.2f} seconds")
    save_metrics_to_csv(final_metrics, "Final")

except Exception as e:
    logger.error(f"Main execution failed: {str(e)}")
    print(f"Main execution failed: {str(e)}")
finally:
    torch.cuda.empty_cache()
    logger.info(f"GPU memory after execution: {torch.cuda.memory_allocated(DEVICE)/1e9:.2f} GB used, {torch.cuda.memory_reserved(DEVICE)/1e9:.2f} GB reserved")
    print("Script execution completed.")
