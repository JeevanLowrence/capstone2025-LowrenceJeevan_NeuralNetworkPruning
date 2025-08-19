from pathlib import Path
import csv
import torch
import torch.nn as nn
import torch_pruning as tp
from transformers import DetrForObjectDetection, DetrImageProcessor
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from tqdm import tqdm
from torchvision.datasets import CocoDetection
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
from PIL import Image
import time
import os
import pkg_resources

try:
    from torchinfo import summary
    TORCHINFO_AVAILABLE = True
except ImportError:
    TORCHINFO_AVAILABLE = False

# ------------------
# Global Configuration
# ------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "facebook/detr-resnet-50"
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # Thesis/
COCO_DIR = BASE_DIR / "Dataset" / "COCO"
VAL_IMG_DIR = COCO_DIR / "val2017"
VAL_ANN_FILE = COCO_DIR / "annotations" / "instances_val2017.json"
TRAIN_IMG_DIR = COCO_DIR / "train2017"
TRAIN_ANN_FILE = COCO_DIR / "annotations" / "instances_train2017.json"
CSV_FILE = BASE_DIR / "detr_metrics_final.csv"

INPUT_SIZE = (1, 3, 800, 800)  # Batch, Channels, Height, Width
TRAIN_SUBSET_SIZE = 5000
VAL_SUBSET_SIZE = 5000
FINE_TUNING_EPOCHS = 2 


def save_metrics_to_csv(model_type, pruning_method, pruning_percentage, ap, ap_drop, inference_time, model_size, flops):
    """Save evaluation metrics for each model configuration into a CSV file."""
    with open(CSV_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(['Model', 'Pruning_Method', 'Pruning_Percentage', 'AP', 'AP_Drop', 'Inference_Time', 'Model_Size', 'FLOPs'])
        writer.writerow([model_type, pruning_method, pruning_percentage, f"{ap:.4f}", f"{ap_drop:.4f}", f"{inference_time:.2f}", f"{model_size:.3f}", f"{flops:.3f}"])


def count_params_flops(model, pixel_values):
    """Estimate the number of parameters (in millions) and FLOPs (in billions) of the model."""
    try:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        params_M = total_params / 1e6
        flops_G = 0.0

        if TORCHINFO_AVAILABLE:
            try:
                summary_stats = summary(model, input_data={'pixel_values': pixel_values}, verbose=0)
                params_M = summary_stats.total_params / 1e6
                flops_G = summary_stats.total_mult_adds / 1e9
            except Exception:
                pass

        return params_M, flops_G
    except Exception:
        return 0.0, 0.0


def get_layers_to_ignore(model):
    """Return layers that should not be pruned (e.g. input/output sensitive layers)."""
    ignored_layers = []

    # First + last conv layers in backbone are critical
    backbone_conv_layers = [l for _, l in model.model.backbone.named_modules() if isinstance(l, nn.Conv2d)]
    if backbone_conv_layers:
        ignored_layers.append(backbone_conv_layers[0])
        if len(backbone_conv_layers) > 1:
            ignored_layers.extend(backbone_conv_layers[-2:])

    # Avoid pruning encoder/decoder + prediction heads
    if model.model.encoder:
        ignored_layers.append(model.model.encoder)
    if model.model.decoder:
        ignored_layers.append(model.model.decoder)
    if hasattr(model, 'bbox_predictor'):
        ignored_layers.append(model.bbox_predictor)
    if hasattr(model, 'class_labels_classifier'):
        ignored_layers.append(model.class_labels_classifier)

    return ignored_layers


def prune_and_analyze(pruning_ratio, pruning_method, dummy_input):
    """
    Apply pruning to the DETR model using the given method and ratio.
    Returns the pruned model along with its new parameter and FLOP counts.
    """
    model_to_prune = DetrForObjectDetection.from_pretrained(MODEL_NAME, revision="no_timm").to(DEVICE)
    model_to_prune.eval()

    try:
        pixel_mask = torch.ones(INPUT_SIZE[0], INPUT_SIZE[2], INPUT_SIZE[3], dtype=torch.long).to(DEVICE)
        example_inputs = {'pixel_values': dummy_input.to(DEVICE), 'pixel_mask': pixel_mask.to(DEVICE)}

        ignored_layers = get_layers_to_ignore(model_to_prune)

        # Group channels for structured pruning
        channel_groups = {}
        for _, module in model_to_prune.model.backbone.named_modules():
            if isinstance(module, nn.Conv2d):
                channel_groups[module] = module.out_channels // 4
        if model_to_prune.model.input_projection:
            channel_groups[model_to_prune.model.input_projection] = 256

        # Select pruning strategy
        if pruning_method == "L1":
            importance_fn = tp.importance.MagnitudeImportance(p=1)
        elif pruning_method == "L2":
            importance_fn = tp.importance.MagnitudeImportance(p=2)
        elif pruning_method == "RANDOM":
            importance_fn = tp.importance.RandomImportance()
        else:
            raise ValueError(f"Unknown pruning method: {pruning_method}")

        pruner = tp.pruner.MagnitudePruner(
            model_to_prune,
            example_inputs=example_inputs,
            importance=importance_fn,
            ignored_layers=ignored_layers,
            channel_groups=channel_groups,
            iterative_steps=3,
            global_pruning=True,
            ch_sparsity=pruning_ratio / 3
        )

        # Actually prune the model
        pruner.step()

        new_params, new_flops = count_params_flops(model_to_prune, dummy_input)
        return model_to_prune, new_params, new_flops

    except Exception:
        return None, 0.0, 0.0


class CocoDetectionWithProcessor(CocoDetection):
    """COCO dataset wrapper that applies the DETR image processor to each sample."""

    def __init__(self, root, annFile, processor):
        super().__init__(root, annFile)
        self.processor = processor

    def __getitem__(self, idx):
        img, anns = super().__getitem__(idx)
        image_id = self.ids[idx]
        annotations = {'image_id': image_id, 'annotations': anns}
        encoding = self.processor(images=img, annotations=annotations, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]
        return pixel_values, target


def collate_fn(batch):
    """Collate function to pad variable-sized images and batch labels together."""
    pixel_values = [item[0] for item in batch]
    encoding = processor.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    return {
        'pixel_values': encoding['pixel_values'],
        'pixel_mask': encoding['pixel_mask'],
        'labels': labels
    }


def fine_tune_model(model, processor, device, epochs=FINE_TUNING_EPOCHS):
    """Fine-tune the pruned model for a few epochs on a COCO subset."""
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=1e-5)

    train_dataset = CocoDetectionWithProcessor(root=str(TRAIN_IMG_DIR), annFile=str(TRAIN_ANN_FILE), processor=processor)
    train_indices = list(range(min(TRAIN_SUBSET_SIZE, len(train_dataset))))
    train_subset = Subset(train_dataset, train_indices)
    train_dataloader = DataLoader(train_subset, collate_fn=collate_fn, batch_size=1, shuffle=True)

    model.train()
    for epoch in range(epochs):
        train_loss = 0.0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}"):
            pixel_values = batch["pixel_values"].to(device)
            pixel_mask = batch["pixel_mask"].to(device)
            labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
            loss = outputs.loss
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f"Epoch {epoch + 1}: Train loss {train_loss / len(train_dataloader):.4f}")


def evaluate_model(model, processor, device):
    """
    Evaluate a model on the COCO validation set.
    Returns mean Average Precision (AP) and average inference time.
    """
    model.to(device)
    model.eval()

    try:
        coco = COCO(str(VAL_ANN_FILE))
        img_ids = coco.getImgIds()[:VAL_SUBSET_SIZE]

        label_to_cat_id = {}
        for label_id, label_name in model.config.id2label.items():
            if label_name == "__background__":
                continue
            cat_ids = coco.getCatIds(catNms=[label_name])
            if cat_ids:
                label_to_cat_id[label_id] = cat_ids[0]

        predictions = []
        inference_times = []

        for img_id in tqdm(img_ids, desc="Evaluating"):
            img_info = coco.loadImgs(img_id)[0]
            img_path = os.path.join(str(VAL_IMG_DIR), img_info['file_name'])
            if not os.path.exists(img_path):
                continue
            image = Image.open(img_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)

            start_time = time.time()
            outputs = model(**inputs)
            inference_times.append(time.time() - start_time)

            target_sizes = torch.tensor([image.size[::-1]]).to(device)
            results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.1)[0]

            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                category_id = label_to_cat_id.get(label.item())
                if category_id is None:
                    continue
                box = [round(i, 2) for i in box.tolist()]
                xmin, ymin, xmax, ymax = box
                predictions.append({
                    "image_id": img_id,
                    "category_id": category_id,
                    "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
                    "score": score.item()
                })

        predictions_file = "predictions.json"
        with open(predictions_file, "w") as f:
            json.dump(predictions, f)

        coco_dt = coco.loadRes(predictions_file)
        coco_eval = COCOeval(coco, coco_dt, "bbox")
        coco_eval.params.imgIds = img_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        ap = coco_eval.stats[0]  # AP @ IoU=0.50:0.95
        average_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0.0

        return ap, average_inference_time

    except Exception:
        return 0.0, 0.0


if __name__ == "__main__":
    processor = DetrImageProcessor.from_pretrained(MODEL_NAME)
    baseline_model = DetrForObjectDetection.from_pretrained(MODEL_NAME, revision="no_timm").to(DEVICE)
    dummy_input = torch.randn(*INPUT_SIZE).to(DEVICE)

    # Baseline evaluation
    baseline_params, baseline_flops = count_params_flops(baseline_model, dummy_input)
    baseline_ap, baseline_inference_time = evaluate_model(baseline_model, processor, DEVICE)
    save_metrics_to_csv("DETR", "None", 0, baseline_ap, 0, baseline_inference_time, baseline_params, baseline_flops)

    pruning_methods = ["L1", "L2", "RANDOM"]
    pruning_ratios = [0.1, 0.3, 0.5, 0.7]

    for method in pruning_methods:
        for ratio in pruning_ratios:
            pruned_model, pruned_params, pruned_flops = prune_and_analyze(ratio, method, dummy_input)
            if pruned_model is None:
                save_metrics_to_csv("DETR", method, int(ratio * 100), 0.0, baseline_ap, 0.0, 0.0, 0.0)
                continue

            fine_tune_model(pruned_model, processor, DEVICE)
            ap, inference_time = evaluate_model(pruned_model, processor, DEVICE)
            ap_drop = baseline_ap - ap
            pruning_percentage = int(ratio * 100)
            save_metrics_to_csv("DETR", method, pruning_percentage, ap, ap_drop, inference_time, pruned_params, pruned_flops)
