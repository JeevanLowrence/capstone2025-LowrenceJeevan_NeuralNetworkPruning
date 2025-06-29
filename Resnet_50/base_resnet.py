import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import torchvision.transforms as transforms
from pathlib import Path
from PIL import Image
import time
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from ptflops import get_model_complexity_info
import scipy.io

# --- Config ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # Thesis/
IMAGENET_DIR = BASE_DIR / "Dataset" / "ImageNet"
VAL_IMG_DIR = IMAGENET_DIR / "ILSVRC2012_img_val"
VAL_GROUND_TRUTH_FILE = IMAGENET_DIR / "data" / "ILSVRC2012_validation_ground_truth.txt"
META_FILE = IMAGENET_DIR / "data" / "meta.mat"
SYNSET_WORDS_FILE = IMAGENET_DIR / "synset_words.txt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_id_to_synset():
    try:
        mat = scipy.io.loadmat(META_FILE)
        synsets = mat['synsets']
        id_to_synset = {}
        for synset in synsets:
            ilsvrc_id = int(synset[0][0].item())  # ILSVRC2012_ID
            wnid = str(synset[0][1][0])           # Synset ID (e.g., 'n01440764')
            if ilsvrc_id <= 1000:  # Filter low-level synsets
                id_to_synset[ilsvrc_id] = wnid
        print(f"Loaded {len(id_to_synset)} low-level synsets from {META_FILE}")
        print(f"Sample id_to_synset mappings: {dict(list(id_to_synset.items())[:5])}")
        return id_to_synset
    except FileNotFoundError:
        print(f"Error: {META_FILE} not found. Download ILSVRC2012_devkit_t12.tar.gz from http://www.image-net.org")
        raise
    except Exception as e:
        print(f"Error parsing {META_FILE}: {e}")
        raise

def get_synset_to_idx(id_to_synset):
    try:
        with open(SYNSET_WORDS_FILE, 'r') as f:
            standard_synsets = [line.split()[0] for line in f]
        if len(standard_synsets) != 1000:
            print(f"Warning: Expected 1000 synsets in {SYNSET_WORDS_FILE}, found {len(standard_synsets)}")
        synset_to_idx = {wnid: idx for idx, wnid in enumerate(standard_synsets)}
        print(f"Loaded {len(synset_to_idx)} synsets from {SYNSET_WORDS_FILE}")
        print(f"Sample standard synsets: {standard_synsets[:5]}")
        
        # Verify that all id_to_synset WNIDs are in standard_synsets
        missing_synsets = [wnid for wnid in id_to_synset.values() if wnid not in synset_to_idx]
        if missing_synsets:
            print(f"Warning: {len(missing_synsets)} synsets from meta.mat not found in synset_words.txt (first 5): {missing_synsets[:5]}")
        return synset_to_idx
    except FileNotFoundError:
        print(f"Warning: {SYNSET_WORDS_FILE} not found. Falling back to meta.mat order (may cause low accuracy).")
        # Fallback: Use sorted ILSVRC2012_IDs (less reliable)
        synset_to_idx = {}
        for ilsvrc_id, wnid in sorted(id_to_synset.items()):
            synset_to_idx[wnid] = ilsvrc_id - 1
        print(f"Sample synset_to_idx (fallback): {dict(list(synset_to_idx.items())[:5])}")
        return synset_to_idx

# --- Custom Dataset for ImageNet ---
class ImageNetValDataset(Dataset):
    def __init__(self, img_dir, ground_truth_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_names = [f"ILSVRC2012_val_{i:08d}" for i in range(1, 50001)]
        self.labels = []
        self.id_to_synset = get_id_to_synset()
        self.synset_to_idx = get_synset_to_idx(self.id_to_synset)

        # Load ground truth
        try:
            with open(ground_truth_file, 'r') as f:
                ilsvrc_ids = [int(line.strip()) for line in f]
            if len(ilsvrc_ids) != 50000:
                print(f"Warning: Expected 50,000 labels in {ground_truth_file}, found {len(ilsvrc_ids)}")
            
            # Map ILSVRC2012_ID to synset and then to index
            unmatched_ids = []
            for idx, ilsvrc_id in enumerate(ilsvrc_ids):
                synset = self.id_to_synset.get(ilsvrc_id)
                if synset and synset in self.synset_to_idx:
                    self.labels.append(self.synset_to_idx[synset])
                else:
                    print(f"Warning: ILSVRC2012_ID {ilsvrc_id} (image {self.img_names[idx]}) has no valid synset. Assigning label 0.")
                    self.labels.append(0)
                    if ilsvrc_id not in unmatched_ids:
                        unmatched_ids.append(ilsvrc_id)
            if unmatched_ids:
                print(f"Unmatched ILSVRC2012_IDs (first 10): {unmatched_ids[:10]}")
        except FileNotFoundError:
            print(f"Error: {ground_truth_file} not found.")
            raise

        print(f"Loaded {len(self.img_names)} images and {len(self.labels)} labels")
        print(f"Sample labels (first 5): {self.labels[:5]}")
        print(f"Sample synsets (first 5): {[self.id_to_synset[ilsvrc_ids[i]] for i in range(min(5, len(ilsvrc_ids)))]}")

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name + '.JPEG')
        if not os.path.exists(img_path):
            img_path = os.path.join(self.img_dir, img_name + '.jpeg')
        if not os.path.exists(img_path):
            print(f"Error: Image not found at {img_path}")
            return None
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error opening image {img_path}: {e}")
            return None
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# --- Data Preprocessing ---
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Custom Collate Function ---
def custom_collate(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

# --- Function to Calculate Model Size ---
def get_model_size(model, save_path="resnet50.pth"):
    torch.save(model.state_dict(), save_path)
    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    os.remove(save_path)
    return size_mb

# --- Function to Calculate Parameters ---
def get_parameters(model):
    params = sum(p.numel() for p in model.parameters())
    return params / 1e6

# --- Function to Calculate FLOPs ---
def get_flops(model):
    flops, _ = get_model_complexity_info(model, (3, 224, 224), as_strings=False)
    return flops / 1e6

# --- Function to Evaluate Accuracy and Inference Time ---
def evaluate_model(model, data_loader):
    correct = 0
    total = 0
    inference_times = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if batch is None:
                continue
            images, labels = batch
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            start_time = time.time()
            outputs = model(images)
            end_time = time.time()

            inference_times.append((end_time - start_time) / images.size(0))
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}: Evaluated {total} images, {correct} correct so far")

    accuracy = 100 * correct / total if total > 0 else 0
    avg_inference_time = np.mean(inference_times) if inference_times else 0
    print(f"Final: Evaluated {total} images, {correct} correct")
    return accuracy, avg_inference_time

# --- Main Execution ---
if __name__ == '__main__':
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Debug: Verify image and ground truth existence
    sample_images = [f"ILSVRC2012_val_{i:08d}.JPEG" for i in [1, 2, 3, 48981, 37956]]
    for img in sample_images:
        img_path = os.path.join(VAL_IMG_DIR, img)
        print(f"Checking {img_path}: {'Exists' if os.path.exists(img_path) else 'Not found'}")
    print(f"Checking {VAL_GROUND_TRUTH_FILE}: {'Exists' if os.path.exists(VAL_GROUND_TRUTH_FILE) else 'Not found'}")
    print(f"Checking {META_FILE}: {'Exists' if os.path.exists(META_FILE) else 'Not found'}")

    # Debug: Check torchvision’s ImageNet class order
    weights = ResNet50_Weights.IMAGENET1K_V1
    print(f"torchvision ImageNet classes (first 5): {weights.meta['categories'][:5]}")

    # Load Validation Dataset
    val_dataset = ImageNetValDataset(VAL_IMG_DIR, VAL_GROUND_TRUTH_FILE, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True, collate_fn=custom_collate)

    # Load Pre-trained ResNet50
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model = model.to(DEVICE)
    model.eval()

    # Run Evaluation
    try:
        accuracy, inference_time_s = evaluate_model(model, val_loader)
        size_mb = get_model_size(model)
        params_m = get_parameters(model)
        flops_m = get_flops(model)

        print(f"ResNet50 Baseline Evaluation on ImageNet:")
        print(f"Top-1 Accuracy: {accuracy:.2f}%")
        print(f"Model Size: {size_mb:.2f} MB")
        print(f"Average Inference Time per Image: {inference_time_s:.6f} seconds")
        print(f"FLOPs: {flops_m:.2f} M")
        print(f"Parameters: {params_m:.2f} M")
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()