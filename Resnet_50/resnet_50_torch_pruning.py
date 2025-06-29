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
import torch_pruning as tp
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

# Setting up paths and config
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # Getting to Thesis/ directory
IMAGENET_DIR = BASE_DIR / "Dataset" / "ImageNet"
VAL_IMG_DIR = IMAGENET_DIR / "ILSVRC2012_img_val"
TRAIN_IMG_DIR = IMAGENET_DIR / "ILSVRC2012_img_train"
VAL_GROUND_TRUTH_FILE = IMAGENET_DIR / "data" / "ILSVRC2012_validation_ground_truth.txt"
SYNSET_WORDS_FILE = IMAGENET_DIR / "synset_words.txt"
MAPPING_FILE = IMAGENET_DIR / "data" / "meta.mat"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_SUBSET_SIZE = 10000  # Using a subset of 10,000 images for fine-tuning
FINE_TUNE_EPOCHS = 10  # Fine-tuning for 10 epochs to recover accuracy
BATCH_SIZE = 64  # Setting batch size for stable training
PRUNING_RATIO = 0.1  # Removing 10% of parameters during pruning

# Loading ILSVRC2012 ID to synset mappings from meta.mat
def get_id_to_synset():
    mat = scipy.io.loadmat(MAPPING_FILE)
    synsets = mat.get('synsets', [])
    id_to_synset = {}
    for synset in synsets:
        ilsvrc_id = int(synset[0][0].item())
        wnid = str(synset[0][1][0])
        if ilsvrc_id <= 1000:  # Only including low-level synsets (1–1000)
            id_to_synset[ilsvrc_id] = wnid
    print(f"Loaded {len(id_to_synset)} synsets from meta.mat")
    return id_to_synset

# Loading synset to index mappings from synset_words.txt
def get_synset_to_idx():
    with open(SYNSET_WORDS_FILE, 'r') as f:
        standard_synsets = [line.split()[0] for line in f]
    synset_to_idx = {wnid: idx for idx, wnid in enumerate(standard_synsets)}
    print(f"Loaded {len(synset_to_idx)} synsets from synset_words.txt")
    return synset_to_idx

# Creating a custom dataset for ImageNet
class ImageNetDataset(Dataset):
    def __init__(self, img_dir, ground_truth_file=None, transform=None, is_train=False):
        self.img_dir = img_dir
        self.transform = transform
        self.is_train = is_train
        self.id_to_synset = get_id_to_synset()
        self.synset_to_idx = get_synset_to_idx()
        self.missing_images = []

        if is_train:
            # Loading training images from subdirectories
            all_images = []
            all_labels = []
            synset_dirs = [d for d in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, d))]
            images_per_class = max(1, TRAIN_SUBSET_SIZE // len(synset_dirs))
            for synset_dir in synset_dirs[:1000]:  # Covering all 1000 classes
                synset_path = os.path.join(img_dir, synset_dir)
                images = [f for f in os.listdir(synset_path) if f.lower().endswith(('.jpeg', '.jpg'))]
                images = images[:images_per_class]
                all_images.extend([os.path.join(synset_dir, img) for img in images])
                all_labels.extend([self.synset_to_idx.get(synset_dir, 0) for _ in images])
            self.img_names = all_images[:TRAIN_SUBSET_SIZE]
            self.labels = all_labels[:TRAIN_SUBSET_SIZE]
            print(f"Training dataset: {len(self.img_names)} images from {len(set([img.split('/')[0] for img in self.img_names]))} classes")
        else:
            # Loading validation images and labels
            self.img_names = []
            self.labels = []
            with open(ground_truth_file, 'r') as f:
                ilsvrc_ids = [int(line.strip()) for line in f]
            for idx, ilsvrc_id in enumerate(ilsvrc_ids):
                img_name = f"ILSVRC2012_val_{idx+1:08d}"
                img_path = os.path.join(img_dir, f"{img_name}.JPEG")
                if not os.path.exists(img_path):
                    img_path = img_path.replace('.JPEG', '.jpeg')
                if not os.path.exists(img_path):
                    self.missing_images.append(img_name)
                    continue
                self.img_names.append(img_name)
                synset = self.id_to_synset.get(ilsvrc_id)
                self.labels.append(self.synset_to_idx.get(synset, 0))
            if self.missing_images:
                print(f"Missing {len(self.missing_images)} validation images")

        print(f"Loaded {len(self.img_names)} images and {len(self.labels)} labels")

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name + ('.JPEG' if not self.is_train else ''))
        if not os.path.exists(img_path):
            img_path = img_path.replace('.JPEG', '.jpeg').replace('.jpg', '.JPEG')
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Setting up image preprocessing
transform = transforms.Compose([
    transforms.Resize(256),  # Resizing images to 256x256
    transforms.CenterCrop(224),  # Cropping to 224x224
    transforms.ToTensor(),  # Converting to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizing with ImageNet stats
])

# Filtering out None items from batches
def custom_collate(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        print("Got an empty batch, skipping")
        return None
    return torch.utils.data.dataloader.default_collate(batch)

# Calculating model size in MB
def get_model_size(model, save_path="temp.pth"):
    torch.save(model.state_dict(), save_path)
    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    os.remove(save_path)
    return size_mb

# Counting model parameters in millions
def get_parameters(model):
    params = sum(p.numel() for p in model.parameters())
    return params / 1e6

# Calculating FLOPs in millions
def get_flops(model):
    flops, _ = get_model_complexity_info(model, (3, 224, 224), as_strings=False)
    return flops / 1e6

# Evaluating model accuracy and inference time
def evaluate_model(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    inference_times = []
    processed_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if batch is None:
                print(f"Batch {batch_idx} skipped: Empty batch")
                continue
            images, labels = batch
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()  # Syncing GPU for accurate timing
            start_time = time.perf_counter()
            outputs = model(images)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            end_time = time.perf_counter()

            batch_time = end_time - start_time
            per_image_time = batch_time / images.size(0)
            inference_times.append(per_image_time)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            processed_batches += 1
            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}: Evaluated {total} images, {correct} correct")

        accuracy = 100 * correct / total if total > 0 else 0
        avg_inference_time = np.mean(inference_times) if inference_times else 0
        print(f"Evaluated {total} images, {correct} correct, {processed_batches} batches")
        return accuracy, avg_inference_time

# Fine-tuning the model
def fine_tune_model(model, train_loader, epochs=10):
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)  # Using AdamW optimizer
    scheduler = StepLR(optimizer, step_size=3, gamma=0.1)  # Reducing LR every 3 epochs
    criterion = nn.CrossEntropyLoss()
    train_time = 0
    processed_images = 0
    processed_batches = 0

    for epoch in range(epochs):
        start_time = time.perf_counter()
        running_loss = 0.0
        epoch_images = 0
        epoch_batches = 0
        for batch_idx, batch in enumerate(train_loader):
            if batch is None:
                print(f"Epoch {epoch+1}, Batch {batch_idx} skipped: Empty batch")
                continue
            images, labels = batch
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            epoch_images += images.size(0)
            processed_images += images.size(0)
            epoch_batches += 1
            processed_batches += 1
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {running_loss / (epoch_batches or 1):.4f}, "
                      f"LR: {scheduler.get_last_lr()[0]:.6f}, Images: {epoch_images}")

        scheduler.step()
        epoch_time = time.perf_counter() - start_time
        train_time += epoch_time
        print(f"Epoch {epoch+1} done: {epoch_images} images, {epoch_batches} batches, Loss: {running_loss / (epoch_batches or 1):.4f}")

    avg_train_time = train_time / epochs if epochs > 0 else 0
    print(f"Average training time per epoch: {avg_train_time:.2f}s")
    return avg_train_time

# Applying pruning to the model
def apply_pruning(model, method, pruning_ratio=0.5):
    print(f"\nApplying {method} pruning with {pruning_ratio*100}% reduction...")
    model = model.to('cpu')  # Moving to CPU for pruning
    if method == "L1":
        importance = tp.importance.MagnitudeImportance(p=1)  # Using L1 norm
    elif method == "L2":
        importance = tp.importance.MagnitudeImportance(p=2)  # Using L2 norm
    elif method == "Random":
        importance = tp.importance.RandomImportance()  # Random pruning
    elif method == "Taylor":
        importance = tp.importance.TaylorImportance()  # Taylor-based pruning
    else:
        print(f"Unknown pruning method: {method}")
        return model

    ignored_layers = []
    for m in model.modules():
        if isinstance(m, nn.Linear) and m.out_features == 1000:
            ignored_layers.append(m)  # Skipping final layer

    example_inputs = torch.randn(1, 3, 224, 224).to('cpu')
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs=example_inputs,
        importance=importance,
        iterative_steps=1,
        ch_sparsity=pruning_ratio,
        ignored_layers=ignored_layers
    )

    if method == "Taylor":
        model.eval()
        example_inputs.requires_grad_(True)
        outputs = model(example_inputs)
        loss = outputs.sum()
        loss.backward()

    pruner.step()
    print(f"{method} pruning done")
    return model.to(DEVICE)

# Plotting the results
def plot_metrics(results, pruning_ratio=PRUNING_RATIO, epochs=FINE_TUNE_EPOCHS):
    output_dir = f"Resnet-50 {int(pruning_ratio*100)}% pruned and fine tuned for {epochs} epochs"  # Naming folder with pruning percentage and epochs
    os.makedirs(output_dir, exist_ok=True)  # Creating plots directory
    methods = [r["method"] for r in results]
    
    # Plotting accuracy
    accuracies = [r["accuracy"] for r in results]
    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, accuracies, color='skyblue')
    plt.title("Model Accuracy by Pruning Method")
    plt.xlabel("Pruning Method")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{acc:.2f}", ha='center')
    plt.savefig(os.path.join(output_dir, "accuracy_plot.png"))
    plt.close()

    # Plotting accuracy drop
    accuracy_drops = [r["accuracy_drop"] for r in results]
    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, accuracy_drops, color='salmon')
    plt.title("Accuracy Drop by Pruning Method")
    plt.xlabel("Pruning Method")
    plt.ylabel("Accuracy Drop (%)")
    plt.ylim(0, max(accuracy_drops) * 1.2)
    for bar, drop in zip(bars, accuracy_drops):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f"{drop:.2f}", ha='center')
    plt.savefig(os.path.join(output_dir, "accuracy_drop_plot.png"))
    plt.close()

    # Plotting size reduction
    size_reductions = [r["size_reduction_MB"] for r in results]
    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, size_reductions, color='lightgreen')
    plt.title("Model Size Reduction by Pruning Method")
    plt.xlabel("Pruning Method")
    plt.ylabel("Size Reduction (MB)")
    plt.ylim(0, max(size_reductions) * 1.2)
    for bar, size in zip(bars, size_reductions):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{size:.2f}", ha='center')
    plt.savefig(os.path.join(output_dir, "size_reduction_plot.png"))
    plt.close()

    # Plotting parameters
    params = [r["params_M"] for r in results]
    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, params, color='lightcoral')
    plt.title("Number of Parameters by Pruning Method")
    plt.xlabel("Pruning Method")
    plt.ylabel("Parameters (M)")
    plt.ylim(0, max(params) * 1.2)
    for bar, param in zip(bars, params):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f"{param:.2f}", ha='center')
    plt.savefig(os.path.join(output_dir, "parameters_plot.png"))
    plt.close()

    # Plotting training time
    train_times = [r["train_time_s"] for r in results]
    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, train_times, color='plum')
    plt.title("Training Time per Epoch by Pruning Method")
    plt.xlabel("Pruning Method")
    plt.ylabel("Training Time (s)")
    plt.ylim(0, max(train_times) * 1.2)
    for bar, time in zip(bars, train_times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f"{time:.2f}", ha='center')
    plt.savefig(os.path.join(output_dir, "train_time_plot.png"))
    plt.close()
    print(f"Saved plots in {output_dir}/")

# Main execution
if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Clearing GPU memory

    # Loading datasets
    print("\nLoading validation dataset...")
    val_dataset = ImageNetDataset(VAL_IMG_DIR, VAL_GROUND_TRUTH_FILE, transform=transform, is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True, collate_fn=custom_collate)

    print("\nLoading training dataset...")
    train_dataset = ImageNetDataset(TRAIN_IMG_DIR, transform=transform, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, collate_fn=custom_collate)

    # Evaluating baseline ResNet50
    print("\nEvaluating baseline model...")
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model = model.to(DEVICE)
    model.eval()

    baseline_acc, baseline_infer_time = evaluate_model(model, val_loader)
    baseline_size = get_model_size(model)
    baseline_params = get_parameters(model)
    baseline_flops = get_flops(model)

    baseline_results = {
        "method": "Baseline",
        "accuracy": baseline_acc,
        "accuracy_drop": 0.0,
        "size_MB": baseline_size,
        "size_reduction_MB": 0.0,
        "train_time_s": 0.0,
        "inference_time_s": baseline_infer_time,
        "flops_M": baseline_flops,
        "flops_reduction_M": 0.0,
        "params_M": baseline_params,
        "params_reduction_M": 0.0
    }

    # Running pruning methods
    pruning_methods = ["L1", "L2", "Random", "Taylor"]
    results = []

    for method in pruning_methods:
        print(f"\n=== Running {method} Pruning ===")
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        model = model.to(DEVICE)

        model = apply_pruning(model, method, PRUNING_RATIO)

        print(f"Fine-tuning {method} model...")
        train_time = fine_tune_model(model, train_loader, epochs=FINE_TUNE_EPOCHS)

        print(f"Evaluating {method} model...")
        acc, infer_time = evaluate_model(model, val_loader)

        size = get_model_size(model)
        params = get_parameters(model)
        flops = get_flops(model)

        results.append({
            "method": method,
            "accuracy": acc,
            "accuracy_drop": baseline_acc - acc,
            "size_MB": size,
            "size_reduction_MB": baseline_size - size,
            "train_time_s": train_time,
            "inference_time_s": infer_time,
            "flops_M": flops,
            "flops_reduction_M": baseline_flops - flops,
            "params_M": params,
            "params_reduction_M": baseline_params - params
        })

    # Showing results table
    results.insert(0, baseline_results)
    print("\n=== Pruning Results ===")
    header = f"{'Method':<10} | {'Acc (%)':<8} | {'Drop (%)':<9} | {'Size (MB)':<10} | {'Params (M)':<11} | {'FLOPs (M)':<10} | {'Train Time':<12} | {'Infer Time':<12}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['method']:<10} | {r['accuracy']:<8.2f} | {r['accuracy_drop']:<9.2f} | {r['size_MB']:<10.2f} | "
              f"{r['params_M']:<11.2f} | {r['flops_M']:<10.2f} | {r['train_time_s']:<12.2f} | {r['inference_time_s']:<12.6f}")

    # Generating plots
    print("\nCreating plots...")
    plot_metrics(results, pruning_ratio=PRUNING_RATIO, epochs=FINE_TUNE_EPOCHS)