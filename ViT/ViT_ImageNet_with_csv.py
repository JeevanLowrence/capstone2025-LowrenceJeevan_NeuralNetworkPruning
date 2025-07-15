import torch
import torch.nn as nn
import timm
from torchvision import transforms
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
from torch.amp import autocast, GradScaler
import pandas as pd

# Set up paths and configuration for the script
BASE_DIR = Path(__file__).resolve().parent.parent.parent
IMAGENET_DIR = BASE_DIR / "DATASET" / "ImageNet"
VAL_IMG_DIR = IMAGENET_DIR / "ILSVRC2012_img_val"
TRAIN_IMG_DIR = IMAGENET_DIR / "ILSVRC2012_img_train"
VAL_GROUND_TRUTH_FILE = IMAGENET_DIR / "data" / "ILSVRC2012_validation_ground_truth.txt"
SYNSET_WORDS_FILE = IMAGENET_DIR / "data" / "synset_words.txt"
MAPPING_FILE = IMAGENET_DIR / "data" / "meta.mat"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_SUBSET_SIZE = 10000  
FINE_TUNE_EPOCHS = 10  
BATCH_SIZE = 32  
PRUNING_RATIOS = [0.45, 0.50] 
NUM_WORKERS = 8  
VAL_SUBSET_SIZE = 10000  
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
CSV_FILE = BASE_DIR / "vit.csv"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Make sure dataset files exist
assert MAPPING_FILE.exists(), f"Missing {MAPPING_FILE}"
assert SYNSET_WORDS_FILE.exists(), f"Missing {SYNSET_WORDS_FILE}"
assert VAL_GROUND_TRUTH_FILE.exists(), f"Missing {VAL_GROUND_TRUTH_FILE}"
assert os.path.exists(TRAIN_IMG_DIR), f"Missing {TRAIN_IMG_DIR}"
assert os.path.exists(VAL_IMG_DIR), f"Missing {VAL_IMG_DIR}"

# Load ImageNet synset mappings from meta.mat
def get_id_to_synset():
    mat = scipy.io.loadmat(str(MAPPING_FILE))
    synsets = mat.get('synsets', [])
    return {int(synset[0][0].item()): str(synset[0][1][0]) for synset in synsets if int(synset[0][0].item()) <= 1000}

# Load synset-to-index mappings from synset_words.txt
def get_synset_to_idx():
    with open(SYNSET_WORDS_FILE, 'r') as f:
        standard_synsets = [line.split()[0] for line in f]
    return {wnid: idx for idx, wnid in enumerate(standard_synsets)}

# Custom dataset class for ImageNet with caching to speed up loading
class ImageNetDataset(Dataset):
    def __init__(self, img_dir, ground_truth_file=None, transform=None, is_train=False):
        self.img_dir = img_dir
        self.transform = transform
        self.is_train = is_train
        self.id_to_synset = get_id_to_synset()
        self.synset_to_idx = get_synset_to_idx()
        self.cache = {}  # Cache images to reduce disk I/O

        if is_train:
            synset_dirs = [d for d in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, d))]
            images_per_class = max(1, TRAIN_SUBSET_SIZE // len(synset_dirs))
            all_images = []
            all_labels = []
            for synset_dir in synset_dirs[:1000]:
                synset_path = os.path.join(img_dir, synset_dir)
                images = [f for f in os.listdir(synset_path) if f.lower().endswith(('.jpeg', '.jpg'))]
                images = images[:images_per_class]
                all_images.extend([os.path.join(synset_dir, img) for img in images])
                all_labels.extend([self.synset_to_idx.get(synset_dir, 0) for _ in images])
            self.img_names = all_images[:TRAIN_SUBSET_SIZE]
            self.labels = all_labels[:TRAIN_SUBSET_SIZE]
        else:
            with open(ground_truth_file, 'r') as f:
                ilsvrc_ids = [int(line.strip()) for line in f][:VAL_SUBSET_SIZE]
            self.img_names = []
            self.labels = []
            for idx, ilsvrc_id in enumerate(ilsvrc_ids):
                img_name = f"ILSVRC2012_val_{idx+1:08d}"
                img_path = os.path.join(img_dir, f"{img_name}.JPEG")
                if not os.path.exists(img_path):
                    img_path = img_path.replace('.JPEG', '.jpeg')
                if os.path.exists(img_path):
                    self.img_names.append(img_name)
                    synset = self.id_to_synset.get(ilsvrc_id)
                    if synset in self.synset_to_idx:
                        self.labels.append(self.synset_to_idx[synset])

        print(f"Loaded {len(self.img_names)} {'train' if is_train else 'val'} images")

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        if idx in self.cache:
            image = self.cache[idx]
        else:
            img_path = os.path.join(self.img_dir, img_name + ('.JPEG' if not self.is_train else ''))
            if not os.path.exists(img_path):
                img_path = img_path.replace('.JPEG', '.jpeg')
            image = Image.open(img_path).convert('RGB')
            if len(self.cache) < 1000:  # Limit cache size
                self.cache[idx] = image
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Set up image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Collate function to skip bad or corrupted images images
def custom_collate(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

# Calculate model size in MB
def get_model_size(model, save_path="temp.pth"):
    torch.save(model.state_dict(), save_path)
    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    os.remove(save_path)
    return size_mb

# Count model parameters in millions
def get_parameters(model):
    return sum(p.numel() for p in model.parameters()) / 1e6

# Calculate FLOPs in millions
def get_flops(model):
    flops, _ = get_model_complexity_info(model, (3, 224, 224), as_strings=False)
    return flops / 1e6

# Evaluate model accuracy and inference time with mixed precision
def evaluate_model(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    inference_times = []
    with torch.no_grad(), autocast('cuda'):
        for batch in data_loader:
            if batch is None:
                continue
            images, labels = batch
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            start_time = time.perf_counter()
            outputs = model(images)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            inference_times.append((time.perf_counter() - start_time) / images.size(0))
            _, predicted = torch.max(outputs.logits if hasattr(outputs, 'logits') else outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total if total > 0 else 0, np.mean(inference_times) if inference_times else 0

# Fine-tune model with early stopping and mixed precision
def fine_tune_model(model, train_loader, val_loader, epochs, checkpoint_path):
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler('cuda')
    best_acc = 0
    patience = 2
    trigger_times = 0
    train_time = 0

    for epoch in range(epochs):
        start_time = time.perf_counter()
        running_loss = 0.0
        epoch_batches = 0
        for batch in train_loader:
            if batch is None:
                continue
            images, labels = batch
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            with autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs.logits if hasattr(outputs, 'logits') else outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            epoch_batches += 1
        scheduler.step()
        train_time += time.perf_counter() - start_time
        val_acc, _ = evaluate_model(model, val_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / (epoch_batches or 1):.4f}, Val Acc: {val_acc:.2f}%")
        if val_acc > best_acc:
            best_acc = val_acc
            trigger_times = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    return train_time / (epoch + 1 or 1)

# Apply pruning to the model (focus on MLP layers, skip attention)
def apply_pruning(model, method, pruning_ratio):
    print(f"Applying {method} pruning with {pruning_ratio*100}% pruning ratio...")
    model = model.to(DEVICE)  # Keep on GPU for gradient computation
    if method == "L1":
        importance = tp.importance.MagnitudeImportance(p=1)
    elif method == "L2":
        importance = tp.importance.MagnitudeImportance(p=2)
    elif method == "Random":
        importance = tp.importance.RandomImportance()
    elif method == "Taylor":
        importance = tp.importance.TaylorImportance()
    else:
        return model

    # Only prune MLP layers, skip attention and head-related layers
    ignored_layers = [m for name, m in model.named_modules() if isinstance(m, nn.Linear) and ('head' in name or 'attn' in name or 'mlp' not in name)]
    unwrapped_parameters = [(model.cls_token, 2), (model.pos_embed, 2)]
    example_inputs = torch.randn(1, 3, 224, 224).to(DEVICE)

    # Compute gradients for Taylor pruning
    if method == "Taylor":
        model.train()
        model.zero_grad()
        with autocast('cuda'):
            output = model(example_inputs)
            loss = output.sum()  # Dummy loss to compute gradients
        loss.backward()

    DG = tp.DependencyGraph()
    DG.build_dependency(model, example_inputs=example_inputs, unwrapped_parameters=unwrapped_parameters)
    pruner = tp.pruner.MetaPruner(model, example_inputs, importance=importance, pruning_ratio=pruning_ratio, ignored_layers=ignored_layers, global_pruning=False)
    pruner.step()

    # Validate the pruned model
    with torch.no_grad():
        model.eval()
        model(example_inputs)
    return model

# Save results to CSV and plot accuracy vs. pruning percentage
def save_and_plot_results(results, pruning_ratio, epochs, baseline_results):
    output_dir = f"ViT-Base {int(pruning_ratio*100)}% pruned and fine tuned for {epochs} epochs"
    os.makedirs(output_dir, exist_ok=True)

    # Save to CSV
    csv_data = [{
        'pruning_percentage': pruning_ratio * 100,
        'architecture': 'ViT-B/16',
        'pruning_algorithm': r['method'],
        'accuracy_drop': r['accuracy_drop'],
        'parameter_drop_M': r['params_reduction_M'],
        'size_drop_MB': r['size_reduction_MB'],
        'avg_fine_tune_time_s': r['train_time_s']
    } for r in results if r['method'] != 'Baseline']
    df_new = pd.DataFrame(csv_data)
    if CSV_FILE.exists():
        df_existing = pd.read_csv(CSV_FILE)
        df = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(CSV_FILE, index=False)
    print(f"Saved results to {CSV_FILE} for {pruning_ratio*100}% pruning")

    # Plot metrics for current pruning ratio
    methods = [r['method'] for r in results]
    for metric, title, ylabel, color in [
        ([r['accuracy'] for r in results], 'Model Accuracy', 'Accuracy (%)', 'skyblue'),
        ([r['accuracy_drop'] for r in results], 'Accuracy Drop', 'Accuracy Drop (%)', 'salmon'),
        ([r['size_reduction_MB'] for r in results], 'Model Size Reduction', 'Size Reduction (MB)', 'lightgreen'),
        ([r['params_M'] for r in results], 'Number of Parameters', 'Parameters (M)', 'lightcoral'),
        ([r['train_time_s'] for r in results], 'Training Time per Epoch', 'Training Time (s)', 'plum')
    ]:
        plt.figure(figsize=(8, 5))
        bars = plt.bar(methods, metric, color=color)
        plt.title(f"{title} ({int(pruning_ratio*100)}% Pruning)")
        plt.xlabel('Pruning Method')
        plt.ylabel(ylabel)
        plt.ylim(0, max(metric) * 1.2 if metric else 1)
        for bar, val in zip(bars, metric):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(metric)*0.02, f'{val:.2f}', ha='center')
        plt.savefig(os.path.join(output_dir, f'{title.lower().replace(" ", "_")}_plot.png'))
        plt.close()

    # Plot accuracy vs. pruning percentage (all runs)
    if CSV_FILE.exists():
        df_all = pd.read_csv(CSV_FILE)
        plt.figure(figsize=(10, 6))
        for method in ['L1', 'L2', 'Random', 'Taylor']:
            method_data = df_all[df_all['pruning_algorithm'] == method]
            if not method_data.empty:
                plt.plot(method_data['pruning_percentage'], 100 - method_data['accuracy_drop'], label=method, marker='o')
        plt.title('Accuracy vs. Pruning Percentage for ViT-B/16')
        plt.xlabel('Pruning Percentage (%)')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'accuracy_vs_pruning.png'))
        plt.close()
    print(f"Saved plots in {output_dir}/")

# Main execution
if __name__ == '__main__':
    torch.cuda.empty_cache()

    # Load datasets
    print("Loading datasets...")
    train_dataset = ImageNetDataset(TRAIN_IMG_DIR, transform=transform, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=custom_collate)
    val_dataset = ImageNetDataset(VAL_IMG_DIR, VAL_GROUND_TRUTH_FILE, transform=transform, is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=custom_collate)

    # Evaluate baseline model (run once)
    print("Evaluating baseline model...")
    model = timm.create_model("vit_base_patch16_224", pretrained=True).to(DEVICE)
    baseline_acc, baseline_infer_time = evaluate_model(model, val_loader)
    baseline_size = get_model_size(model)
    baseline_params = get_parameters(model)
    baseline_flops = get_flops(model)
    baseline_results = {
        'method': 'Baseline',
        'accuracy': baseline_acc,
        'accuracy_drop': 0.0,
        'size_MB': baseline_size,
        'size_reduction_MB': 0.0,
        'train_time_s': 0.0,
        'inference_time_s': baseline_infer_time,
        'flops_M': baseline_flops,
        'flops_reduction_M': 0.0,
        'params_M': baseline_params,
        'params_reduction_M': 0.0
    }

    # Run pruning experiments for each pruning ratio
    all_results = [baseline_results]  # Store baseline once
    for pruning_ratio in PRUNING_RATIOS:
        print(f"\n=== Starting experiments for {pruning_ratio*100}% pruning ===")
        results = []
        pruning_methods = ['L1', 'L2', 'Random', 'Taylor']
        for method in pruning_methods:
            print(f"\nRunning {method} pruning...")
            model = timm.create_model("vit_base_patch16_224", pretrained=True)
            checkpoint_path = CHECKPOINT_DIR / f"{method}_pruned_{int(pruning_ratio*100)}%.pth"
            pruned_model = apply_pruning(model, method, pruning_ratio)
            print(f"Fine-tuning {method} model...")
            train_time = fine_tune_model(pruned_model, train_loader, val_loader, FINE_TUNE_EPOCHS, checkpoint_path)
            print(f"Evaluating {method} model...")
            acc, infer_time = evaluate_model(pruned_model, val_loader)
            size = get_model_size(pruned_model)
            params = get_parameters(pruned_model)
            flops = get_flops(pruned_model)
            results.append({
                'method': method,
                'accuracy': acc,
                'accuracy_drop': baseline_acc - acc,
                'size_MB': size,
                'size_reduction_MB': baseline_size - size,
                'train_time_s': train_time,
                'inference_time_s': infer_time,
                'flops_M': flops,
                'flops_reduction_M': baseline_flops - flops,
                'params_M': params,
                'params_reduction_M': baseline_params - params
            })

        # Add baseline to results for plotting
        results.insert(0, baseline_results)
        print(f"\nPruning Results for {pruning_ratio*100}%:")
        header = f"{'Method':<10} | {'Acc (%)':<8} | {'Drop (%)':<9} | {'Size (MB)':<10} | {'Params (M)':<11} | {'FLOPs (M)':<10} | {'Train Time':<12} | {'Infer Time':<12}"
        print(header)
        print('-' * len(header))
        for r in results:
            print(f"{r['method']:<10} | {r['accuracy']:<8.2f} | {r['accuracy_drop']:<9.2f} | {r['size_MB']:<10.2f} | "
                  f"{r['params_M']:<11.2f} | {r['flops_M']:<10.2f} | {r['train_time_s']:<12.2f} | {r['inference_time_s']:<12.6f}")
        save_and_plot_results(results, pruning_ratio, FINE_TUNE_EPOCHS, baseline_results)
        all_results.extend([r for r in results if r['method'] != 'Baseline'])

    # Final plot for all pruning percentages
    if CSV_FILE.exists():
        df_all = pd.read_csv(CSV_FILE)
        output_dir = BASE_DIR / "final_plots"
        os.makedirs(output_dir, exist_ok=True)
        plt.figure(figsize=(10, 6))
        for method in ['L1', 'L2', 'Random', 'Taylor']:
            method_data = df_all[df_all['pruning_algorithm'] == method]
            if not method_data.empty:
                plt.plot(method_data['pruning_percentage'], 100 - method_data['accuracy_drop'], label=method, marker='o')
        plt.title('Accuracy vs. Pruning Percentage for ViT-B/16 (All Runs)')
        plt.xlabel('Pruning Percentage (%)')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'final_accuracy_vs_pruning.png'))
        plt.close()
        print(f"Saved final plot in {output_dir}/")