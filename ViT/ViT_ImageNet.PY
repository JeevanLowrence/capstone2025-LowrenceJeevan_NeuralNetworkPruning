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

# Setting up paths and config
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # Thesis/
IMAGENET_DIR = BASE_DIR / "Dataset" / "ImageNet"
VAL_IMG_DIR = IMAGENET_DIR / "ILSVRC2012_img_val"
TRAIN_IMG_DIR = IMAGENET_DIR / "ILSVRC2012_img_train"
VAL_GROUND_TRUTH_FILE = IMAGENET_DIR / "data" / "ILSVRC2012_validation_ground_truth.txt"
SYNSET_WORDS_FILE = IMAGENET_DIR / "data" / "synset_words.txt"
MAPPING_FILE = IMAGENET_DIR / "data" / "meta.mat"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_SUBSET_SIZE = 10000
FINE_TUNE_EPOCHS = 10
BATCH_SIZE = 64
PRUNING_RATIO = 0.05

# Verify paths
print(f"BASE_DIR: {BASE_DIR}")
print(f"IMAGENET_DIR: {IMAGENET_DIR}")
print(f"MAPPING_FILE: {MAPPING_FILE}")
print(f"SYNSET_WORDS_FILE: {SYNSET_WORDS_FILE}")
print(f"VAL_GROUND_TRUTH_FILE: {VAL_GROUND_TRUTH_FILE}")

# Loading ILSVRC2012 ID to synset mappings
def get_id_to_synset():
    if not MAPPING_FILE.exists():
        print(f"Error: {MAPPING_FILE} does not exist")
        raise FileNotFoundError(f"{MAPPING_FILE} not found")
    try:
        print(f"Loading meta.mat from {MAPPING_FILE}")
        mat = scipy.io.loadmat(str(MAPPING_FILE))
        synsets = mat.get('synsets', [])
        id_to_synset = {}
        for synset in synsets:
            ilsvrc_id = int(synset[0][0].item())
            wnid = str(synset[0][1][0])
            if ilsvrc_id <= 1000:
                id_to_synset[ilsvrc_id] = wnid
        print(f"Loaded {len(id_to_synset)} synsets from meta.mat")
        return id_to_synset
    except Exception as e:
        print(f"Error loading meta.mat: {e}")
        raise

# Loading synset to index mappings
def get_synset_to_idx():
    if not SYNSET_WORDS_FILE.exists():
        print(f"Error: {SYNSET_WORDS_FILE} does not exist")
        raise FileNotFoundError(f"{SYNSET_WORDS_FILE} not found")
    try:
        with open(SYNSET_WORDS_FILE, 'r') as f:
            standard_synsets = [line.split()[0] for line in f]
        synset_to_idx = {wnid: idx for idx, wnid in enumerate(standard_synsets)}
        print(f"Loaded {len(synset_to_idx)} synsets from synset_words.txt")
        return synset_to_idx
    except Exception as e:
        print(f"Error loading synset_words.txt: {e}")
        raise

# Custom ImageNet dataset
class ImageNetDataset(Dataset):
    def __init__(self, img_dir, ground_truth_file=None, transform=None, is_train=False):
        self.img_dir = img_dir
        self.transform = transform
        self.is_train = is_train
        self.id_to_synset = get_id_to_synset()
        self.synset_to_idx = get_synset_to_idx()
        self.missing_images = []

        if is_train:
            if not os.path.exists(img_dir):
                print(f"Error: Training directory {img_dir} does not exist")
                raise FileNotFoundError(f"{img_dir} not found")
            all_images = []
            all_labels = []
            synset_dirs = [d for d in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, d))]
            images_per_class = max(1, TRAIN_SUBSET_SIZE // len(synset_dirs))
            for synset_dir in synset_dirs[:1000]:
                synset_path = os.path.join(img_dir, synset_dir)
                images = [f for f in os.listdir(synset_path) if f.lower().endswith(('.jpeg', '.jpg'))]
                images = images[:images_per_class]
                all_images.extend([os.path.join(synset_dir, img) for img in images])
                all_labels.extend([self.synset_to_idx.get(synset_dir, 0) for _ in images])
            self.img_names = all_images[:TRAIN_SUBSET_SIZE]
            self.labels = all_labels[:TRAIN_SUBSET_SIZE]
            print(f"Training dataset: {len(self.img_names)} images from {len(set([img.split('/')[0] for img in self.img_names]))} classes")
        else:
            if not os.path.exists(img_dir):
                print(f"Error: Validation directory {img_dir} does not exist")
                raise FileNotFoundError(f"{img_dir} not found")
            if not ground_truth_file.exists():
                print(f"Error: {ground_truth_file} does not exist")
                raise FileNotFoundError(f"{ground_truth_file} not found")
            self.img_names = []
            self.labels = []
            try:
                with open(ground_truth_file, 'r') as f:
                    ilsvrc_ids = [int(line.strip()) for line in f]
            except Exception as e:
                print(f"Error reading ground truth file: {e}")
                raise
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
                if synset not in self.synset_to_idx:
                    print(f"Warning: Synset {synset} not found in synset_to_idx for image {img_name}")
                    continue
                self.labels.append(self.synset_to_idx[synset])
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
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Collate function to filter None items
def custom_collate(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        print("Got an empty batch, skipping")
        return None
    return torch.utils.data.dataloader.default_collate(batch)

# Calculate model size in MB
def get_model_size(model, save_path="temp.pth"):
    try:
        torch.save(model.state_dict(), save_path)
        size_mb = os.path.getsize(save_path) / (1024 * 1024)
        os.remove(save_path)
        return size_mb
    except Exception as e:
        print(f"Error calculating model size: {e}")
        return 0.0

# Count model parameters in millions
def get_parameters(model):
    try:
        params = sum(p.numel() for p in model.parameters())
        return params / 1e6
    except Exception as e:
        print(f"Error counting parameters: {e}")
        return 0.0

# Calculate FLOPs in millions
def get_flops(model):
    try:
        flops, _ = get_model_complexity_info(model, (3, 224, 224), as_strings=False)
        return flops / 1e6
    except Exception as e:
        print(f"Error calculating FLOPs: {e}")
        return 0.0

# Evaluate model accuracy and inference time
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
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            outputs = model(images)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            end_time = time.perf_counter()

            batch_time = end_time - start_time
            per_image_time = batch_time / images.size(0)
            inference_times.append(per_image_time)

            _, predicted = torch.max(outputs.logits if hasattr(outputs, 'logits') else outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            processed_batches += 1
            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}: Evaluated {total} images, {correct} correct")

        accuracy = 100 * correct / total if total > 0 else 0
        avg_inference_time = np.mean(inference_times) if inference_times else 0
        print(f"Evaluated {total} images, {correct} correct, {processed_batches} batches")
        return accuracy, avg_inference_time

# Fine-tune the model
def fine_tune_model(model, train_loader, epochs=10):
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
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
            loss = criterion(outputs.logits if hasattr(outputs, 'logits') else outputs, labels)
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

# Apply pruning to the model
def apply_pruning(model, method, pruning_ratio=0.05):
    print(f"\nApplying {method} pruning with {pruning_ratio*100}% reduction...")
    model = model.to('cpu')
    if method == "L1":
        importance = tp.importance.MagnitudeImportance(p=1)
    elif method == "L2":
        importance = tp.importance.MagnitudeImportance(p=2)
    elif method == "Random":
        importance = tp.importance.RandomImportance()
    elif method == "Taylor":
        importance = tp.importance.TaylorImportance()
        # For Taylor, consider head pruning
        num_heads = 12  # ViT-B/16 has 12 heads
        head_pruning_ratio = pruning_ratio  # Prune 5% of heads
        num_heads_to_prune = int(num_heads * head_pruning_ratio)
        if num_heads_to_prune > 0:
            print(f"Pruning {num_heads_to_prune} attention heads for Taylor method")
    else:
        print(f"Unknown pruning method: {method}")
        return model

    ignored_layers = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and ('head' in name or 'attn' in name):
            ignored_layers.append(m)  # Skip all attention and head layers
        elif isinstance(m, nn.Linear) and ('mlp' not in name):
            ignored_layers.append(m)  # Skip non-MLP linear layers

    unwrapped_parameters = [
        (model.cls_token, 2),  # Prune last dimension if needed
        (model.pos_embed, 2)
    ]

    example_inputs = torch.randn(1, 3, 224, 224).to('cpu')
    DG = tp.DependencyGraph()
    DG.build_dependency(model, example_inputs=example_inputs, unwrapped_parameters=unwrapped_parameters)

    # Debug: Print MLP layer shapes before pruning
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and 'mlp' in name:
            print(f"Before pruning - {name} weight shape: {module.weight.shape}")

    pruner = tp.pruner.MetaPruner(
        model,
        example_inputs=example_inputs,
        importance=importance,
        pruning_ratio=pruning_ratio,
        ignored_layers=ignored_layers,
        global_pruning=False
    )

    if method == "Taylor" and num_heads_to_prune > 0:
        for block in model.blocks:
            attn = block.attn
            head_dim = attn.head_dim  # 64 for ViT-B/16
            current_heads = attn.num_heads
            new_heads = current_heads - num_heads_to_prune
            if new_heads < 1:
                print(f"Warning: Cannot prune {num_heads_to_prune} heads, skipping head pruning")
                continue
            attn.num_heads = new_heads
            new_qkv_out_dim = new_heads * head_dim * 3
            # Adjust qkv layer
            old_qkv = attn.qkv
            new_qkv = nn.Linear(old_qkv.in_features, new_qkv_out_dim).to('cpu')
            with torch.no_grad():
                new_qkv.weight.copy_(old_qkv.weight[:new_qkv_out_dim])
                if old_qkv.bias is not None:
                    new_qkv.bias.copy_(old_qkv.bias[:new_qkv_out_dim])
            attn.qkv = new_qkv
            # Adjust projection layer
            old_proj = attn.proj
            new_proj = nn.Linear(new_heads * head_dim, old_proj.out_features).to('cpu')
            with torch.no_grad():
                new_proj.weight.copy_(old_proj.weight[:, :new_heads * head_dim])
                if old_proj.bias is not None:
                    new_proj.bias.copy_(old_proj.bias)
            attn.proj = new_proj

    pruner.step()

    # Debug: Print MLP layer shapes after pruning
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and 'mlp' in name:
            print(f"After pruning - {name} weight shape: {module.weight.shape}")

    # Validate model
    try:
        model.eval()
        with torch.no_grad():
            _ = model(example_inputs)
        print(f"{method} pruning done and model validated")
    except Exception as e:
        print(f"Error validating pruned model: {e}")
        return None

    return model.to(DEVICE)

# Plot results
def plot_metrics(results, pruning_ratio=PRUNING_RATIO, epochs=FINE_TUNE_EPOCHS):
    output_dir = f"ViT-Base {int(pruning_ratio*100)}% pruned and fine tuned for {epochs} epochs"
    os.makedirs(output_dir, exist_ok=True)
    methods = [r["method"] for r in results]

    # Plot accuracy
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

    # Plot accuracy drop
    accuracy_drops = [r["accuracy_drop"] for r in results]
    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, accuracy_drops, color='salmon')
    plt.title("Accuracy Drop by Pruning Method")
    plt.xlabel("Pruning Method")
    plt.ylabel("Accuracy Drop (%)")
    plt.ylim(0, max(accuracy_drops) * 1.2 if accuracy_drops else 1)
    for bar, drop in zip(bars, accuracy_drops):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f"{drop:.2f}", ha='center')
    plt.savefig(os.path.join(output_dir, "accuracy_drop_plot.png"))
    plt.close()

    # Plot size reduction
    size_reductions = [r["size_reduction_MB"] for r in results]
    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, size_reductions, color='lightgreen')
    plt.title("Model Size Reduction by Pruning Method")
    plt.xlabel("Pruning Method")
    plt.ylabel("Size Reduction (MB)")
    plt.ylim(0, max(size_reductions) * 1.2 if size_reductions else 1)
    for bar, size in zip(bars, size_reductions):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{size:.2f}", ha='center')
    plt.savefig(os.path.join(output_dir, "size_reduction_plot.png"))
    plt.close()

    # Plot parameters
    params = [r["params_M"] for r in results]
    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, params, color='lightcoral')
    plt.title("Number of Parameters by Pruning Method")
    plt.xlabel("Pruning Method")
    plt.ylabel("Parameters (M)")
    plt.ylim(0, max(params) * 1.2 if params else 1)
    for bar, param in zip(bars, params):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f"{param:.2f}", ha='center')
    plt.savefig(os.path.join(output_dir, "parameters_plot.png"))
    plt.close()

    # Plot training time
    train_times = [r["train_time_s"] for r in results]
    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, train_times, color='plum')
    plt.title("Training Time per Epoch by Pruning Method")
    plt.xlabel("Pruning Method")
    plt.ylabel("Training Time (s)")
    plt.ylim(0, max(train_times) * 1.2 if train_times else 1)
    for bar, time in zip(bars, train_times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f"{time:.2f}", ha='center')
    plt.savefig(os.path.join(output_dir, "train_time_plot.png"))
    plt.close()
    print(f"Saved plots in {output_dir}/")

# Main execution
if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nLoading validation dataset...")
    try:
        val_dataset = ImageNetDataset(VAL_IMG_DIR, VAL_GROUND_TRUTH_FILE, transform=transform, is_train=False)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True, collate_fn=custom_collate)
    except Exception as e:
        print(f"Error loading validation dataset: {e}")
        exit(1)

    print("\nLoading training dataset...")
    try:
        train_dataset = ImageNetDataset(TRAIN_IMG_DIR, transform=transform, is_train=True)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, collate_fn=custom_collate)
    except Exception as e:
        print(f"Error loading training dataset: {e}")
        exit(1)

    print("\nEvaluating baseline model...")
    try:
        model = timm.create_model("vit_base_patch16_224", pretrained=True)
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
    except Exception as e:
        print(f"Error evaluating baseline model: {e}")
        exit(1)

    pruning_methods = ["L1", "L2", "Random", "Taylor"]
    results = []

    for method in pruning_methods:
        print(f"\n=== Running {method} Pruning ===")
        try:
            model = timm.create_model("vit_base_patch16_224", pretrained=True)
            model = model.to(DEVICE)

            pruned_model = apply_pruning(model, method, PRUNING_RATIO)
            if pruned_model is None:
                print(f"Skipping {method} due to validation failure")
                continue

            print(f"Fine-tuning {method} model...")
            train_time = fine_tune_model(pruned_model, train_loader, epochs=FINE_TUNE_EPOCHS)

            print(f"Evaluating {method} model...")
            acc, infer_time = evaluate_model(pruned_model, val_loader)

            size = get_model_size(pruned_model)
            params = get_parameters(pruned_model)
            flops = get_flops(pruned_model)

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
        except Exception as e:
            print(f"Error during {method} pruning or evaluation: {e}")
            continue

    if results:
        results.insert(0, baseline_results)
        print("\n=== Pruning Results ===")
        header = f"{'Method':<10} | {'Acc (%)':<8} | {'Drop (%)':<9} | {'Size (MB)':<10} | {'Params (M)':<11} | {'FLOPs (M)':<10} | {'Train Time':<12} | {'Infer Time':<12}"
        print(header)
        print("-" * len(header))
        for r in results:
            print(f"{r['method']:<10} | {r['accuracy']:<8.2f} | {r['accuracy_drop']:<9.2f} | {r['size_MB']:<10.2f} | "
                  f"{r['params_M']:<11.2f} | {r['flops_M']:<10.2f} | {r['train_time_s']:<12.2f} | {r['inference_time_s']:<12.6f}")

    print("\nCreating plots...")
    try:
        plot_metrics(results, pruning_ratio=PRUNING_RATIO, epochs=FINE_TUNE_EPOCHS)
    except Exception as e:
        print(f"Error generating plots: {e}")
