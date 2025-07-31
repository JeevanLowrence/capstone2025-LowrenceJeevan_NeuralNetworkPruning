import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch_pruning as tp
from torch_pruning import importance
import time
import os
import csv

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(1600, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train(model, device, loader, optimizer, criterion, epoch=1):
    model.train()
    for _ in range(epoch):
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()

def evaluate(model, device, loader):
    model.eval()
    test_loss = 0
    correct = 0
    start = time.time()
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    end = time.time()
    inference_time = end - start
    test_loss /= len(loader.dataset)
    accuracy = 100. * correct / len(loader.dataset)
    return test_loss, accuracy, inference_time

def prune_model(model, example_input, method='random', prune_amount=0.2):
    if method == 'l1':
        imp = importance.MagnitudeImportance(p=1)
    elif method == 'l2':
        imp = importance.MagnitudeImportance(p=2)
    elif method == 'random':
        imp = importance.RandomImportance()
    elif method == 'taylor':
        model.train()
        dummy_input = example_input.clone().requires_grad_(True)
        dummy_output = model(dummy_input)
        dummy_loss = dummy_output.mean()
        dummy_loss.backward()
        imp = importance.TaylorImportance()
    else:
        raise ValueError(f"Unknown method: {method}")

    ignored_layers = [model.fc2]
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_input,
        importance=imp,
        pruning_ratio=prune_amount,
        global_pruning=True,
        root_module_types=[nn.Conv2d, nn.Linear],
        ignored_layers=ignored_layers
    )
    pruner.step()
    return model

def model_size(model):
    torch.save(model.state_dict(), "temp.pth")
    size = os.path.getsize("temp.pth") / 1e6  # MB
    os.remove("temp.pth")
    return size

def get_flops_and_params(model, example_input):
    total_ops, total_params = tp.utils.count_ops_and_params(model, example_input)
    return total_ops / 1e6, total_params / 1e6  # in Mega

# Pruning configuration
methods = ['random', 'l1', 'l2', 'taylor']
prune_percents = [0.1, 0.3, 0.5, 0.7]

# Baseline training
print("Training baseline model...")
baseline_model = CNN().to(device)
optimizer = optim.Adam(baseline_model.parameters(), lr=0.001)
criterion = nn.NLLLoss()

base_start_time = time.time()
train(baseline_model, device, train_loader, optimizer, criterion, epoch=3)
base_end_time = time.time()

baseline_train_time = base_end_time - base_start_time
_, baseline_acc, baseline_infer_time = evaluate(baseline_model, device, test_loader)
baseline_size = model_size(baseline_model)
baseline_flops, baseline_params = get_flops_and_params(baseline_model, torch.randn(1, 1, 28, 28).to(device))

# Store results
results = []

baseline_results = {
    "method": "Baseline",
    "prune_percent": 0,
    "accuracy": baseline_acc,
    "accuracy_drop": 0.0,
    "size_MB": baseline_size,
    "size_reduction_MB": 0.0,
    "train_time_s": baseline_train_time,
    "inference_time_s": baseline_infer_time,
    "flops_M": baseline_flops,
    "flops_reduction_M": 0.0,
    "params_M": baseline_params,
    "params_reduction_M": 0.0
}
results.append(baseline_results)

# Pruning and retraining loop
for method in methods:
    for pct in prune_percents:
        print(f"\n=== Method: {method.upper()} | Prune: {int(pct*100)}% ===")
        model = CNN().to(device)
        model.load_state_dict(baseline_model.state_dict())

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        example_input = torch.randn(1, 1, 28, 28).to(device)

        # Prune
        model = prune_model(model, example_input, method=method, prune_amount=pct)

        start_train = time.time()
        train(model, device, train_loader, optimizer, criterion, epoch=2)
        train_time = time.time() - start_train

        _, acc, infer_time = evaluate(model, device, test_loader)
        size = model_size(model)
        flops, params = get_flops_and_params(model, example_input)

        results.append({
            "method": method,
            "prune_percent": int(pct*100),
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

# Save results to CSV
csv_file = "mnist_pruning_results.csv"
with open(csv_file, mode="w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print(f"\nResults saved to {csv_file}")

# Print Summary Table
print("\n=== Pruning Summary ===")
header = f"{'Method':<10} | {'Prune%':<7} | {'Acc (%)':<8} | {'Drop (%)':<9} | {'Size (MB)':<10} | {'Params (M)':<11} | {'FLOPs (M)':<10} | {'Train Time':<12} | {'Infer Time':<12}"
print(header)
print("-" * len(header))

for r in results:
    print(f"{r['method']:<10} | {r['prune_percent']:<7} | {r['accuracy']:<8.2f} | {r['accuracy_drop']:<9.2f} | "
          f"{r['size_MB']:<10.2f} | {r['params_M']:<11.2f} | {r['flops_M']:<10.2f} | "
          f"{r['train_time_s']:<12.2f} | {r['inference_time_s']:<12.2f}")
