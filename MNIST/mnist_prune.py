import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils.prune as prune
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import os
import copy

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define transforms and data loaders
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

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

def train(model, optimizer, criterion, masks, epochs=2):
    model.train()
    start_time = time.time()
    for epoch in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            for name, module in model.named_modules():
                if name in masks:
                    module.weight.data *= masks[name]
    return time.time() - start_time

def evaluate(model):
    model.eval()
    test_loss, correct = 0, 0
    start = time.time()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    inference_time = time.time() - start
    accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy, inference_time

def get_size(model):
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p") / 1e6
    os.remove("temp.p")
    return size

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

def dummy_flops(model):
    # Placeholder for FLOPs, could use ptflops or fvcore
    return count_params(model) * 3.5

def apply_masks(model):
    masks = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            mask = (module.weight != 0).float()
            masks[name] = mask
    return masks

def prune_model(method, model, amount):
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if method == 'l1':
                prune.l1_unstructured(module, name='weight', amount=amount)
            elif method == 'random':
                prune.random_unstructured(module, name='weight', amount=amount)
            elif method == 'ln':
                prune.ln_structured(module, name='weight', amount=amount, n=2, dim=0)
            elif method == 'rand_struct':
                prune.random_structured(module, name='weight', amount=amount, dim=0)
    return apply_masks(model)

def run_experiment():
    methods = {
        'baseline': None,
        'l1_unstr': 'l1',
        'random_unstr': 'random',
        'ln_struct': 'ln',
        'rand_struct': 'rand_struct'
    }

    baseline_model = CNN().to(device)
    optimizer = optim.Adam(baseline_model.parameters(), lr=0.001)
    criterion = nn.NLLLoss()
    print("Training baseline...")
    train(baseline_model, optimizer, criterion, {}, epochs=2)
    original_acc, original_infer = evaluate(baseline_model)
    original_size = get_size(baseline_model)
    original_params = count_params(baseline_model)
    original_flops = dummy_flops(baseline_model)

    results = []
    results.append(['baseline', original_acc, 0.0, original_size, original_params, original_flops, 'N/A', original_infer])

    for key, method in methods.items():
        if key == 'baseline':
            continue
        model = CNN().to(device)
        model.load_state_dict(copy.deepcopy(baseline_model.state_dict()))
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        masks = prune_model(method, model, amount=0.5)
        train_time = train(model, optimizer, criterion, masks, epochs=2)
        acc, infer_time = evaluate(model)
        size = get_size(model)
        params = count_params(model)
        flops = dummy_flops(model)

        results.append([
            key, acc, original_acc - acc, size, params, flops, train_time, infer_time
        ])

    print("\n=== Pruning Summary ===")
    print(f"{'Method':<12} | {'Acc (%)':<8} | {'Drop (%)':<9} | {'Size (MB)':<10} | {'Params (M)':<11} | {'FLOPs (M)':<10} | {'Train Time':<11} | {'Infer Time'}")
    print("-" * 100)
    for r in results:
        print(f"{r[0]:<12} | {r[1]:<8.2f} | {r[2]:<9.2f} | {r[3]:<10.2f} | {r[4]:<11.2f} | {r[5]:<10.2f} | {str(r[6]):<11} | {r[7]:.2f}")

if __name__ == '__main__':
    run_experiment()
