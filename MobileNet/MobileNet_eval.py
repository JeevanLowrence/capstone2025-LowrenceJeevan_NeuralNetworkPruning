import torch
import torch.nn as nn
import torchvision.transforms as transforms
from pathlib import Path
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Define paths for dataset and model
BASE_DIR = Path(__file__).resolve().parent.parent.parent
CIFAR_DIR = BASE_DIR / "Dataset" / "CIFAR"
META_FILE = CIFAR_DIR / "batches.meta"
TEST_BATCH = CIFAR_DIR / "test_batch"
MODEL_PATH = Path(__file__).parent / "cifar10_mobilenetv2_x1_0-fe6a5b48.pt"

# Custom Dataset class for CIFAR-10 batch files
class CIFAR10Dataset(Dataset):
    def __init__(self, batch_file, transform=None):
        with open(batch_file, 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
        self.data = batch[b'data']  # Raw pixel data (N, 3072)
        self.labels = batch[b'labels']
        self.transform = transform

        # Reshape from flat array to image format (N, 3, 32, 32)
        self.data = self.data.reshape(-1, 3, 32, 32).astype(np.uint8)
        # Convert to (N, 32, 32, 3) for transforms compatibility
        self.data = np.transpose(self.data, (0, 2, 3, 1))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.data[idx]  # Single image
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

# Load class names from meta file
def load_class_names(meta_file):
    with open(meta_file, 'rb') as f:
        meta = pickle.load(f, encoding='bytes')
    return meta[b'label_names']

# Evaluate the model on test data
def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def main():
    # Select device (GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    # Image transformations (convert to tensor and normalize)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    # Load test data
    test_dataset = CIFAR10Dataset(TEST_BATCH, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    # Load class names
    class_names = load_class_names(META_FILE)
    print("Class names:", [name.decode('utf-8') for name in class_names])

    # Load pre-trained MobileNetV2 model
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_mobilenetv2_x1_0", pretrained=False, trust_repo=True)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    # Evaluate model accuracy
    accuracy = evaluate_model(model, test_loader, device)
    print(f'Test Accuracy of MobileNetV2_x1_0: {accuracy:.2f}%')

if __name__ == '__main__':
    main()
