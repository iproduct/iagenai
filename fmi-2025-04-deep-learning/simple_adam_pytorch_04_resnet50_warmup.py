import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.models import resnet50
import math

# --------------------------------------------------
# 1. Select Device
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --------------------------------------------------
# 2. Transforms & CIFAR-10
# --------------------------------------------------
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616)),
])

train_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform_test)

train_loader = DataLoader(
    train_dataset, batch_size=128, shuffle=True,
    num_workers=4, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=128, shuffle=False,
    num_workers=4, pin_memory=True
)

# --------------------------------------------------
# 3. Load ResNet-50 for CIFAR-10
# --------------------------------------------------
model = resnet50(weights=None)      # training from scratch
model.fc = nn.Linear(2048, 10)      # CIFAR-10 classifier
model = model.to(device)

# --------------------------------------------------
# 4. Loss, Optimizer, Warmup + Cosine Scheduler
# --------------------------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scaler = torch.cuda.amp.GradScaler()

# Training configuration
epochs = 40
warmup_epochs = 5      # warmup for first 5 epochs
base_lr = 0.001        # starting lr after warmup
min_lr = 1e-6          # min cosine LR

def get_lr(epoch):
    """Linear warmup for warmup_epochs, then cosine."""
    if epoch < warmup_epochs:
        # warmup: linearly scale from 0 → base_lr
        return base_lr * (epoch + 1) / warmup_epochs
    # cosine decay:
    cosine_epoch = epoch - warmup_epochs
    cosine_total = epochs - warmup_epochs
    return min_lr + (base_lr - min_lr) * \
           0.5 * (1 + math.cos(math.pi * cosine_epoch / cosine_total))

# --------------------------------------------------
# 5. Evaluation Function
# --------------------------------------------------
def evaluate(loader):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                outputs = model(images)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    model.train()
    return correct / total

# --------------------------------------------------
# 6. Training Loop with Warmup + Cosine LR
# --------------------------------------------------
train_acc_history = []
test_acc_history = []
lr_history = []

for epoch in range(epochs):
    # update learning rate
    lr = get_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    lr_history.append(lr)

    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    # evaluate accuracy
    train_acc = evaluate(train_loader)
    test_acc = evaluate(test_loader)
    train_acc_history.append(train_acc)
    test_acc_history.append(test_acc)

    print(f"Epoch {epoch+1}/{epochs} "
          f"LR: {lr:.6f}  "
          f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

# --------------------------------------------------
# 7. Plot Accuracy & LR Schedule
# --------------------------------------------------
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_acc_history, label="Train Accuracy")
plt.plot(test_acc_history, label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("ResNet-50 CIFAR-10 Accuracy (GPU + AMP + Warmup + Cosine)")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(lr_history)
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title("Warmup + Cosine LR Schedule")
plt.grid(True)

plt.show()
