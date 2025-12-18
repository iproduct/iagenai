import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50
import matplotlib.pyplot as plt


# --------------------------------------------------
# Lookahead optimizer wrapper
# --------------------------------------------------
class Lookahead(torch.optim.Optimizer):
    def __init__(self, optimizer, alpha=0.5, k=6):
        self.optimizer = optimizer
        self.alpha = alpha
        self.k = k
        self.step_counter = 0

        # Cache the slow weights
        self.slow_weights = [
            p.clone().detach()
            for group in optimizer.param_groups
            for p in group["params"]
            if p.requires_grad
        ]
        for w in self.slow_weights:
            w.requires_grad = False

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        loss = self.optimizer.step()
        self.step_counter += 1

        if self.step_counter % self.k != 0:
            return loss

        # Sync fast weights to slow weights
        idx = 0
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                slow = self.slow_weights[idx]
                slow.data.add_(self.alpha, p.data - slow.data)
                p.data.copy_(slow.data)
                idx += 1
        return loss

    def state_dict(self):
        return {
            "optimizer": self.optimizer.state_dict(),
            "slow_weights": self.slow_weights,
            "alpha": self.alpha,
            "k": self.k,
            "step_counter": self.step_counter,
        }


# --------------------------------------------------
# Device
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --------------------------------------------------
# Data transforms
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

# --------------------------------------------------
# CIFAR10
# --------------------------------------------------
train_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=128,
                          shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=128,
                         shuffle=False, num_workers=4, pin_memory=True)

# --------------------------------------------------
# Model
# --------------------------------------------------
model = resnet50(weights=None)
model.fc = nn.Linear(2048, 10)
model = model.to(device)

# --------------------------------------------------
# Loss with label smoothing
# --------------------------------------------------
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# --------------------------------------------------
# Optimizer + Lookahead
# --------------------------------------------------
base_optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer = Lookahead(base_optimizer, alpha=0.5, k=5)

# --------------------------------------------------
# Cosine LR Scheduler + Warmup
# --------------------------------------------------
epochs = 30
warmup_epochs = 5
total_steps = len(train_loader) * epochs
warmup_steps = len(train_loader) * warmup_epochs

scheduler = CosineAnnealingLR(base_optimizer, T_max=epochs)

# --------------------------------------------------
# Mixed precision scaler
# --------------------------------------------------
scaler = torch.cuda.amp.GradScaler()


# --------------------------------------------------
# Evaluation
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
                _, preds = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

    model.train()
    return correct / total


# --------------------------------------------------
# Training Loop
# --------------------------------------------------
train_acc_history = []
test_acc_history = []

global_step = 0

for epoch in range(epochs):
    for images, labels in train_loader:
        global_step += 1

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        # AMP forward
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        # AMP backward
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Warmup LR
        if global_step < warmup_steps:
            warmup_factor = global_step / warmup_steps
            for param_group in base_optimizer.param_groups:
                param_group["lr"] = warmup_factor * 0.001

    scheduler.step()

    train_acc = evaluate(train_loader)
    test_acc = evaluate(test_loader)
    train_acc_history.append(train_acc)
    test_acc_history.append(test_acc)

    print(f"Epoch {epoch+1}/{epochs} | "
          f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

# --------------------------------------------------
# Accuracy Plot
# --------------------------------------------------
plt.figure(figsize=(7, 5))
plt.plot(train_acc_history, label="Train Accuracy")
plt.plot(test_acc_history, label="Test Accuracy")
plt.title("ResNet-50 on CIFAR-10 (Cosine LR + Warmup + Lookahead + Label Smoothing)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()
plt.show()
