import torch
import torch.nn as nn
import torch.optim as optim

# Simple dataset (random for demonstration)
X = torch.randn(100, 3)   # 100 samples, 3 features
y = torch.randn(100, 1)   # regression target

# Define a small neural network
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = Net()

# Use the Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Loss function
criterion = nn.MSELoss()

# Training loop
for epoch in range(600):
    # Forward pass
    y_pred = model(X)
    loss = criterion(y_pred, y)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")