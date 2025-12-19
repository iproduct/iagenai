import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ----------------------------
# Data augmentations (SimCLR)
# ----------------------------
class SimCLRAugment:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x)

if __name__ == '__main__':
    # ----------------------------
    # Dataset
    # ----------------------------
    train_set = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=SimCLRAugment(),
    )

    loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=4)

    # ----------------------------
    # ViT Components
    # ----------------------------
    class PatchEmbedding(nn.Module):
        def __init__(self, embed_dim=256):
            super().__init__()
            self.proj = nn.Conv2d(3, embed_dim, kernel_size=16, stride=16)
            self.num_patches = (224 // 16) ** 2

        def forward(self, x):
            x = self.proj(x)
            return x.flatten(2).transpose(1, 2)

    class TransformerBlock(nn.Module):
        def __init__(self, dim, heads):
            super().__init__()
            self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)
            self.mlp = nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim),
            )

        def forward(self, x):
            attn, _ = self.attn(x, x, x)
            x = self.norm1(x + attn)
            return self.norm2(x + self.mlp(x))

    class ViT(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch = PatchEmbedding()
            self.pos = nn.Parameter(torch.randn(1, self.patch.num_patches, 256))
            self.blocks = nn.Sequential(*[TransformerBlock(256, 4) for _ in range(6)])
            self.norm = nn.LayerNorm(256)

        def forward(self, x):
            x = self.patch(x) + self.pos
            x = self.blocks(x)
            return self.norm(x).mean(dim=1)

    class ProjectionHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
            )

        def forward(self, x):
            return self.net(x)

    # ----------------------------
    # Contrastive loss
    # ----------------------------
    def nt_xent(z1, z2, t=0.2):
        z1, z2 = F.normalize(z1, dim=1), F.normalize(z2, dim=1)
        logits = z1 @ z2.T / t
        labels = torch.arange(z1.size(0), device=z1.device)
        return (F.cross_entropy(logits, labels) +
                F.cross_entropy(logits.T, labels)) / 2

    # ----------------------------
    # Training loop
    # ----------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = ViT().to(device)
    projector = ProjectionHead().to(device)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(projector.parameters()), lr=3e-4
    )

    for epoch in range(10):
        for (x1, x2), _ in loader:
            x1, x2 = x1.to(device), x2.to(device)
            z1 = projector(encoder(x1))
            z2 = projector(encoder(x2))
            loss = nt_xent(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")
