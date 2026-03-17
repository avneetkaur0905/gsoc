"""
Retrain Style CNN and Style CNN+RNN for 5 epochs with ImageNet normalisation.
Matches the preprocessing pipeline in task1_final.ipynb exactly.
"""
import os, sys, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE       = Path("/Users/avneet/project/assignment")
WIKIART    = BASE / "datasets" / "ArtGAN" / "WikiArt Dataset" / "Style"
IMAGE_ROOT = BASE / "datasets" / "wikiart_images" / "wikiart"
ARTIFACT   = BASE / "artifacts"

random.seed(42); np.random.seed(42); torch.manual_seed(42)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Device: {device}")

# ── Transforms (WITH ImageNet normalisation — matching task1_final.ipynb) ──────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ── Dataset ────────────────────────────────────────────────────────────────────
class WikiArtDataset(Dataset):
    def __init__(self, df, image_root, transform=None):
        valid = [i for i, row in df.iterrows()
                 if (image_root / row["image_path"]).exists()]
        self.df = df.loc[valid].reset_index(drop=True)
        self.image_root = image_root
        self.transform  = transform

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        img   = Image.open(self.image_root / row["image_path"]).convert("RGB")
        label = int(row["label"])
        if self.transform: img = self.transform(img)
        return img, label

# ── Load Style data ────────────────────────────────────────────────────────────
def load_split(csv_path):
    rows = []
    with open(csv_path) as f:
        for line in f:
            parts = line.strip().rsplit(",", 1)
            if len(parts) == 2:
                try:
                    rows.append({"image_path": parts[0].strip(), "label": int(parts[1].strip())})
                except ValueError:
                    pass
    return pd.DataFrame(rows)

style_train = load_split(WIKIART / "style_train.csv")
style_val   = load_split(WIKIART / "style_val.csv")
num_classes = len(sorted(style_train["label"].unique()))
print(f"Style classes: {num_classes}")

def make_loaders(train_df, val_df, train_size=5000, val_size=1500, batch_size=32):
    tr = train_df.sample(min(train_size, len(train_df)), random_state=42)
    va = val_df.sample(min(val_size,   len(val_df)),   random_state=42)
    return (
        DataLoader(WikiArtDataset(tr, IMAGE_ROOT, train_transform),
                   batch_size=batch_size, shuffle=True,  num_workers=0),
        DataLoader(WikiArtDataset(va, IMAGE_ROOT, val_transform),
                   batch_size=batch_size, shuffle=False, num_workers=0),
    )

train_loader, val_loader = make_loaders(style_train, style_val)
print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

# ── Models ─────────────────────────────────────────────────────────────────────
class BaselineCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        resnet = models.resnet18(weights="DEFAULT")
        resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
        for name, mod in resnet.named_children():
            self.add_module(name, mod)

    def forward(self, x):
        x = self.conv1(x);  x = self.bn1(x);  x = self.relu(x);  x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x); x = torch.flatten(x, 1)
        return self.fc(x)

class CNNRNN(nn.Module):
    def __init__(self, num_classes, hidden_dim=256, num_layers=1):
        super().__init__()
        backbone = models.resnet18(weights="DEFAULT")
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])
        self.gru = nn.GRU(input_size=512, hidden_size=hidden_dim,
                          num_layers=num_layers, batch_first=True)
        self.dropout    = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        feats = self.feature_extractor(x)
        B, C, H, W = feats.shape
        seq = feats.view(B, C, H*W).permute(0, 2, 1)
        _, h = self.gru(seq)
        return self.classifier(self.dropout(h[-1]))

# ── Training ───────────────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimiser, device):
    training = optimiser is not None
    model.train() if training else model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.set_grad_enabled(training):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out  = model(imgs)
            loss = criterion(out, labels)
            if training:
                optimiser.zero_grad(); loss.backward(); optimiser.step()
            total_loss += loss.item() * len(labels)
            correct    += (out.argmax(1) == labels).sum().item()
            total      += len(labels)
    return total_loss / total, correct / total

def train_model(model, train_loader, val_loader, epochs, lr, save_path):
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    history, best_acc = [], 0.0
    for epoch in range(1, epochs+1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimiser, device)
        va_loss, va_acc = run_epoch(model, val_loader,   criterion, None,      device)
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                         "val_loss": va_loss, "val_acc": va_acc})
        print(f"  Epoch {epoch}/{epochs}  train_acc={tr_acc:.4f}  val_acc={va_acc:.4f}", flush=True)
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), save_path)
            print(f"    -> Saved best checkpoint ({best_acc:.4f})", flush=True)
    return pd.DataFrame(history), best_acc

EPOCHS = 5

# ── Train CNN ──────────────────────────────────────────────────────────────────
print("\n=== Training Style BaselineCNN (5 epochs, with normalisation) ===")
cnn_path = ARTIFACT / "task1_style_baseline" / "best_resnet18_style.pth"
cnn_path.parent.mkdir(parents=True, exist_ok=True)
style_cnn = BaselineCNN(num_classes).to(device)
cnn_history, cnn_best = train_model(style_cnn, train_loader, val_loader,
                                     EPOCHS, lr=1e-4, save_path=cnn_path)
cnn_history.to_csv(ARTIFACT / "task1_style_baseline" / "baseline_history.csv", index=False)
print(f"Style CNN best val acc: {cnn_best:.4f}")

# ── Train CNN+RNN ──────────────────────────────────────────────────────────────
print("\n=== Training Style CNNRNN (5 epochs, with normalisation) ===")
rnn_path = ARTIFACT / "task1_style_cnn_rnn" / "best_cnn_rnn_style.pth"
rnn_path.parent.mkdir(parents=True, exist_ok=True)
style_rnn = CNNRNN(num_classes).to(device)
rnn_history, rnn_best = train_model(style_rnn, train_loader, val_loader,
                                     EPOCHS, lr=5e-5, save_path=rnn_path)
rnn_history.to_csv(ARTIFACT / "task1_style_cnn_rnn" / "cnn_rnn_history.csv", index=False)
print(f"Style CNN+RNN best val acc: {rnn_best:.4f}")

print("\n=== ALL DONE ===")
print(f"Style CNN best:     {cnn_best:.4f}")
print(f"Style CNN+RNN best: {rnn_best:.4f}")
