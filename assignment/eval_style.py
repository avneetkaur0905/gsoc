"""
Evaluate corrected Style CNN and CNN+RNN checkpoints.
Gets top-1, top-3 accuracy and macro F1 — writes to eval_style_results.csv
Run from Jupyter terminal: python eval_style.py
"""
import os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import f1_score
from PIL import Image
from pathlib import Path

BASE       = Path("/Users/Avneet/projects/GSOC 2026/assignment")
WIKIART    = BASE / "datasets" / "ArtGAN" / "WikiArt Dataset" / "Style"
IMAGE_ROOT = BASE / "datasets" / "wikiart_images" / "wikiart"
ARTIFACT   = BASE / "artifacts"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

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

style_val   = load_split(WIKIART / "style_val.csv")
num_classes = 27
print(f"Val samples: {len(style_val)}")

va = style_val.sample(min(1500, len(style_val)), random_state=42)
val_loader = DataLoader(
    WikiArtDataset(va, IMAGE_ROOT, val_transform),
    batch_size=32, shuffle=False, num_workers=0
)
print(f"Val batches: {len(val_loader)}")

class BaselineCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        resnet = models.resnet18(weights=None)
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
        backbone = models.resnet18(weights=None)
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

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, top3_correct, total = [], [], 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            top3 = out.topk(3, dim=1).indices
            top3_correct += sum(labels[i] in top3[i] for i in range(len(labels)))
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total += len(labels)
    top1 = sum(p == l for p, l in zip(all_preds, all_labels)) / total
    top3 = top3_correct / total
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return top1, top3, macro_f1

results = []

print("\n--- Evaluating Style CNN ---")
cnn = BaselineCNN(num_classes).to(device)
cnn.load_state_dict(torch.load(ARTIFACT / "task1_style_baseline" / "best_resnet18_style.pth",
                                map_location=device))
top1, top3, f1 = evaluate(cnn, val_loader, device)
print(f"  Top-1: {top1:.4f}  Top-3: {top3:.4f}  Macro-F1: {f1:.4f}")
results.append({"task": "Style", "model": "ResNet18 (CNN)",
                "top1_acc": round(top1, 4), "top3_acc": round(top3, 4), "macro_f1": round(f1, 4)})

print("\n--- Evaluating Style CNN+RNN ---")
rnn = CNNRNN(num_classes).to(device)
rnn.load_state_dict(torch.load(ARTIFACT / "task1_style_cnn_rnn" / "best_cnn_rnn_style.pth",
                                map_location=device))
top1, top3, f1 = evaluate(rnn, val_loader, device)
print(f"  Top-1: {top1:.4f}  Top-3: {top3:.4f}  Macro-F1: {f1:.4f}")
results.append({"task": "Style", "model": "ResNet18+GRU (CNN+RNN)",
                "top1_acc": round(top1, 4), "top3_acc": round(top3, 4), "macro_f1": round(f1, 4)})

out_path = BASE / "eval_style_results.csv"
pd.DataFrame(results).to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")
print("\n=== DONE ===")
