import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import TensorDataset, DataLoader

PROC, MODELS = "data/processed", "models"

class CNN(nn.Module):
    def __init__(self, f, dropout):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
                nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
                nn.MaxPool2d(2))
        self.features = nn.Sequential(block(3, f), block(f, f * 2), block(f * 2, f * 4))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(f * 4 * 4 * 4, 256), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(256, 10))

    def forward(self, x):
        return self.classifier(self.features(x))

def augment(x):
    if torch.rand(1).item() < 0.5:
        x = torch.flip(x, dims=[3])
    padded = F.pad(x, (4, 4, 4, 4), mode="reflect")
    i, j = torch.randint(0, 9, (2,))
    return padded[:, :, i:i + 32, j:j + 32]

def load(split):
    x = torch.tensor(np.load(f"{PROC}/x_{split}.npy"))
    y = torch.tensor(np.load(f"{PROC}/y_{split}.npy")).long()
    return TensorDataset(x, y)

def main():
    p = yaml.safe_load(open("params.yaml"))["train"]
    os.makedirs(MODELS, exist_ok=True)
    torch.manual_seed(p["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Training on", device)

    train_dl = DataLoader(load("train"), batch_size=p["batch_size"], shuffle=True)
    val_dl = DataLoader(load("val"), batch_size=p["batch_size"])

    model = CNN(p["num_filters"], p["dropout_rate"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=p["learning_rate"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=p["epochs"])
    loss_fn = nn.CrossEntropyLoss()
    history = []

    for epoch in range(1, p["epochs"] + 1):
        model.train()
        total, correct, running = 0, 0, 0.0
        for xb, yb in train_dl:
            xb, yb = augment(xb).to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            running += loss.item() * len(yb)
            correct += (out.argmax(1) == yb).sum().item()
            total += len(yb)
        sched.step()

        model.eval()
        v_total, v_correct, v_running = 0, 0, 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                v_running += loss_fn(out, yb).item() * len(yb)
                v_correct += (out.argmax(1) == yb).sum().item()
                v_total += len(yb)

        row = {"epoch": epoch, "train_loss": running / total, "train_acc": correct / total,
               "val_loss": v_running / v_total, "val_acc": v_correct / v_total}
        history.append(row)
        print(f"epoch {epoch:02d}  train_acc {row['train_acc']:.4f}  val_acc {row['val_acc']:.4f}")

    torch.save(model.state_dict(), f"{MODELS}/model.pth")
    pd.DataFrame(history).to_csv(f"{MODELS}/history.csv", index=False)
    print("Saved model and history")

if __name__ == "__main__":
    main()