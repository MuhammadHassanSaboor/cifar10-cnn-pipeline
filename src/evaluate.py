import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import confusion_matrix
from torch.utils.data import TensorDataset, DataLoader
from train import CNN

PROC, MODELS = "data/processed", "models"
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

def main():
    p = yaml.safe_load(open("params.yaml"))["train"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    x = torch.tensor(np.load(f"{PROC}/x_test.npy"))
    y = torch.tensor(np.load(f"{PROC}/y_test.npy")).long()
    dl = DataLoader(TensorDataset(x, y), batch_size=p["batch_size"])

    model = CNN(p["num_filters"], p["dropout_rate"]).to(device)
    model.load_state_dict(torch.load(f"{MODELS}/model.pth", map_location=device))
    model.eval()

    loss_fn = nn.CrossEntropyLoss()
    total_loss, preds, trues = 0.0, [], []
    with torch.no_grad():
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            total_loss += loss_fn(out, yb).item() * len(yb)
            preds.extend(out.argmax(1).cpu().numpy())
            trues.extend(yb.cpu().numpy())

    preds, trues = np.array(preds), np.array(trues)
    metrics = {"test_accuracy": float((preds == trues).mean()),
               "test_loss": float(total_loss / len(trues))}
    json.dump(metrics, open("metrics.json", "w"), indent=2)
    print(metrics)

    cm = confusion_matrix(trues, preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(10), CLASSES)
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig("confusion_matrix.png", dpi=120)

if __name__ == "__main__":
    main()