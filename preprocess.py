import os
import numpy as np
import yaml

# This line edited <-----

RAW, PROC = "data/raw", "data/processed"

def main():
    p = yaml.safe_load(open("params.yaml"))["preprocess"]
    os.makedirs(PROC, exist_ok=True)

    x = np.load(f"{RAW}/train_images.npy").astype(np.float32) / 255.0
    y = np.load(f"{RAW}/train_labels.npy")
    x_test = np.load(f"{RAW}/test_images.npy").astype(np.float32) / 255.0
    y_test = np.load(f"{RAW}/test_labels.npy")

    # standardize with the TRAIN mean/std, per colour channel
    mean = x.mean(axis=(0, 1, 2))
    std = x.std(axis=(0, 1, 2))
    x = (x - mean) / std
    x_test = (x_test - mean) / std

    # (N, H, W, C) -> (N, C, H, W) because PyTorch wants channels first
    x = x.transpose(0, 3, 1, 2)
    x_test = x_test.transpose(0, 3, 1, 2)

    rng = np.random.default_rng(p["seed"])
    idx = rng.permutation(len(x))
    n_val = int(len(x) * p["val_size"])
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    np.save(f"{PROC}/x_train.npy", x[train_idx])
    np.save(f"{PROC}/y_train.npy", y[train_idx])
    np.save(f"{PROC}/x_val.npy", x[val_idx])
    np.save(f"{PROC}/y_val.npy", y[val_idx])
    np.save(f"{PROC}/x_test.npy", x_test)
    np.save(f"{PROC}/y_test.npy", y_test)
    print(f"Train {len(train_idx)}, val {len(val_idx)}, test {len(x_test)}")

if __name__ == "__main__":
    main()