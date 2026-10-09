import os
import numpy as np
from torchvision import datasets

RAW = "data/raw"

def main():
    os.makedirs(RAW, exist_ok=True)
    train = datasets.CIFAR10(root="data/download", train=True, download=True)
    test = datasets.CIFAR10(root="data/download", train=False, download=True)

    np.save(f"{RAW}/train_images.npy", train.data)
    np.save(f"{RAW}/train_labels.npy", np.array(train.targets))
    np.save(f"{RAW}/test_images.npy", test.data)
    np.save(f"{RAW}/test_labels.npy", np.array(test.targets))
    print(f"Saved raw CIFAR-10: {train.data.shape} train, {test.data.shape} test")

if __name__ == "__main__":
    main()