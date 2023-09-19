from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
import torch


class VoxelDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.int64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {"voxel": self.data[idx], "label": self.labels[idx]}

        return sample


def get_dataloaders(batch_size):
    data = np.load("modelnet10.npz", allow_pickle=True)
    train_voxel = data["train_voxel"]  # Training 3D voxel samples
    test_voxel = data["test_voxel"]  # Test 3D voxel samples
    train_labels = data["train_labels"]  # Training labels (integers from 0 to 9)
    test_labels = data["test_labels"]  # Test labels (integers from 0 to 9)
    class_map = data["class_map"]  # Dictionary mapping the labels to their class names.

    train_dataset = VoxelDataset(train_voxel, train_labels)
    test_dataset = VoxelDataset(test_voxel, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
