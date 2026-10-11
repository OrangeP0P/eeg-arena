import torch
from torch.utils.data import Dataset


class FinetuneReshapeDataset(Dataset):
    """
    Reshape data from shape (C, T) to (C, T//patch_size, patch_size)
    Applicable for models requiring patchification
    """
    def __init__(self, dataset, patch_size=200):
        self.dataset = dataset
        self.patch_size = patch_size

        # Support both TensorDataset (has .tensors) and custom Dataset
        if hasattr(dataset, 'tensors'):
            self.data = dataset.tensors[0]
            self.labels = dataset.tensors[1]
        else:
            # For custom Dataset, we'll load on-demand
            self.data = None
            self.labels = None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Handle both pre-loaded and lazy-loaded datasets
        if self.data is not None:
            x = self.data[idx]
            y = self.labels[idx]
        else:
            x, y = self.dataset[idx]

        c, t = x.shape
        if t % self.patch_size != 0:
            raise ValueError(f"Time length {t} cannot be divided by patch_size {self.patch_size}")

        num_patches = t // self.patch_size
        x_reshaped = x.view(c, num_patches, self.patch_size)
        return x_reshaped, y.long()

    def collate(self, batch):
        x_batch = torch.stack([item[0] for item in batch])
        y_batch = torch.stack([item[1] for item in batch])
        return x_batch, y_batch

    def close(self):
        if hasattr(self.dataset, 'close'):
            self.dataset.close()
