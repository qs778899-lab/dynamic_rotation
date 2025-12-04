import torch
from torch.utils.data import Dataset


class BGRXYDataset(Dataset):
    """The BGRXY Datast."""

    def __init__(self, bgrxys, gxyangles):
        self.bgrxys = bgrxys
        self.gxyangles = gxyangles

    def __len__(self):
        return len(self.bgrxys)

    def __getitem__(self, index):
        bgrxy = torch.tensor(self.bgrxys[index], dtype=torch.float32)
        gxyangles = torch.tensor(self.gxyangles[index], dtype=torch.float32)
        return bgrxy, gxyangles


