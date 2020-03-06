import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


def get_loader(data: np.array, batch_size: int, is_train=True):
    data = torch.from_numpy(data)
    dataset = TensorDataset(data)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        drop_last=is_train,
    )
    return loader