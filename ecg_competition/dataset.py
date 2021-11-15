from torch.utils.data import Dataset
import torch
from scipy.io import loadmat
import os

class ECGDataset(Dataset):
    def __init__(self, data_path, train=True) -> None:
        super().__init__()
        print(data_path)
        data = torch.load(data_path)
        print(data)
        self.data = data["train"] if train else data["val"]
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        label = self.data.loc[index]["tag"]
        file = self.data.loc[index]["name"]
        data = loadmat(os.path.join("data/train", file))
        ecg_data = data["ecgdata"]
        
        return ecg_data, label