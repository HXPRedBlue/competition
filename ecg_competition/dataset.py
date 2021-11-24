from torch.utils.data import Dataset
import torch
from scipy.io import loadmat
import os
import random

class ECGDataset(Dataset):
    def __init__(self, data_path, train=True) -> None:
        super().__init__()
        print(data_path)
        data = torch.load(data_path)
        print(data)
        self.data = data["train"] if train else data["val"]
        self.train = train
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        label = self.data.loc[index]["tag"]
        file = self.data.loc[index]["name"]
        data = loadmat(os.path.join("data/train", file))
        num = random.randint(0,200)
        if not self.train:
            num = 0
        ecg_data = data["ecgdata"][:,num:num+3000]
        
        return ecg_data, label