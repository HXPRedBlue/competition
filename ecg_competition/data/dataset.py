from torch.utils.data import Dataset
import torch
from scipy.io import loadmat
import os
import random
from torchvision import transforms 

class ECGDataset(Dataset):
    def __init__(self, data_path, train=True) -> None:
        super().__init__()
        print(data_path)
        data = torch.load(data_path)
        print(data)
        self.data = data["train"] if train else data["val"]
        self.train = train
        self.transformer = transforms.Normalize(normMean = [0.02102842710458631, 0.01870473789104995, -0.002323689225004715, -0.016818765162802455, 0.012285585395475473, 0.008799268047423623, -0.010402479687361263, -0.021703821823799017, -0.005407059704317367, 0.023126367419523692, 0.03341531410865756, 0.016268959850275762], 
        normStd = [0.3053013037015497, 0.24986147063669015, 0.29824570044148807, 0.23575938566213706, 0.27472264903231897, 0.2288858965974916, 0.3108398675616677, 0.38992954676508795, 0.46233869773182745, 0.40961406719874305, 0.3935515632033764, 0.4898131524605047])
        
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