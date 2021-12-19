from torch.utils.data import Dataset
import torch
from scipy.io import loadmat
import os
import random
from torchvision import transforms 
import numpy as np
from scipy.signal import savgol_filter
from scipy import signal
import scipy
import scipy.io as sio


def _np_array(x):
    return np.array(x, dtype=np.float32)

class ECGDataset(Dataset):
    def __init__(self, data_path) -> None:
        super().__init__()
        labels = 12
        train_lines = open(f'{data_path}/trainreference.csv.bak').read().splitlines()
        train_l_ori = np.array([
                        np.sum(np.eye(labels)[np.array(i.strip().split(',')[1:]).astype(int) - 1], axis=0)
                        for i in train_lines
                        ], dtype=np.float32)
        train_names = [f"{data_path}/Train/{i.strip().split(',')[0]}" for i in train_lines]
        train_s_ori = [sio.loadmat(x)['ecgdata'] for x in train_names]
        train_s_ori = np.array(train_s_ori, dtype=np.float32)

        self.data = train_s_ori
        self.label = train_l_ori


    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):

        return _np_array(self.data[index]), _np_array(self.label[index])