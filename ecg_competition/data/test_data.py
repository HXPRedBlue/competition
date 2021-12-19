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
    def __init__(self) -> None:
        val_data_root = "/ai/223/competition/ecg/task2/Test/"
        self.files = os.listdir(val_data_root)
        self.files.sort()
        self.train_s_ori = [sio.loadmat(val_data_root + x)['ecgdata'] for x in self.files]


    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):

        return self.files[index], self.train_s_ori[index]