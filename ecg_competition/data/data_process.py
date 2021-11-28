import pandas as pd
import torch
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASE_DIR)
#添加系统环境变量
sys.path.append(BASE_DIR)
os.chdir(BASE_DIR)
from config import config

seed = 15


def process_data():
    label = pd.read_csv(config.label)
    
    train_data = label.sample(frac=config.train_data_rate, random_state=seed)
    val_data = label[~label.index.isin(train_data.index)]
    train_data = train_data.append(train_data).append(train_data)
    train_data = train_data.reset_index(drop=True)
    val_data = val_data.sort_index().reset_index(drop=True)
    data = {"train" : train_data, "val": val_data}
    torch.save(data, config.save_data)
    print("save data success")


def load_data(path):
    data = torch.load(path)
    df = data["val"]
    print(df['tag'].mean())
    df = data["train"]
    print(df['tag'])
if __name__ == '__main__':
    
    # process_data()
    
    load_data(config.save_data)