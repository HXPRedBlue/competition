import pandas as pd
import torch

from config import config

seed = 15



def process_data():
    label = pd.read_csv(config.label)
    
    train_data = label.sample(frac=config.train_data_rate, random_state=seed).sort_index().reset_index(drop=True)
    val_data = label[~label.index.isin(train_data.index)].sort_index().reset_index(drop=True)
    data = {"train" : train_data, "val": val_data}
    torch.save(data, config.save_data)
    print("save data success")


def load_data(path):
    data = torch.load(path)
    df = data["train"]
    print(df.loc[0])
if __name__ == '__main__':
    
    process_data()
    
    load_data(config.save_data)