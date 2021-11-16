import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from scipy.io import loadmat
import pandas as pd
from tqdm import tqdm
from models.resnet import resnet34


def predict(path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



    # load image
    data = loadmat(path)
    ecg_data = torch.tensor(data["ecgdata"])
    # expand batch dimension
    img = torch.unsqueeze(ecg_data, dim=0)

    # read class_indict
    class_indict = {}

    # create model
    model = resnet34(num_classes=2).to(device)

    # load model weights
    weights_path = "./resNet34.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # prediction
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(str(predict_cla),
                                                 predict[predict_cla].numpy())
    return predict_cla


if __name__ == '__main__':
    val_data_root = "./data/val/"
    files = os.listdir(val_data_root)
    names = []
    tags = []
    files.sort()
    for file in tqdm(files):
        path = val_data_root + file
        label = predict(path)
        names.append(os.path.basename(file))
        tags.append(label)
    dataframe = pd.DataFrame({'name':names,'tag':tags})

    #将DataFrame存储为csv,index表示是否显示行名，default=True
    dataframe.to_csv("test.csv",index=False,sep=',')
