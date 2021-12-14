import os
import torch
from scipy.io import loadmat
import pandas as pd
from tqdm import tqdm
from models.resnet import resnet101, resnet50, resnet34

from config import config
import numpy as np
from scipy.signal import savgol_filter


def predict(path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    # load image
    data = loadmat(path)
    ecg_data = data["ecgdata"][:,0:3000]
    ecg_data = savgol_filter(ecg_data, 51, 3)
    ecg_data = np.array([data for data in ecg_data[:,::5]])
    ecg_data = torch.tensor(ecg_data)
    # expand batch dimension
    img = torch.unsqueeze(ecg_data, dim=0)

    # read class_indict
    class_indict = {}

    # create model
    model = resnet34(num_classes=2).to(device)

    # load model weights
    weights_path = "./weights/resNet34_0.pth"
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
        names.append(os.path.splitext(file)[0])
        tags.append(label)
    dataframe = pd.DataFrame({'name':names,'tag':tags})

    #将DataFrame存储为csv,index表示是否显示行名，default=True
    dataframe.to_csv("answer.csv",index=False,sep=',')
