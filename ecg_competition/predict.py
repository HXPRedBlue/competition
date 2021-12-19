import os
import torch
from scipy.io import loadmat
import pandas as pd
from tqdm import tqdm
from models.resnet import resnet101, resnet50, resnet34
from models.ECGNet import ECGNet

from config import config
import numpy as np
from scipy.signal import savgol_filter
import torch.nn as nn
from data.test_data import ECGDataset
from torch.utils.data import DataLoader


def predict(path):
    


    # load image
    ecg_data = path
    # ecg_data = data["ecgdata"][:,0:3000]
    # ecg_data = savgol_filter(ecg_data, 51, 3)
    # ecg_data = np.array([data for data in ecg_data[:,::5]])
    ecg_data = torch.tensor(ecg_data)
    # expand batch dimension
    img = ecg_data

    # read class_indict
    class_indict = {}

    # create model
    # model = resnet34(num_classes=2).to(device)


    # prediction
    model.eval()
    with torch.no_grad():
        # predict class
        # output = torch.squeeze(model(img.to(device))).cpu()
        # predict = torch.softmax(output, dim=0)
        # predict_cla = torch.argmax(predict).numpy()

        prob = model(img.to(device)) #表示模型的预测输出
        prob = nn.Sigmoid()(prob)[0]
        predict_cla = (prob.detach().cpu().numpy() > 0.5).astype(int)

        result_cls = (np.argwhere((prob.detach().cpu().numpy() > 0.5).astype(int) == 1) + 1).flatten()

        print(f"predict_cla:{predict_cla}")
        print(f"result_cls:{result_cls}")

    # print_res = "class: {}   prob: {:.3}".format(str(predict_cla),
    #                                              predict[predict_cla].numpy())
    return prob.detach().cpu().numpy()


if __name__ == '__main__':
    val_data_root = "/ai/223/competition/ecg/task2/Test/"
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    files = os.listdir(val_data_root)
    names = []
    tags = []
    files.sort()
    dataset = ECGDataset()
    model = ECGNet(input_channel=1,num_classes=12).to(device)

    # load model weights
    weights_path = "./work_dir/ECG_lr0.0001.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    data_loader = DataLoader(dataset, num_workers=16)
    for file in tqdm(data_loader):
        name, data = file
        labels = predict(data)
        names.append(name[0])
        tags.append(labels)
    with open(f'answer2.csv','w') as f:
        for name, labels in zip(names, tags):
            result_cls = (np.argwhere((labels > 0.5).astype(int) == 1) + 1).flatten()
            if len(result_cls) > 0:
                cls = ','.join(str(i) for i in result_cls)
            else:
                cls = ','.join(str(labels.argmax()+1))
            f.write(name + ',' + cls + '\n')

    #将DataFrame存储为csv,index表示是否显示行名，default=True
    # dataframe.to_csv("answer1.csv",index=False,sep=',')
