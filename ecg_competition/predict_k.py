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
import glob


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

def test_model(model, test_loader, aug=False):
    test_pred_max = []
    test_pred_nomax = []
    with torch.no_grad():
        model.eval()
        for i, data in tqdm(enumerate(test_loader)):
            names, inputs = data
            inputs = inputs.to(device)
            predictions = model(inputs)

            pred = nn.Sigmoid()(predictions)
            predictions = pred.detach().cpu().numpy()
            pred = (pred.detach().cpu().numpy() > 0.5).astype(int)

            test_pred_max.extend(pred)
            test_pred_nomax.extend(predictions)


    return test_pred_max, test_pred_nomax


if __name__ == '__main__':
    val_data_root = "/ai/223/competition/ecg/task2/Test/"
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    files = os.listdir(val_data_root)
    names = []
    tags = []
    files.sort()
    dataset = ECGDataset()

    data_loader = DataLoader(dataset, num_workers=16)
    test_pred = np.zeros((len(data_loader), 12), dtype=np.float32)
    weights = [
        "ecg_0_12_0.8587013920405796.pth",
        "ecg_1_5_0.8556711235249899.pth",
        "ecg_2_11_0.8615479237952182.pth",
        "ecg_3_10_0.8605205943695171.pth",
        "ecg_4_14_0.8531236369915708.pth"
    ]
    for fold_idx in range(5):
        model = ECGNet(input_channel=1,num_classes=12).to(device)

        # load model weights
        weights_path = f'./weights/{weights[fold_idx]}'
        assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
        model.load_state_dict(torch.load(weights_path, map_location=device))
        test_pred_max, test_pred_nomax = test_model(model, data_loader)
        if True:
            test_pred += test_pred_nomax
        else:
            test_pred += test_pred_max
    test_pred /= 5

    test_path = glob.glob(f'/ai/223/competition/ecg/task2/Test/*.mat')
    # test_path = glob.glob('../dataset/train/*.mat')
    test_path = [os.path.basename(x)[:-4] for x in test_path]
    test_path.sort()
    with open(f'./test1.csv','w') as f:
        for name, result in zip(test_path, test_pred):
            # result_cls = (np.argwhere((result > 0.4).astype(int) == 1) + 1).flatten()
            result_cls = result
            if len(result_cls) > 0:
                cls = ','.join(str(i) for i in result_cls)
            else:
                # cls = ','.join(str(result.argmax()+1))
                cls = str(result.argmax()+1)   # 修复bug
            f.write(name + ',' + cls + '\n')

    #将DataFrame存储为csv,index表示是否显示行名，default=True
    # dataframe.to_csv("answer1.csv",index=False,sep=',')
