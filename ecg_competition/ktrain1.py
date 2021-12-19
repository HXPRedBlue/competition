from datetime import datetime
from operator import le
import os
import json
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, average_precision_score
from torchvision import transforms, datasets
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from tensorboardX import SummaryWriter
from sklearn.model_selection import KFold
import pandas as pd
from torchsummary import summary
from torch.optim.lr_scheduler import CosineAnnealingLR

from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma
from types import SimpleNamespace
from scipy.signal import savgol_filter
from scipy.io import loadmat
import pandas as pd

from datetime import datetime
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(BASE_DIR)
#添加系统环境变量
sys.path.append(BASE_DIR)
os.chdir(BASE_DIR)
import numpy as np
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from tensorboardX import SummaryWriter

from models.resnet import resnet50
from models.ECGNet import ECGNet
from config import config
from data.task2_dataset_all import ECGDataset

device = torch.device("cuda:1")
print("using {} device.".format(device))
batch_size = 32
epochs = 15
nw = 8  # number of workers
k_fold = 5
classes_num = 2
lr = 0.0001

def train():

    # ------------------------------------ step 1/5 : 加载数据------------------------------------

    all_data = ECGDataset("/ai/223/competition/ecg/task2/")
    kfold = KFold(n_splits=k_fold, shuffle=True,random_state=233333)

    for fold_idx, (train_index, val_index) in enumerate(kfold.split(all_data)):
        # if fold_idx <= 1:
        #     continue
        print('*'*25,'第', fold_idx + 1,'折','*'*25)
        writer = SummaryWriter(f"./tensorboard/{datetime.now().strftime('%y%m%d_%H%M')}_{fold_idx}")   # 数据存放在这个文件夹
        train_data = Subset(all_data, train_index)
        val_data = Subset(all_data, val_index)

        train_loader = DataLoader(train_data,batch_size=batch_size, shuffle=True,
                                            num_workers=nw)

        validate_loader = DataLoader(val_data,batch_size=batch_size, shuffle=True,
                                            num_workers=nw)

        mixup_args = {
            'mixup_alpha': 1.,
            'cutmix_alpha': 0.,
            'cutmix_minmax': None,
            'prob': 1.0,
            'switch_prob': 0.,
            'mode': 'batch',
            'label_smoothing': 0,
            'num_classes': 12}
        mixup_fn = Mixup(**mixup_args)
        # ------------------------------------ step 2/5 : 初始化网络------------------------------------

        net = ECGNet(input_channel=1,num_classes=12).to(device)
        # net.load_state_dict(torch.load("./work_dir/ECG_lr0.0001.pth", map_location=device))
        ema_model = ModelEma(net)

        # ------------------------------------ step 3/5 : 定义损失函数和优化器 ------------------------------------
        # define loss function
        # loss_function = nn.CrossEntropyLoss()
        # loss_function = LabelSmoothingCrossEntropy()
        loss_function = nn.BCEWithLogitsLoss()
        # loss_function = nn.BCEWithLogitsLoss()
        val_loss_function = nn.BCEWithLogitsLoss()
        

        optimizer = optim.Adam(net.parameters(), lr=lr)
        # args = SimpleNamespace()
        # args.lr = lr
        # args.weight_decay = 0.05
        # args.opt = 'adam' #'lookahead_adam' to use `lookahead`
        # args.momentum = 0.9 
        # optimizer = create_optimizer(args=args, model=net)
        # scheduler = CosineAnnealingLR(optimizer,T_max=20)


        # ------------------------------------ step 4/5 : 训练并保存模型 --------------------------------------------------

        best_acc = 0.0
        

        for epoch in range(epochs):
            
            # train
            net.train()
            running_loss = 0.0
            train_bar = tqdm(train_loader)
            for step, data in enumerate(train_bar):
                images, labels = data
                images, labels = mixup_fn(images, labels)
                optimizer.zero_grad()
                logits = net(images.to(device))
                loss = loss_function(logits, labels.to(device))
                loss.backward()
                optimizer.step()
                # scheduler.step()

                # print statistics
                running_loss += loss.item()

                train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                            epochs,
                                                                            loss)

                ema_model.update(net)

                # print statistics
                running_loss += loss.item()
            # scheduler.step()
            # train_acc = accuracy_score(train_trues, train_pres)                                                           
            # validate
            net.eval()
            with torch.no_grad():

                prob_all = []
                label_all = []
                val_bar = tqdm(validate_loader)
                for data,label in val_bar:
                    prob = net(data.to(device)) #表示模型的预测输出
                    prob = nn.Sigmoid()(prob)
                    prob_all.extend((prob.detach().cpu().numpy() > 0.5).astype(int)) 
                    label_all.extend(label.cpu().numpy())
            labelss, preds = np.array(label_all, dtype=np.float32), np.array(prob_all, dtype=np.float32)
            ap = np.mean([accuracy_score(labelss[:, i], preds[:, i])
                        for i in range(labelss.shape[1])])
            f1s = np.mean([f1_score(labelss[:, i], preds[:, i]) for i in range(labelss.shape[1])])
            hb_ap = np.mean([(labelss[i] == preds[i]).all()
                            for i in range(labelss.shape[0])])
            final_score = 0.8 * f1s + 0.25 * hb_ap
            save_path = f'./weights/ecg_{fold_idx}_{epoch}_{final_score}.pth'
            torch.save(net.state_dict(), save_path)

            print(f"f1_score:{f1s}, final_score:{final_score}")
            
            
        print(best_acc)
        print('Finished Training')

if __name__ == "__main__":

    train()

    # test()
    
    


