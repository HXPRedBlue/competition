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
from data.dataset import ECGDataset

device = torch.device("cuda:0")
print("using {} device.".format(device))
batch_size = 32
epochs = 50
nw = 8  # number of workers
k_fold = 5
classes_num = 2
lr = 0.0001



def train():

    # ------------------------------------ step 1/5 : 加载数据------------------------------------

    all_data = ECGDataset(config.save_data,True)
    kfold = KFold(n_splits=k_fold, shuffle=True,random_state=233333)

    for fold_idx, (train_index, val_index) in enumerate(kfold.split(all_data)):
        print('*'*25,'第', fold_idx + 1,'折','*'*25)
        writer = SummaryWriter(f"./tensorboard/{datetime.now().strftime('%y%m%d_%H%M')}_{fold_idx}")   # 数据存放在这个文件夹
        train_data = Subset(all_data, train_index)
        val_data = Subset(all_data, val_index)

        train_loader = DataLoader(train_data,batch_size=batch_size, shuffle=True,
                                            num_workers=nw)

        validate_loader = DataLoader(val_data,batch_size=batch_size, shuffle=True,
                                            num_workers=nw)
        
        # ------------------------------------ step 2/5 : 初始化网络------------------------------------

        net = ECGNet(input_channel=1,num_classes=2).to(device)

        # ------------------------------------ step 3/5 : 定义损失函数和优化器 ------------------------------------
        # define loss function
        loss_function = nn.CrossEntropyLoss()

        optimizer = optim.Adam(net.parameters(), lr=lr)


        # ------------------------------------ step 4/5 : 训练并保存模型 --------------------------------------------------

        best_acc = 0.0
        save_path = f'./weights/ecg_{fold_idx}.pth'
        train_steps = len(train_loader)
        val_steps = len(validate_loader)

        for epoch in range(epochs):
            # train
            net.train()
            running_loss = 0.0
            val_run_loss = 0.0 
            train_bar = tqdm(train_loader)
            train_trues  = []
            train_pres = []
            for step, data in enumerate(train_bar):
                images, labels = data
                optimizer.zero_grad()
                logits = net(images.to(device, dtype=torch.float))
                loss = loss_function(logits, labels.to(device))
                train_pres.extend(logits.argmax(dim=1).cpu().numpy())
                train_trues.extend(labels.cpu().numpy())
                loss.backward()
                optimizer.step()
                # scheduler.step(epoch)

                # print statistics
                running_loss += loss.item()

                train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)
            train_acc = accuracy_score(train_trues, train_pres)                                                           
            # validate
            val_preds = []
            val_trues = []
            val_labels_pr = []
            val_prebs = []
            net.eval()
            with torch.no_grad():
                for i, (val_data, val_labels) in enumerate(validate_loader):
                    outputs = net(val_data.to(device, dtype=torch.float))
                    loss = loss_function(outputs, val_labels.to(device))
                    class_probs_batch = [F.softmax(el, dim=0) for el in outputs]
                    outputs = outputs.argmax(dim=1)
                    val_preds.extend(outputs.cpu().numpy())
                    val_trues.extend(val_labels.cpu().numpy())
                    val_labels_pr.append(val_labels)
                    val_run_loss += loss.item()
                    val_prebs.append(class_probs_batch)

            print(val_labels)
            print(outputs)
            precision = precision_score(val_trues, val_preds)
            recall = recall_score(val_trues, val_preds)
            val_f1 = f1_score(val_trues, val_preds)
            val_accurate = accuracy_score(val_trues, val_preds)
            train_loss = running_loss / train_steps
            val_loss = val_run_loss / val_steps
            ap = average_precision_score(val_trues, val_preds)
            train_f1 = f1_score(train_trues, train_pres)
            print(f'[epoch {epoch + 1}] train_loss: {train_loss}  val_accuracy: {val_accurate} \
            val_f1: {val_f1}, recall:{recall}, train_f1:{train_f1}')
            

            if val_f1 > best_acc:
                best_acc = val_f1
                torch.save(net.state_dict(), save_path)

            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('f1', val_f1, epoch)
            writer.add_scalar('precision', precision, epoch)
            writer.add_scalar('recall', recall, epoch)
            writer.add_scalar('val_accurate', val_accurate, epoch)
            writer.add_scalar('ap', ap, epoch)
            
            
        print(best_acc)
        print('Finished Training')
        return



def test():
    # ------------------------------------ step 5/5 : 预测 --------------------------------------------------
    val_data_root = "./data/val/"
    files = os.listdir(val_data_root)
    files.sort()
    results = list([] for i in range(len(files)))
    for i in range(k_fold):
        net =resnet50(num_classes=classes_num).to(device)

        # load model weights
        weights_path = f"./weights/resNet50_{i}.pth"
        assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
        net.load_state_dict(torch.load(weights_path, map_location=device))

        net.eval()
        with torch.no_grad():
            # predict class
            for index, file in tqdm(enumerate(files)):
                path = val_data_root + file
                # output = torch.squeeze(net(img.to(device))).cpu()
                # predict = torch.softmax(output, dim=0)
                # predict_cla = int(torch.argmax(predict).numpy())
                data = loadmat(path)
                ecg_data = data["ecgdata"][:,0:3000]
                ecg_data = savgol_filter(ecg_data, 51, 3)
                ecg_data = np.array([data for data in ecg_data[:,::5]])
                ecg_data = torch.tensor(ecg_data)
                # expand batch dimension
                img = torch.unsqueeze(ecg_data, dim=0)
                outputs = net(img.to(device)).cpu()
                predict_cla = torch.argmax(outputs).numpy()
                results[index].append(predict_cla)  
    
    labels = []
    output_files = []
    for index, result in enumerate(results):
        amxlabel = max(result, key=result.count)
        labels.append(amxlabel)
        file = files[index]
        output_files.append(os.path.basename(file))
    df = pd.DataFrame({'file':files,'label':labels})
    print(df)
    df.to_csv('pcb_result.csv', index=False)
if __name__ == "__main__":

    train()

    # test()
    
    


