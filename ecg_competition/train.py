from datetime import datetime
import os
import sys

from torch.optim import lr_scheduler
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(BASE_DIR)
#添加系统环境变量
sys.path.append(BASE_DIR)
os.chdir(BASE_DIR)
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from tensorboardX import SummaryWriter

from models.resnet import resnet34, resnet50, resnet101
from models.ECGNet import ECGNet
from config import config
# from data.dataset import ECGDataset
from data.task2_dataset import ECGDataset
from torch.optim.lr_scheduler import CosineAnnealingLR


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))
batch_size = 16
epochs = 100

# 定义Summary_Writer
writer = SummaryWriter(f"./tensorboard/{datetime.now().strftime('%y%m%d_%H%M')}")   # 数据存放在这个文件夹

# ------------------------------------ step 1/5 : 加载数据------------------------------------
train_dataset = ECGDataset("/ai/223/competition/ecg/task2",True)
train_num = len(train_dataset)
nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
print('Using {} dataloader workers every process'.format(nw))
train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=batch_size, shuffle=True,
                                            num_workers=nw)

validate_dataset = ECGDataset("/ai/223/competition/ecg/task2/",False)
val_num = len(validate_dataset)
validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                batch_size=1, shuffle=False,
                                                num_workers=nw)

print("using {} images for training, {} images for validation.".format(train_num,
                                                                        val_num))

# ------------------------------------ step 2/5 : 初始化网络------------------------------------

net = ECGNet(input_channel=1,num_classes=12).to(device)

# ------------------------------------ step 3/5 : 定义损失函数和优化器 ------------------------------------
# define loss function
# loss_function = nn.CrossEntropyLoss()
loss_function = nn.BCEWithLogitsLoss()

# construct an optimizer

optimizer = optim.Adam(net.parameters(), lr=0.0001)
# scheduler = CosineAnnealingLR(optimizer,T_max=20)

# ------------------------------------ step 4/5 : 训练并保存模型 --------------------------------------------------

best_acc = 0.0
save_path = "./work_dir/ECG_lr0.0001.pth"

for epoch in range(epochs):
    # train
    net.train()
    running_loss = 0.0
    train_bar = tqdm(train_loader)
    for step, data in enumerate(train_bar):
        images, labels = data
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

    if final_score > best_acc:
        best_acc = final_score
        torch.save(net.state_dict(), save_path)

    print(f"f1_score:{f1s}, final_score:{final_score}")
    # writer.add_scalar('loss', running_loss, epoch)
    # writer.add_scalar('val_f1', val_f1, epoch)

print('Finished Training')


