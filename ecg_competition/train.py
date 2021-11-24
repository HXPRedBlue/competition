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

from models.resnet import resnet34, resnet50, resnet101
from config import config
from dataset import ECGDataset


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))
batch_size = 16
epochs = 100

# 定义Summary_Writer
writer = SummaryWriter('./tensorboard')   # 数据存放在这个文件夹

# ------------------------------------ step 1/5 : 加载数据------------------------------------
# @TODO 数据标准化
train_dataset = ECGDataset(config.save_data,True)
train_num = len(train_dataset)
nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
print('Using {} dataloader workers every process'.format(nw))
train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=batch_size, shuffle=True,
                                            num_workers=nw)

validate_dataset = ECGDataset(config.save_data,False)
val_num = len(validate_dataset)
validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                batch_size=batch_size, shuffle=False,
                                                num_workers=nw)

print("using {} images for training, {} images for validation.".format(train_num,
                                                                        val_num))

# ------------------------------------ step 2/5 : 初始化网络------------------------------------

net = resnet50(num_classes=2).to(device)

# ------------------------------------ step 3/5 : 定义损失函数和优化器 ------------------------------------
# define loss function
loss_function = nn.CrossEntropyLoss()

# construct an optimizer
params = [p for p in net.parameters() if p.requires_grad]
optimizer = optim.Adam(params, lr=0.0001)

# ------------------------------------ step 4/5 : 训练并保存模型 --------------------------------------------------

best_acc = 0.0
save_path = "./work_dir/resnet50_lr0.0001.pth"

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
            prob = prob.cpu().numpy() #先把prob转到CPU上，然后再转成numpy，如果本身在CPU上训练的话就不用先转成CPU了
            prob_all.extend(np.argmax(prob,axis=1)) 
            label_all.extend(label)
    val_f1 = f1_score(label_all,prob_all)
    print("val F1-Score:{:.4f}".format(val_f1))

    net.eval()
    with torch.no_grad():

        prob_all = []
        label_all = []
        val_bar = tqdm(train_loader)
        for data,label in val_bar:
            prob = net(data.to(device)) #表示模型的预测输出
            prob = prob.cpu().numpy() #先把prob转到CPU上，然后再转成numpy，如果本身在CPU上训练的话就不用先转成CPU了
            prob_all.extend(np.argmax(prob,axis=1)) 
            label_all.extend(label)
    train_f1 = f1_score(label_all,prob_all)
    print("train F1-Score:{:.4f}".format(train_f1))

    if val_f1 > best_acc:
        best_acc = val_f1
        torch.save(net.state_dict(), save_path)

    writer.add_scalar('loss', running_loss, epoch)
    writer.add_scalar('train_f1', train_f1, epoch)
    writer.add_scalar('val_f1', val_f1, epoch)

print('Finished Training')


