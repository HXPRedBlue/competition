from datetime import datetime
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(BASE_DIR)
#添加系统环境变量
sys.path.append(BASE_DIR)
os.chdir(BASE_DIR)
import json
import numpy as np
from sklearn.metrics import f1_score
from torchvision import transforms, datasets
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split

from tqdm import tqdm
from tensorboardX import SummaryWriter


from model import resnet34
np.random.seed(15)
torch.manual_seed(15)
torch.cuda.manual_seed_all(15)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))
batch_size = 16
epochs = 10

# 定义Summary_Writer
writer = SummaryWriter(f"./tensorboard/{datetime.now().strftime('%y%m%d_%H%M')}")   # 数据存放在这个文件夹

# ------------------------------------ step 1.1 : 加载数据集,并进行归一化------------------------------------
transform_image = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

image_path = "./data/flower_photos/"  # flower data set path
assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

dataset = datasets.ImageFolder(root=image_path, transform=transform_image, generator=torch.Generator().manual_seed(0))
flower_list = dataset.class_to_idx
cla_dict = dict((val, key) for key, val in flower_list.items())
json_str = json.dumps(cla_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, validate_dataset = random_split(dataset, [train_size, test_size])

# train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
#                                         transform=data_transform["train"])

train_num = len(train_dataset)

batch_size = 16
nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
print('Using {} dataloader workers every process'.format(nw))

train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=batch_size, shuffle=True,
                                            num_workers=nw)

# validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
#                                         transform=data_transform["val"])
val_num = len(validate_dataset)
validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                batch_size=batch_size, shuffle=False,
                                                num_workers=nw)

print("using {} images for training, {} images for validation.".format(train_num,
                                                                        val_num))

# ------------------------------------ step 2/5 : 初始化网络------------------------------------

net = resnet34()
# load pretrain weights
# download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
model_weight_path = "./resnet34_pre.pth"
assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
net.load_state_dict(torch.load(model_weight_path, map_location=device))
# for param in net.parameters():
#     param.requires_grad = False

# change fc layer structure
in_channel = net.fc.in_features
net.fc = nn.Linear(in_channel, 5)
net.to(device)

# ------------------------------------ step 3/5 : 定义损失函数和优化器 ------------------------------------
# define loss function
loss_function = nn.CrossEntropyLoss()

# construct an optimizer
params = [p for p in net.parameters() if p.requires_grad]
optimizer = optim.Adam(params, lr=0.0001)

# ------------------------------------ step 4/5 : 训练并保存模型 --------------------------------------------------

best_acc = 0.0
save_path = './resNet34.pth'
train_steps = len(train_loader)
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
    acc = 0.0  # accumulate accurate number / epoch
    with torch.no_grad():
        val_bar = tqdm(validate_loader)
        for val_data in val_bar:
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            # loss = loss_function(outputs, test_labels)
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

            val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                        epochs)

    val_accurate = acc / val_num
    print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
            (epoch + 1, running_loss / train_steps, val_accurate))

    if val_accurate > best_acc:
        best_acc = val_accurate
        torch.save(net.state_dict(), save_path)

print('Finished Training')

