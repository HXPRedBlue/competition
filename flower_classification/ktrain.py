from datetime import datetime
from operator import le
import os
import json
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, average_precision_score
from torchvision import transforms, datasets
from torchvision.models import resnet18, mobilenetv3,resnet101
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

device = torch.device("cuda:0")
print("using {} device.".format(device))
batch_size = 16
epochs = 30
image_path = "./data/PCB_DATASET/classisfication_300_2"  # flower data set path
nw = 8  # number of workers
k_fold = 5
classes_num = 2



def train():

    # ------------------------------------ step 1/5 : 加载数据------------------------------------
    transform_image = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.RandomHorizontalFlip(0.5),
                                    transforms.RandomVerticalFlip(0.5),
                                    # transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
                                    # transforms.ColorJitter(brightness=0.5, hue=0.3),
                                    # transforms.RandomAdjustSharpness(sharpness_factor=2),
                                    # transforms.RandomAutocontrast(),
                                    transforms.RandomRotation(degrees=(0, 180)),
                                    # transforms.RandomPerspective(distortion_scale=0.6, p=0.5),
                                    # transforms.RandomAffine(degrees=(30, 70),translate=(0.1, 0.3), scale=(0.5, 0.75)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ])
    all_data = datasets.ImageFolder(root=image_path, transform=transform_image)
    data_list = all_data.class_to_idx
    cla_dict = dict((val, key) for key, val in data_list.items())
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)
    kfold = KFold(n_splits=k_fold, shuffle=True,random_state=233333)

    for fold_idx, (train_index, val_index) in enumerate(kfold.split(all_data)):
        print('*'*25,'第', fold_idx + 1,'折','*'*25)
        writer = SummaryWriter(f"./tensorboard/{datetime.now().strftime('%y%m%d_%H%M')}_{fold_idx}")   # 数据存放在这个文件夹
        train_data = Subset(all_data, train_index)
        val_data = Subset(all_data, val_index)

        train_loader = DataLoader(train_data,
                                            batch_size=batch_size, shuffle=True,
                                            num_workers=nw)

        validate_loader = DataLoader(val_data,
                                            batch_size=batch_size, shuffle=True,
                                            num_workers=nw)
        test_loader = validate_loader
        val_num = len(val_index)
        
        # ------------------------------------ step 2/5 : 初始化网络------------------------------------

        net = resnet18()
        model_weight_path = "./weights/resnet18_pre.pth"
        assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
        net.load_state_dict(torch.load(model_weight_path, map_location=device))
   

        # change fc layer structure
        in_channel = net.fc.in_features
        net.fc = nn.Linear(in_channel, classes_num)

        # summary(net,(3,224,224),batch_size=1,device="cpu")
        net.to(device)

        # ------------------------------------ step 3/5 : 定义损失函数和优化器 ------------------------------------
        # define loss function
        loss_function = nn.CrossEntropyLoss()

        # construct an optimizer
        # param = [p for p in net.parameters() if p.requires_grad]
        # optimizer = optim.Adam(net.parameters(), lr=0.0001)
        args = SimpleNamespace()
        args.lr = 0.0001
        args.weight_decay = 4e-5
        args.opt = 'adam' #'lookahead_adam' to use `lookahead`
        args.momentum = 0.9 
        optimizer = create_optimizer(args=args, model=net)
        # scheduler = CosineAnnealingLR(optimizer,T_max=10)

        args.sched = "cosine"
        args.epochs = epochs
        args.min_lr = 1e-5
        args.decay_rate = 30
        args.warmup_lr = 1e-6
        args.warmup_epochs = 5
        args.cooldown_epochs = 10
        scheduler, _ = create_scheduler(args, optimizer)

        # ------------------------------------ step 4/5 : 训练并保存模型 --------------------------------------------------

        best_acc = 0.0
        save_path = f'./weights/resNet18_{fold_idx}.pth'
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
                logits = net(images.to(device))
                loss = loss_function(logits, labels.to(device))
                train_pres.extend(logits.argmax(dim=1).cpu().numpy())
                train_trues.extend(labels.cpu().numpy())
                loss.backward()
                optimizer.step()
                scheduler.step(epoch)

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
                    outputs = net(val_data.to(device))
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
            f1 = f1_score(val_trues, val_preds)
            val_accurate = accuracy_score(val_trues, val_preds)
            train_loss = running_loss / train_steps
            val_loss = val_run_loss / val_steps
            ap = average_precision_score(val_trues, val_preds)
            print(f'[epoch {epoch + 1}] train_loss: {train_loss}  val_accuracy: {val_accurate} \
            precision: {precision}, recall:{recall}, f1:{f1}')
            

            if val_accurate > best_acc:
                best_acc = val_accurate
                torch.save(net.state_dict(), save_path)

            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('f1', f1, epoch)
            writer.add_scalar('precision', precision, epoch)
            writer.add_scalar('recall', recall, epoch)
            writer.add_scalar('val_accurate', val_accurate, epoch)
            writer.add_scalar('ap', ap, epoch)
            
            

            test_probs = torch.cat([torch.stack(batch) for batch in val_prebs])
            test_label = torch.cat(val_labels_pr)
            for i in range(classes_num):
                val_true = test_label == i
                val_pred = test_probs[:, i]
                writer.add_pr_curve(cla_dict[i], val_true, val_pred, epoch)

            writer.add_scalars("train_step", 
            {"train_loss": train_loss, "val_acc": val_accurate, "train_acc": train_acc}, epoch)

            writer.add_scalars("loss", 
            {"train_loss": train_loss, "val_loss": val_loss}, epoch)

            dummy_input = torch.rand(20, 3, 224, 224)  # 假设输入20张1*28*28的图片
            writer.add_graph(net, (dummy_input.to(device)))

        print(best_acc)
        return
        print('Finished Training')

if __name__ == "__main__":
    re_train = True
    best_nets = [0 for x in range(5)]
    transform_image = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    if re_train:
       train()
    
    # ------------------------------------ step 5/5 : 预测 --------------------------------------------------

    for i in range(1):
        model = resnet18(num_classes=classes_num).to(device)

        # load model weights
        weights_path = f"./weights/resNet18_{i}.pth"
        assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
        model.load_state_dict(torch.load(weights_path, map_location=device))
        best_nets[i] = model
    test_path = "./data/pcb_test/"
    test_data = datasets.ImageFolder(root=test_path, transform=transform_image)

    test_data_loader = DataLoader(test_data,
                                            batch_size=1, shuffle=False,
                                            num_workers=nw)

    results = list([] for i in range(len(test_data)))
    for net in best_nets:
        net.eval()
        with torch.no_grad():
            # predict class
            for index, data in enumerate(test_data_loader):
                img, lab = data
                # output = torch.squeeze(net(img.to(device))).cpu()
                # predict = torch.softmax(output, dim=0)
                # predict_cla = int(torch.argmax(predict).numpy())
                outputs = net(img.to(device)).cpu()
                predict_cla = torch.argmax(outputs).numpy()
                results[index].append(predict_cla)  
    
    labels = []
    files = []
    for index, result in enumerate(results):
        amxlabel = max(result, key=result.count)
        labels.append(amxlabel)
        file = test_data.imgs[index][0]
        files.append(os.path.basename(file))
    df = pd.DataFrame({'file':files,'label':labels})
    print(df)
    df.to_csv('pcb_result.csv', index=False)


