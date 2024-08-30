import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import warnings
from torch.autograd import Variable
from torchvision import transforms, datasets
from tqdm import tqdm
from ResNet50 import resnet50
from MultiscaleFusionNet import MultiscaleFusionNet
from AdaptiveModelFusionBlock import AdaptiveModelFusionBlock

warnings.filterwarnings("ignore")

class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=3, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:

            self.alpha = Variable(torch.tensor([0.1, 0.1, 0.2, 0.05]))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = inputs.softmax(dim=1)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        if inputs.is_cuda and not self.alpha.is_cuda:
             self.alpha = self.alpha.cuda()

        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


def main():

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),  # 随机长宽比裁剪为224×224
                                     transforms.RandomHorizontalFlip(),  # 依概率p=0.5水平翻转
                                     transforms.ToTensor(),  # 转化为张量
                                     transforms.Normalize([0.48748466, 0.4877048, 0.48798144], [0.22610623, 0.22610672, 0.22624451])]),   # 将图像标准化处理
        "val": transforms.Compose([transforms.Resize(256),   # 重置图像分辨率
                                   transforms.CenterCrop(224),  # 依据给定的size从中心裁剪
                                   transforms.ToTensor(),  # 转化为张量
                                   transforms.Normalize([0.48755363, 0.4894032, 0.49044675], [0.23555388, 0.23538584, 0.23563574])])}  # 将图像标准化处理

    image_path = ""
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    disease_list = train_dataset.class_to_idx
    cla_dict = dict((test, key) for key, test in disease_list.items())

    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = ""
    nw = ""
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw, pin_memory=True)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    net1 = MultiscaleFusionNet(num_classes=4)
    net2 = resnet50()
    net3 = AdaptiveModelFusionBlock()

    net1.to(device)
    net2.to(device)
    net3.to(device)

    loss_function = FocalLoss(class_num=4)
    loss_function2 = FocalLoss(class_num=4)
    loss_function3 = FocalLoss(class_num=4)

    param = [p for p in net1.parameters() if p.requires_grad]
    param2 = [p for p in net2.parameters() if p.requires_grad]
    param3 = [p for p in net3.parameters() if p.requires_grad]

    optimizer = optim.Adam(param, lr=0.0001)
    optimizer2 = optim.Adam(param2, lr=0.0001)
    optimizer3 = optim.Adam(param3, lr=0.0001)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=90, gamma=0.1)
    scheduler2 = lr_scheduler.StepLR(optimizer2, step_size=90, gamma=0.1)

    epochs = 200

    for epoch in range(epochs):
        # train
        net1.train()
        net2.train()
        net3.train()
        running_loss = 0.0
        running_loss2 = 0.0
        running_loss3 = 0.0

        train_bar = tqdm(train_loader, file=sys.stdout)

        for step, data in enumerate(train_bar):
            images, labels = data
            images, labels = Variable(images.cuda()), Variable(labels.cuda())
            optimizer.zero_grad()
            optimizer2.zero_grad()
            optimizer3.zero_grad()
            logits = net1(images)
            logits2 = net2(images)
            logits3 = net3(x=logits, y=logits2)

            loss = loss_function(logits, labels).to(device)
            loss_2 = loss_function2(logits2, labels).to(device)
            loss_3 = loss_function3(logits3, labels).to(device)

            loss.backward(retain_graph=True)
            loss_2.backward(retain_graph=True)
            loss_3.backward(retain_graph=True)

            optimizer.step()
            optimizer2.step()
            optimizer3.step()

            running_loss += loss.item()
            running_loss2 += loss_2.item()
            running_loss3 += loss_3.item()

            train_bar.desc = "train epoch[{}/{}] loss_3:{:.3f}".format(epoch + 1, epochs, loss_3)

        scheduler.step()
        scheduler2.step()

        # validate

        net1.eval()
        net2.eval()
        net3.eval()
        acc = 0.0
        acc2 = 0.0
        acc3 = 0.0
        val_loss = 0.0
        val_loss2 = 0.0
        val_loss3 = 0.0
        val_true = []

        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                val_images, val_labels = Variable(val_images.cuda()), Variable(val_labels.cuda())
                val_truel = val_labels.to(device).cpu().numpy()

                val_true.extend(val_truel)

                outputs = net1(val_images.to(device))
                outputs2 = net2(val_images.to(device))
                outputs3 = net3(x=outputs, y=outputs2)

                loss = loss_function(outputs, val_labels.to(device))
                loss_2 = loss_function2(outputs2, val_labels.to(device))
                loss_3 = loss_function3(outputs3, val_labels.to(device))

                predict_y = torch.max(outputs, dim=1)[1]

                predict_y2 = torch.max(outputs2, dim=1)[1]

                predict_y3 = torch.max(outputs3, dim=1)[1]

                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                acc2 += torch.eq(predict_y2, val_labels.to(device)).sum().item()
                acc3 += torch.eq(predict_y3, val_labels.to(device)).sum().item()

                val_loss += loss.item()
                val_loss2 += loss_2.item()
                val_loss3 += loss_3.item()

                val_bar.desc = "valid epoch[{}/{}] val_loss3:{:.3f}".format(epoch + 1, epochs, val_loss3)

    print('Finished Training')
