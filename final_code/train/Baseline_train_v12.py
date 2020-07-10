import os, sys, glob, shutil, json

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7,8,9'
import cv2

from PIL import Image
import numpy as np
from tqdm import tqdm, tqdm_notebook
import torch

torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
# 设置这个 flag 可以让内置的 cuDNN 的 auto-tuner 自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。
torch.backends.cudnn.benchmark = True
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset

use_cuda = True
my_test = False
# 这里代表pseudo label训练4次
epoch_num = 4

WEIGHT_PATH = '../models/model_v9.pt'
TEST_PATH = '../input/test_a/*.png'
INPUT_PATH = '../input'

from tensorboardX import SummaryWriter

writer = SummaryWriter('logv10')


# 定义读取数据集
class SVHNDataset(Dataset):
    def __init__(self, img_path, img_label, transform=None):
        self.img_path = img_path
        self.img_label = img_label
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):
        img = Image.open(self.img_path[index]).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        # change
        lbl = np.array(self.img_label[index][:4], dtype=np.int)
        # 如果label不足5个，扩充为5个字符
        lbl = list(lbl) + (4 - len(lbl)) * [10]
        return img, torch.from_numpy(np.array(lbl[:4]))

    def __len__(self):
        return len(self.img_path)


# 定义分类模型，使用ResNet18进行特征提取
class SVHN_Model1(nn.Module):
    def __init__(self):
        super(SVHN_Model1, self).__init__()

        model_conv = models.resnet18(pretrained=True)
        model_conv.avgpool = nn.AdaptiveAvgPool2d(1)
        model_conv = nn.Sequential(*list(model_conv.children())[:-1])
        self.cnn = model_conv
        self.bn = nn.BatchNorm2d(512)
        self.dp = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(512, 11)
        self.fc2 = nn.Linear(512, 11)
        self.fc3 = nn.Linear(512, 11)
        self.fc4 = nn.Linear(512, 11)
        # self.fc5 = nn.Linear(512, 11)

    def forward(self, img):
        feat = self.cnn(img)
        feat = self.bn(feat)
        feat = self.dp(feat)
        feat = self.relu(feat)
        # print(feat.shape)
        feat = feat.view(feat.shape[0], -1)
        c1 = self.fc1(feat)
        c2 = self.fc2(feat)
        c3 = self.fc3(feat)
        c4 = self.fc4(feat)
        # c5 = self.fc5(feat)
        return c1, c2, c3, c4


# 训练与验证
def train(train_loader, model, criterion, optimizer, epoch):
    # 切换模型为训练模式
    model.train()
    train_loss = []

    for i, (input, target) in enumerate(train_loader):
        # change
        target = target.long()
        if use_cuda:
            input = input.cuda()
            target = target.cuda()

        c0, c1, c2, c3 = model(input)
        loss = criterion(c0, target[:, 0]) + \
               criterion(c1, target[:, 1]) + \
               criterion(c2, target[:, 2]) + \
               criterion(c3, target[:, 3])
        # criterion(c4, target[:, 4])

        # loss /= 6
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
    return np.mean(train_loss)


def predict(test_loader, model, tta=10):
    model.eval()
    test_pred_tta = None

    # TTA 次数
    for _ in range(tta):
        test_pred = []

        with torch.no_grad():
            for i, (input, target) in tqdm(enumerate(test_loader)):
                target = target.long()
                if use_cuda:
                    input = input.cuda()

                c0, c1, c2, c3 = model(input)
                if use_cuda:
                    output = np.concatenate([
                        c0.data.cpu().numpy(),
                        c1.data.cpu().numpy(),
                        c2.data.cpu().numpy(),
                        c3.data.cpu().numpy()], axis=1)
                    # c4.data.cpu().numpy()], axis=1)
                else:
                    output = np.concatenate([
                        c0.data.numpy(),
                        c1.data.numpy(),
                        c2.data.numpy(),
                        c3.data.numpy()], axis=1)
                    # c4.data.numpy()], axis=1)

                test_pred.append(output)

        test_pred = np.vstack(test_pred)
        if test_pred_tta is None:
            test_pred_tta = test_pred
        else:
            test_pred_tta += test_pred

    return test_pred_tta


model = SVHN_Model1()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), 0.001)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

best_loss = 1000.0

if use_cuda:
    model = model.cuda()

# 预测并生成提交文件
test_path = glob.glob(TEST_PATH)
test_path.sort()
# test_json = json.load(open('../input/test_a.json'))
test_label = [[1]] * len(test_path)
# print(len(test_path), len(test_label))

if my_test:
    test_path = test_path[:10]
    test_label = test_label[:10]

test_loader = torch.utils.data.DataLoader(
    SVHNDataset(test_path, test_label,
                transforms.Compose([
                    transforms.Resize((64, 128)),
                    transforms.ColorJitter(0.3, 0.3, 0.2),
                    transforms.RandomRotation(10),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])),
    batch_size=1000,
    shuffle=False,
    num_workers=4,
)

# 加载保存的最优模型
model.load_state_dict(torch.load(WEIGHT_PATH))

# test_label_pred = []
# for x in test_predict_label:
#     test_label_pred.append(''.join(map(str, x[x!=10])))
# print(test_predict_label.shape)
############## PSEUDO #######################################

best_loss = 1000.0

train_path = glob.glob(f'{INPUT_PATH}/train/*.png')
val_path = glob.glob(f'{INPUT_PATH}/val/*.png')
image_path = train_path + val_path
image_path.sort()

train_json = json.load(open(f'{INPUT_PATH}/train.json'))
train_label = [train_json[x]['label'] for x in train_json]
val_json = json.load(open(f'{INPUT_PATH}/val.json'))
val_label = [val_json[x]['label'] for x in val_json]
image_label = train_label + val_label

train_loader = torch.utils.data.DataLoader(
    SVHNDataset(image_path, image_label,
                transforms.Compose([
                    transforms.Resize((64, 128)),
                    transforms.RandomCrop((50, 100)),
                    transforms.Resize((64, 128)),
                    # transforms.ColorJitter(0.3, 0.3, 0.2),
                    transforms.RandomRotation(10),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])),
    batch_size=1000,
    shuffle=True,
    num_workers=4,
)

for epoch in tqdm(range(epoch_num)):

    ####预测pseudo label
    test_predict_label = predict(test_loader, model, 1)
    test_predict_label = np.vstack([
        test_predict_label[:, :11].argmax(1),
        test_predict_label[:, 11:22].argmax(1),
        test_predict_label[:, 22:33].argmax(1),
        test_predict_label[:, 33:44].argmax(1),
        # test_predict_label[:, 44:55].argmax(1),
    ]).T

    test_label = test_predict_label
    # 创建新的pesudo label
    test_loader = torch.utils.data.DataLoader(
        SVHNDataset(test_path, test_label,
                    transforms.Compose([
                        # transforms.RandomCrop((60, 120)),
                        transforms.ColorJitter(0.3, 0.3, 0.2),
                        transforms.RandomRotation(10),
                        transforms.Resize((64, 128)),
                        transforms.RandomCrop((50, 100)),
                        transforms.Resize((64, 128)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])),
        batch_size=1000,
        shuffle=False,
        num_workers=4,
    )

    train_loss = train(train_loader, model, criterion, optimizer, epoch)
    test_loss = train(test_loader, model, criterion, optimizer, epoch)
    writer.add_scalar('Test/Loss', test_loss, epoch)
    print(r'Epoch: {0}, Train loss: {1} \t Val loss: {2}'.format(epoch, train_loss, test_loss))

    # 记录下验证集精度
    if test_loss < best_loss:
        best_loss = test_loss
        # print('Find better model in Epoch {0}, saving model.'.format(epoch))
        torch.save(model.state_dict(), 'model_v10.pt')

########################## END INFERENCE ##############################
test_loader = torch.utils.data.DataLoader(
    SVHNDataset(test_path, test_label,
                transforms.Compose([
                    #                     transforms.RandomCrop((60, 120)),
                    #                     transforms.ColorJitter(0.3, 0.3, 0.2),
                    #                     transforms.RandomRotation(10),
                    #                     transforms.Resize((64, 128)),
                    #                     transforms.RandomCrop((50, 100)),
                    transforms.Resize((64, 128)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])),
    batch_size=1000,
    shuffle=False,
    num_workers=4,
)

test_predict_label = predict(test_loader, model, 1)

test_predict_label = np.vstack([
    test_predict_label[:, :11].argmax(1),
    test_predict_label[:, 11:22].argmax(1),
    test_predict_label[:, 22:33].argmax(1),
    test_predict_label[:, 33:44].argmax(1),
    # test_predict_label[:, 44:55].argmax(1),
]).T

test_label_pred = []
for x in test_predict_label:
    test_label_pred.append(''.join(map(str, x[x != 10])))

import pandas as pd

df_submit = pd.read_csv(f'{INPUT_PATH}/test_A_sample_submit.csv')
df_submit['file_code'] = test_label_pred
df_submit.to_csv('submit_v12.csv', index=None)