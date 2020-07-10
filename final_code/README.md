# 配置环境

pytorch1.x

torchvision 0.4.0

（主要是在kaggle内核环境下训练和inference）

# 训练文件

## Baseline_train_v9.py

在官方提供的baseline基础上，使用K折交叉验证，并且将Adam初始学习率从0.005阶段性的调到了0.001，还加了图像变换增强，模型加上加上  dropout和bn层。

经过数据分析：

| label长度 | train集中的数量 |
| --------- | --------------- |
| 1         | 6554            |
| 2         | 22655           |
| 3         | 9382            |
| 4         | 1398            |
| 5         | 10              |
| 6         | 1               |

发现label长度为5、6的图像比较少，所以将5个定长字符串的识别改为4个定长字符串识别。

## Baseline_train_v12.py

在train_v9版本上进行了改进，加入了PseudoLabel的方式，也尝试了不同的图像增强方式。

PseudoLabel：用训练好的模型预测test_a中的标签，再用预测出的伪标签来重新训练模型，可以起到扩充数据集的效果。

## PseudoLabels_train.ipynb

在kaggle内核运行的notebook，使用PseudoLabel的方式进行训练，同时对参数（学习率、数据增强方式等）进行微调。同时由于样本的不均衡（字符串长度为2、3的图片明显多于长度为1、4的），对Loss采取加权措施，让模型能更好的去关注数据量较小的样本：

```
        loss = (1 - 0.17) * criterion(c0, target[:, 0]) + \
               (1 - 0.57) * criterion(c1, target[:, 1]) + \
               (1 - 0.23) * criterion(c2, target[:, 2]) + \
               (1 - 0.03) * criterion(c3, target[:, 3])
```

# 推理文件

## inference.ipynb

将训练效果差不多的几个模型集成在一起，也就是model_ensemble。然后预测时用这几个模型轮流预测，最后把他们的预测结果平均下来，作为最后的结果。

```
modelv30 = SVHN_Model1()
modelv30.load_state_dict(torch.load('../models/model_v30.pt'))
modelv23 = SVHN_Model1()
modelv23.load_state_dict(torch.load('../models/model_v23.pt'))
modelv31 = SVHN_Model1()
modelv31.load_state_dict(torch.load('../models/model_v31.pt'))
modelv29 = SVHN_Model1()
modelv29.load_state_dict(torch.load('../models/model_v29.pt'))
modelv28 = SVHN_Model1()
modelv28.load_state_dict(torch.load('../models/model_v28.pt'))

modelv27 = SVHN_Model1()
modelv27.load_state_dict(torch.load('../models/model_v27.pt'))
modelv20 = SVHN_Model1()
modelv20.load_state_dict(torch.load('../models/model_v20.pt'))
modelv19 = SVHN_Model1()
modelv19.load_state_dict(torch.load('../models/model_v19.pt'))

Models = [model,modelv30,modelv31,modelv23,modelv29,modelv28,modelv27,modelv20,modelv19]
```

# 训练记录

| 训练版本号 | 单一变量                                                     | 实验结果【准确率】 |
| ---------- | ------------------------------------------------------------ | ------------------ |
| v0         | Baseline，TTA=10                                             | 0.2895             |
| v1         | 数据集装载Resize大小改为一致的（64,128），卷积层在训练时，每一层的视野训练的区域是固定的，不能随意修改输入图像的大小 | 0.6716             |
| V2         | 把5个定长字符检测改为4个定长字符检测，因为Val集中5个以上的字符很少 | 0.3934             |
| V3         | 数据增强，在dataloader处改变，只修改train/val_dataloader中的transform.使用RandomCrop+RandomRotation(10),SVNH数据集有过拟合现象 | 0.7604             |
| V6         | 添加Dropout层，并且阶段性的调整Adam学习率                    | 0.8016             |
| V12        | 使用伪标签训练模型，有效地利用测试集数据                     | 0.8704             |
| V23        | 将Adam优化器换为SGD，并且去手动调节学习率                    | 0.9123             |
| V28        | 优化器为SGD，训练的batch_size由1000调整为64，数据增强中把crop、rotation去掉，模型的dropout改为0.2 | 0.921              |
| V31        | batch_size调整为32，dropout=0.1                              | 0.9231             |
| V32        | 最后使用模型增强策略，可以达到0.9237                         |                    |

# 联系方式：

nininnk@163.com