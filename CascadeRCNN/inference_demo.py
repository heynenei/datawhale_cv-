#!/usr/bin/env python
# coding: utf-8


from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
import os
import pandas as pd
import json


config_file = './configs/cascade_rcnn/cascade_rcnn_r101_fpn_20e_coco.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file = './work_dirs/cascade_rcnn_r101_fpn_20e_coco/epoch_17.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')


d = {}

df = pd.DataFrame(columns=['file_name','file_code'])
image_path = "data/mchar_test_a/mchar_test_a/"
piclist = os.listdir(image_path)


piclist.sort()
index = 0
for pic_name in piclist:
    index += 1
    if index % 1000 == 0:
        print(f"{index}/40000")
    pic_path = os.path.join(image_path, pic_name)
    result = inference_detector(model, pic_path)
    boxes = []
    for i in range(10):
        for box in result[i]:
            copybox = box.tolist()
            #copybox.append(i)

            if i==9:
                copybox.append(0)
            else:
                copybox.append(i+1)

            if copybox[-2]>=0.4:
                boxes.append(copybox)

    boxes.sort(key=lambda x:x[0])

    d[pic_name] = []

    s = ""
    for b in boxes:
        s = s+str(b[-1])
        d[pic_name].append(b)

    if len(boxes)==0:
        s="1"
    df = df.append([{"file_name": pic_name, "file_code": s}], ignore_index=True)

with open("r101.json","w") as f:
    json.dump(d,f)

df.to_csv("r101.csv",index=False)