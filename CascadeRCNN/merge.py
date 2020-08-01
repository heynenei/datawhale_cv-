# coding:utf-8
import numpy as np
import json
import pandas as pd

jsonlist = ["r101.json",
            "x101_32.json",
            "x101_64.json"]

with open(jsonlist[0]) as f:
    load_dic = json.load(f)

for jsonpath in jsonlist[1:]:
    with open (jsonpath) as f:
        temp_dic = json.load(f)
        for k in load_dic.keys():
            load_dic[k] += temp_dic[k]


def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]  # bbox打分

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # 打分从大到小排列，取index
    order = scores.argsort()[::-1]
    # keep为最后保留的边框
    keep = []
    while order.size > 0:
        # order[0]是当前分数最大的窗口，肯定保留
        i = order[0]
        keep.append(i)
        # 计算窗口i与其他所有窗口的交叠部分的面积
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        # 交/并得到iou值
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # inds为所有与窗口i的iou值小于threshold值的窗口的index，其他窗口此次都被窗口i吸收
        inds = np.where(ovr <= thresh)[0]
        # order里面只保留与窗口i交叠面积小于threshold的那些窗口，由于ovr长度比order长度少1(不包含i)，所以inds+1对应到保留的窗口
        order = order[inds + 1]

    return dets[keep]


df = pd.DataFrame(columns=['file_name','file_code'])

for picname in  load_dic.keys():
    print(picname)
    boxes = load_dic[picname]
    if len(boxes)>1:
        n = np.array(boxes[0])
        for box in boxes[1:]:
            n = np.vstack((n, np.array(box)))
        keep = py_cpu_nms(n, 0.4)
        keep = keep.tolist()
        keep.sort(key=lambda x: x[0])

        s = ""
        for b in keep:
            if b[-2]>=0.4:
                s = s + str(int(b[-1]))

        df = df.append([{"file_name": picname, "file_code": s}], ignore_index=True)

    else:
        s = ""
        for b in boxes:
            if b[-2] >= 0.2:
                s = s + str(int(b[-1]))

        df = df.append([{"file_name": picname, "file_code": s}], ignore_index=True)

df.to_csv("submit.csv", index=False)