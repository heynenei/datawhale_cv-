import os
import json
from PIL import Image

image_path2 = "data/mchar_val/mchar_val"
json_path2 = "data/mchar_val.json"

image_path1 = "data/mchar_train (复件)/mchar_train"
json_path1 = "data/mchar_train.json"

split_path = "split.json"

with open (split_path) as f:
    split_d = json.load(f)
smalllist = split_d['trainval']
largelist = split_d['valval']

d1 = {}
d1['info'] = {}
d1['licenses'] = []
d1['images'] = []
d1['annotations'] = []
d1['categories'] = []

d2 = {}
d2['info'] = {}
d2['licenses'] = []
d2['images'] = []
d2['annotations'] = []
d2['categories'] = []

# categories
for i in range(1,10):
    temp = {}
    temp['supercategory'] = str(i)
    temp['id'] = i
    temp['name'] = str(i)
    d1['categories'].append(temp)
    d2['categories'].append(temp)
temp = {}
temp['supercategory'] = str(0)
temp['id'] = 10
temp['name'] = str(0)
d1['categories'].append(temp)
d2['categories'].append(temp)

"""
处理train
"""
# images
piclist = os.listdir(image_path1)
for pic_name in piclist:
    pic_path = os.path.join(image_path1, pic_name)
    w,h = Image.open(pic_path).size
    temp = {}
    if pic_name == '000000.png':
        temp['id'] = 0
    else:
        temp['id'] = int(str(pic_name.split('.')[0]))
    temp['file_name'] = pic_name
    temp['width'] = w
    temp['height'] = h
    if pic_name in split_d['trainval']:
        d2['images'].append(temp)
    else:
        d1['images'].append(temp)

index = 0
#annotations
with open (json_path1) as f:
    load_dic = json.load(f)
    for pic_name in load_dic.keys():
        heightlist = load_dic[pic_name]['height']
        labellist = load_dic[pic_name]['label']
        leftlist = load_dic[pic_name]['left']
        toplist = load_dic[pic_name]['top']
        widthlist = load_dic[pic_name]['width']

        n = len(labellist)
        for i in range(n):
            box = [leftlist[i],toplist[i],widthlist[i],heightlist[i]]
            temp = {}
            if pic_name == '000000.png':
                temp['image_id'] = 0
            else:
                temp['image_id'] = int(str(pic_name.split('.')[0]))
            temp['segmentation'] = []
            temp['iscrowd'] = 0
            if labellist[i] == 0:
                temp['category_id'] = 10
            else:
                temp['category_id'] = labellist[i]
            temp['id'] = index
            index += 1
            temp['bbox'] = box
            temp['area'] = widthlist[i]*heightlist[i]

            if pic_name in split_d['trainval']:
                d2['annotations'].append(temp)
            else:
                d1['annotations'].append(temp)

"""
处理val
"""
# images
piclist = os.listdir(image_path2)
for pic_name in piclist:
    pic_path = os.path.join(image_path2, pic_name)
    w,h = Image.open(pic_path).size
    temp = {}
    temp['id'] = int(str(pic_name.split('.')[0]))
    temp['file_name'] = pic_name
    temp['width'] = w
    temp['height'] = h
    if pic_name in split_d['valval']:
        d2['images'].append(temp)
    else:
        d1['images'].append(temp)

#annotations
with open (json_path2) as f:
    load_dic = json.load(f)
    for pic_name in load_dic.keys():
        heightlist = load_dic[pic_name]['height']
        labellist = load_dic[pic_name]['label']
        leftlist = load_dic[pic_name]['left']
        toplist = load_dic[pic_name]['top']
        widthlist = load_dic[pic_name]['width']

        n = len(labellist)
        for i in range(n):
            box = [leftlist[i],toplist[i],widthlist[i],heightlist[i]]
            temp = {}
            indexno = pic_name.split('.')[0]
            indexno = '03' + indexno[2:]
            newpic_name = f"{indexno}.png"
            temp['image_id'] = int(str(newpic_name.split('.')[0]))
            temp['segmentation'] = []
            temp['iscrowd'] = 0
            if labellist[i] == 0:
                temp['category_id'] = 10
            else:
                temp['category_id'] = labellist[i]
            temp['id'] = index
            index += 1
            temp['bbox'] = box
            temp['area'] = widthlist[i]*heightlist[i]

            if newpic_name in split_d['valval']:
                d2['annotations'].append(temp)
            else:
                d1['annotations'].append(temp)

print("train:",len(d1['images']))
print("val:",len(d2['images']))

with open("data/newtrain.json","w") as f:
    json.dump(d1,f)
with open("data/newval.json","w") as f:
    json.dump(d2,f)