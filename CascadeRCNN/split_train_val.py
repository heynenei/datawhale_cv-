import os
import json
import random

image_path1 = "data/mchar_train (复件)/mchar_train"
image_path2 = "data/mchar_val/mchar_val"

d={}
d['trainval'] = []
d['valval'] = []

piclist = os.listdir(image_path1)
for pic in piclist:
    r=random.random()
    if r<=0.1:
        d['trainval'].append(pic)

piclist = os.listdir(image_path2)
for pic in piclist:
    r = random.random()
    if r <= 0.1:
        index = pic.split('.')[0]
        index = '03' + index[2:]
        newpic = f"{index}.png"
        d['valval'].append(newpic)

print('trainval:',len(d['trainval']))
print('valval:',len(d['valval']))

with open('split.json','w') as f:
    json.dump(d,f)