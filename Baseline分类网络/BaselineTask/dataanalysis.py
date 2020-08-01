import glob
import json

train_path = glob.glob('../input/train/*.png')
train_path.sort()
train_json = json.load(open('../input/train.json'))
train_label = [train_json[x]['label'] for x in train_json]

val_path = glob.glob('../input/val/*.png')
val_path.sort()
val_json = json.load(open('../input/val.json'))
val_label = [val_json[x]['label'] for x in val_json]

from tensorboardX import SummaryWriter
writer = SummaryWriter('log')
wordspace = None
dlen = dict()
for i in train_label+val_label:
    if str(len(i)) not in dlen.keys():
        dlen[str(len(i))] = 0
    else:
        dlen[str(len(i))] += 1

for (key,item) in dlen.items():
    writer.add_histogram('nums len',  item, key)
writer.close()