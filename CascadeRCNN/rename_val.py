import os
path = 'data/mchar_val/mchar_val'

files = os.listdir(path)

for file in files:
    old = os.path.join(path,file)
    index = file.split('.')[0]
    index = '03' + index[2:]
    newpic = f"{index}.png"
    new = os.path.join(path,newpic)
    os.rename(old,new)