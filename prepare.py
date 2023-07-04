#you can run the folllow code to unzip the dataset or you can use other methods
#!unzip -oq data/data184197/tiny-ImageNet-200.zip -d ./

#run the follow codes to prepare data
import os
from random import shuffle
from shutil import copy
from tqdm import tqdm
labels = sorted([v for v in os.listdir('tiny-ImageNet-200/train') if v.startswith('n')])
splits = ['train', 'val', 'test']
for sp in splits:
    for lbl in labels:
        os.makedirs(os.path.join(sp, lbl))
base = 'tiny-ImageNet-200'
sp='train'
datas = []
for i, lbl in tqdm(enumerate(labels)):
    lbl_base = os.path.join(base, sp, lbl,'images')
    for img in os.listdir(lbl_base):
        if img.endswith('.JPEG'):
            src = os.path.join(lbl_base, img)
            dst = os.path.join(sp, lbl, img)
            copy(src, dst)

labels_s = []
labels_n = []
f = open('tiny-ImageNet-200/label_list.txt', 'r',encoding='utf-8')
line = f.readline()
while line:
    txt_data = line.split(' ')[0]
    txt_label = line.split(' ')[1].split('\n')[0]
    labels_s.append(txt_data)
    labels_n.append(txt_label)
    line = f.readline()

val_txt_tables = []
val_txt_labels = []
f = open('tiny-ImageNet-200/val_list.txt', 'r',encoding='utf-8')
line = f.readline() # 读取第一行
while line:
    txt_data = line.split(' ')[0].split('images/')[1]# 可将字符串变为元组
    txt_label = line.split(' ')[1].split('\n')[0]
    val_txt_tables.append(txt_data) # 列表增加
    val_txt_labels.append(labels_n[int(txt_label)])
    line = f.readline() # 读取下一行
print(val_txt_tables)
print(val_txt_labels)

base = 'tiny-ImageNet-200'
sp='val'
datas = []
lbl_base = os.path.join(base, sp,'images')
for img in os.listdir(lbl_base):
    if img.endswith('.JPEG'):
        for i,item in enumerate(val_txt_tables):
            if img==item:
                src = os.path.join(lbl_base, img)
                dst = os.path.join(sp, val_txt_labels[i], img)
                copy(src, dst)