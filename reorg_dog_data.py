'''
此模块的作用是将原始的train,test文件夹数据重新整理为gluon读取的文件夹结构
'''

import math
import os
import shutil
from collections import Counter

data_dir = './data'

label_file = 'labels.csv'
train_dir = 'train'
test_dir = 'test'
input_dir = 'train_valid_test'
valid_ratio = 0.1

#将train,test文件夹整理为train_valid_test文件夹结构，便于gluon读取
def reorg_dog_data(data_dir, label_file, train_dir, test_dir, input_dir,
                   valid_ratio):
    # 读取训练数据标签labels.csv内容。
    with open(os.path.join(data_dir, label_file), 'r') as f:
        # 跳过文件头行（栏名称）。
        lines = f.readlines()[1:]  #从第二行开始读取所有行的内容
        tokens = [l.rstrip().split(',') for l in lines]  #每一行之间使用逗号分隔，表示一个键值对
        idx_label = dict(((idx, label) for idx, label in tokens))#将所有的键值对保存在一个字典中
    labels = set(idx_label.values()) #取出字典中的所有键值对的值

    #train和test文件夹中图像总数
    num_train = len(os.listdir(os.path.join(data_dir, train_dir)))

    # 训练集中数量最少一类的狗的数量。
    min_num_train_per_label = (
        Counter(idx_label.values()).most_common()[:-2:-1][0][1])
    # 验证集中每类狗的数量。按照最少狗的数量划分出一定比例的验证集
    num_valid_per_label = math.floor(min_num_train_per_label * valid_ratio)
    label_count = dict() #创建字典保存label总数量

    #没有对应文件夹则创建
    def mkdir_if_not_exist(path):
        if not os.path.exists(os.path.join(*path)):
            os.makedirs(os.path.join(*path))

    # 整理训练和验证集。
    for train_file in os.listdir(os.path.join(data_dir, train_dir)):#从训练集中取出每一张图片
        # 取出每张图片扩展名前的文件名，这个就是这个文件对应的id,然后根据这个id在labels.csv中查找对应的label
        idx = train_file.split('.')[0]
        label = idx_label[idx]  #根据id得到这个图片的标签label
        mkdir_if_not_exist([data_dir, input_dir, 'train_valid', label])#文件夹不存在，则创建
        # 将图片从train文件夹中复制train_valid_test/train_valid中
        shutil.copy(os.path.join(data_dir, train_dir, train_file),
                    os.path.join(data_dir, input_dir, 'train_valid', label))

        #放入到验证集train_valid_test/valid文件中
        if label not in label_count or label_count[label] < num_valid_per_label:
            mkdir_if_not_exist([data_dir, input_dir, 'valid', label])
            shutil.copy(os.path.join(data_dir, train_dir, train_file),
                        os.path.join(data_dir, input_dir, 'valid', label))
            label_count[label] = label_count.get(label, 0) + 1
        else:#放入到训练集train_valid_test/train文件夹中
            mkdir_if_not_exist([data_dir, input_dir, 'train', label])
            shutil.copy(os.path.join(data_dir, train_dir, train_file),
                        os.path.join(data_dir, input_dir, 'train', label))

    # 整理测试集。直接从test文件夹中copy每一张图片放在train_valid_test/test文件夹中
    mkdir_if_not_exist([data_dir, input_dir, 'test', 'unknown'])
    for test_file in os.listdir(os.path.join(data_dir, test_dir)):
        shutil.copy(os.path.join(data_dir, test_dir, test_file),
                    os.path.join(data_dir, input_dir, 'test', 'unknown'))

reorg_dog_data(data_dir, label_file, train_dir, test_dir, input_dir,
                   valid_ratio)