'''
此模块的主要作用是通过网络net将训练集，验证集，训练验证集的所有数据分别抽取特征
保存在对应的train.nd, test.nd, input.nd
'''

from mxnet import gluon
from mxnet import nd
from mxnet.gluon.data import vision
import numpy as np
import mxnet as mx
import pickle
from tqdm import tqdm
import os
from model import Net, transform_train, transform_test

data_dir = './data'
train_dir = 'train'
test_dir = 'test'
valid_dir = 'valid'
input_dir = 'train_valid_test'

train_valid_dir = 'train_valid'

input_str = data_dir + '/' + input_dir + '/'


batch_size = 32

#训练数据，从train_valid_test/train文件夹中加载，使用函数transform_train进行数据转换和增强
#transform_train函数生成2种图像，分别给两个网络进行训练
# flag为0,则表示加载为灰度图，为1则为三通道彩色图
train_ds = vision.ImageFolderDataset(input_str + train_dir, flag=1,
                                     transform=transform_train)
#验证数据，从train_valid_test/valid文件夹中加载，使用函数transform_test进行数据转换和增强
valid_ds = vision.ImageFolderDataset(input_str + valid_dir, flag=1,
                                     transform=transform_test)
#训练验证数据，从train_valid_test/train_valid文件夹中加载，使用函数transform_train进行数据转换和增强
train_valid_ds = vision.ImageFolderDataset(input_str + train_valid_dir,
                                           flag=1, transform=transform_train)

loader = gluon.data.DataLoader  #gluon提供的数据加载函数，Loads data from a dataset and returns mini-batches of data
#从数据集中每次加载指定batchsize大小的数据
train_data = loader(train_ds, batch_size, shuffle=True, last_batch='keep')
valid_data = loader(valid_ds, batch_size, shuffle=True, last_batch='keep')
train_valid_data = loader(train_valid_ds, batch_size, shuffle=True,last_batch='keep')

#net = get_features(mx.gpu())
net = Net(mx.gpu()).features #得到网络Net的features，具体可查看Net模块，查看网络如何构造
net.hybridize()

#将输入的data通过net提取特征，保存到name文件中
def SaveNd(data, net, name):
    x = []
    y = []
    print('提取特征 %s' % name)
    #tqdm就是为了显示数据加载的进度条，没有其它用途
    for fear1, fear2, label in tqdm(data):
        fear1 = fear1.as_in_context(mx.gpu()) #取出transform_train函数第一种生成的数据
        fear2 = fear2.as_in_context(mx.gpu()) #取出transform_train函数第二种生成的数据
        #将两种数据分别给两个网络进行训练
        out = net(fear1, fear2).as_in_context(mx.cpu())
        x.append(out)
        y.append(label)
    x = nd.concat(*x, dim=0)#按行拉伸为一个list
    y = nd.concat(*y, dim=0)
    print('保存特征 %s' % name)
    nd.save(name, [x, y]) #将抽取到的特征和对应的label存到name文件中

#通过net网络抽取所有训练集的特征，保存到train.nd文件中
SaveNd(train_data, net, 'train.nd')
#通过net网络抽取所有验证集的特征，保存到valid.nd文件中
SaveNd(valid_data, net, 'valid.nd')
#通过net网络将所有的训练验证数据抽取特征，保存到input.nd，用来进行训练
SaveNd(train_valid_data, net, 'input.nd')

#对测试数据重新排序
ids = ids = sorted(os.listdir(os.path.join(data_dir, input_dir, 'test/unknown')))
synsets = train_valid_ds.synsets #labels

#将labels写入到ids_synsets文件中
f = open('ids_synsets','wb')
pickle.dump([ids, synsets], f)
f.close()
