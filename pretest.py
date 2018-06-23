'''
此模块的主要作用：使用测试数据来验证模型效果，并将结果保存在kaggle.csv中
'''

import datetime
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from mxnet import autograd
from mxnet import gluon
from mxnet.gluon.data import vision
from mxnet import nd
import mxnet as mx
import pickle
from tqdm import tqdm
from model import Net,transform_test

batch_size = 32

data_dir = 'E:\\WorkSpace\\gluon-tutorials-zh_aa12356jm\\data\\kaggle_dog'
test_dir = 'test'
input_dir = 'train_valid_test'
valid_dir = 'valid'

netparams = 'train.params'
csvname = 'kaggle.csv'
ids_synsets_name = 'ids_synsets'

input_str = data_dir + '/' + input_dir + '/'

#读取数据label
f = open(ids_synsets_name, 'rb')
ids_synsets = pickle.load(f)
f.close()

#读取测试数据
test_ds = vision.ImageFolderDataset(input_str + test_dir, flag=1,
                                     transform =transform_test)
#读取验证数据
valid_ds = vision.ImageFolderDataset(input_str + valid_dir, flag=1,
                                     transform=transform_test)

loader = gluon.data.DataLoader

#使用gluon接口读取batch_size的数据
test_data = loader(test_ds, batch_size, shuffle=False, last_batch='keep')
valid_data = loader(valid_ds, batch_size, shuffle=True, last_batch='keep')

#计算预测和ground truth的loss
def get_loss(data, net, ctx):
    loss = 0.0
    for feas1, feas2, label in tqdm(data):
        label = label.as_in_context(ctx)
        feas1 = feas1.as_in_context(ctx)
        feas2 = feas2.as_in_context(ctx)
        output = net(feas1, feas2)
        cross_entropy = softmax_cross_entropy(output, label) #使用交叉熵损失softmax来计算loss
        loss += nd.mean(cross_entropy).asscalar()
    return loss / len(data)

#使用测试数据来测试模型效果，保存测试结果到文件kaggle.csv中
def SaveTest(test_data, net, ctx, name, ids, synsets):
    outputs = []
    #tqdm是为了显示读取进度条
    for data1, data2, label in tqdm(test_data):
        data1 = data1.as_in_context(ctx)
        data2 = data2.as_in_context(ctx)
        output = nd.softmax(net(data1,data2))
        outputs.extend(output.asnumpy())
    with open(name, 'w') as f:
        f.write('id,' + ','.join(synsets) + '\n')
        for i, output in zip(ids, outputs):
            f.write(i.split('.')[0] + ',' + ','.join(
                [str(num) for num in output]) + '\n')
    net.export('new_model')

net = Net(mx.gpu(), netparams).net #构造网络来进行测试
#net = get_net(netparams,mx.gpu())
net.hybridize()

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

print(get_loss(valid_data, net, mx.gpu())) #使用验证数据来验证网络模型

SaveTest(test_data, net, mx.gpu(), csvname, ids_synsets[0], ids_synsets[1])

