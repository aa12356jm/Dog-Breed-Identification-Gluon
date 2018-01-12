'''
此模块的主要作用：对训练数据提取特征后的数据进行训练，将训练loss曲线和模型文件保存到本地
'''

import datetime
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from mxnet import autograd
from mxnet import gluon
from mxnet import nd
import mxnet as mx
import pickle
from model import Net

train_nd = nd.load('train.nd') #加载抽取特征后的训练数据train.nd
valid_nd = nd.load('valid.nd') #加载抽取特征后的验证数据valid.nd
input_nd = nd.load('input.nd') #加载抽取特征后的训练验证数据input.nd

f = open('ids_synsets', 'rb')
ids_synsets = pickle.load(f) #加载训练数据的label文件ids_synsets
f.close()

num_epochs = 100
batch_size = 128
learning_rate = 1e-4
weight_decay = 1e-4
pngname ='train.png'
modelparams ='train.params'

#使用gluon接口加载数据
train_data = gluon.data.DataLoader(gluon.data.ArrayDataset(train_nd[0], train_nd[1]),
                                   batch_size=batch_size, shuffle=True)
valid_data = gluon.data.DataLoader(gluon.data.ArrayDataset(valid_nd[0], valid_nd[1]),
                                   batch_size=batch_size, shuffle=True)
input_data = gluon.data.DataLoader(gluon.data.ArrayDataset(input_nd[0], input_nd[1]),
                                   batch_size=batch_size, shuffle=True)

#训练loss
def get_loss(data, net, ctx):
    loss = 0.0
    for feas, label in data:
        label = label.as_in_context(ctx)
        output = net(feas.as_in_context(ctx))
        cross_entropy = softmax_cross_entropy(output, label)
        loss += nd.mean(cross_entropy).asscalar()
    return loss / len(data)

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

#训练，输入：网络，训练数据，验证数据，epochs数，lr学习率，学习率降低weight decay
def train(net, train_data, valid_data, num_epochs, lr, wd, ctx):
    #使用gluon接口训练
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr, 'wd': wd})
    train_loss = []
    if valid_data is not None:
        test_loss = []

    prev_time = datetime.datetime.now()#记录每一个epoch的时间
    for epoch in range(num_epochs):
        _loss = 0.
        for data, label in train_data:
            label = label.as_in_context(ctx) #这个图片对应的ground trueth标签
            with autograd.record():
                output = net(data.as_in_context(ctx)) #预测值
                loss = softmax_cross_entropy(output, label) #和真实label对比，计算loss
            loss.backward() #反向传播梯度
            trainer.step(batch_size)
            _loss += nd.mean(loss).asscalar()
        cur_time = datetime.datetime.now() #记录一个epoch的时间
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        __loss = _loss/len(train_data)
        train_loss.append(__loss)

        #如果有验证数据，则给出训练loss和验证loss
        if valid_data is not None:  
            valid_loss = get_loss(valid_data, net, ctx)
            epoch_str = ("Epoch %d. Train loss: %f, Valid loss %f, "
                         % (epoch, __loss, valid_loss))
            test_loss.append(valid_loss)
        else:
            epoch_str = ("Epoch %d. Train loss: %f, "
                         % (epoch, __loss))
        #打印出一个epoch的时间和loss
        prev_time = cur_time
        print(epoch_str + time_str + ', lr ' + str(trainer.learning_rate))

    #训练完成则画出loss曲线，保存到本地train.png
    plt.plot(train_loss, 'r')
    if valid_data is not None: 
        plt.plot(test_loss, 'g')
    plt.legend(['Train_Loss', 'Test_Loss'], loc=2)

    #保存训练参模型文件
    plt.savefig(pngname, dpi=1000)
    net.collect_params().save(modelparams)

ctx = mx.gpu()
net = Net(ctx).output
net.hybridize()
#开始训练
train(net, train_data, valid_data, num_epochs, learning_rate, weight_decay, ctx)
