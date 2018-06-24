'''
此模块的主要作用：对训练数据提取特征后的数据进行训练，将训练loss曲线和模型文件保存到本地
迁移学习：冻结预训练模型的features，只训练output输出层
'''

import datetime
import matplotlib
import argparse
matplotlib.use('agg')
import matplotlib.pyplot as plt
from mxnet import autograd,gluon,nd
import mxnet as mx
import numpy as np
import pickle
from mxboard import SummaryWriter
from model_new import Net,Net_updateAllFea

#设置模型参数
parser = argparse.ArgumentParser(description='MXNet Gluon train kaggle_dog example')
parser.add_argument('--batch-size', type=int, default=128,
                    help='batch size for training and testing (default: 100)')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate,use adam :1e-4,  use SGD:0.01')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='weight_decay (default: 1e-4)')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='Train on GPU with CUDA')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
#指定每次隔多少个epoch学习率衰减一次
parser.add_argument('--lr-steps', default='30,60,90', type=str,
                    help='list of learning rate decay epochs as in str')
#学习率衰减，每次衰减为原来的系数
parser.add_argument('--lr-factor', default=0.5, type=float,
                    help='learning rate decay ratio')
opt = parser.parse_args()


#加载抽取特征后的训练数据，验证数据
train_nd = nd.load('train.nd')
valid_nd = nd.load('valid.nd')
#input_nd = nd.load('input.nd')

#加载训练数据的label文件ids_synsets
f = open('ids_synsets', 'rb')
ids_synsets = pickle.load(f)
f.close()

pngname ='train.png'  #画出loss曲线并保存到此图像
modelparams ='train.params'  #保存的模型名

#使用gluon接口加载数据
train_data = gluon.data.DataLoader(gluon.data.ArrayDataset(train_nd[0], train_nd[1]),
                                   batch_size=opt.batch_size, shuffle=True)
valid_data = gluon.data.DataLoader(gluon.data.ArrayDataset(valid_nd[0], valid_nd[1]),
                                   batch_size=opt.batch_size, shuffle=True)
#input_data = gluon.data.DataLoader(gluon.data.ArrayDataset(input_nd[0], input_nd[1]),batch_size=opt.batch_size, shuffle=True)


#达到多少epochs学习率衰减
def update_learning_rate(lr, trainer, epoch, ratio, steps):
    """Set the learning rate to the initial value decayed by ratio every N epochs."""
    new_lr = lr * (ratio ** int(np.sum(np.array(steps) < epoch)))
    trainer.set_learning_rate(new_lr)
    return trainer

#验证数据精度
def test(valid_data, ctx, net):
    metric = mx.metric.Accuracy()
    for data, label in valid_data:
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        output = net(data)
        metric.update([label], [output])
    return metric.get()

#验证集loss
def get_loss(data, net, ctx):
    loss = 0.0
    for feas, label in data:
        label = label.as_in_context(ctx)
        output = net(feas.as_in_context(ctx))
        cross_entropy = softmax_cross_entropy(output, label)
        loss += nd.mean(cross_entropy).asscalar()
    return loss / len(data)


softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
lr_steps = [int(x) for x in opt.lr_steps.split(',') if x.strip()] #学习率衰减

#训练，输入：网络，训练数据，验证数据，epochs数，lr学习率，学习率降低weight decay
def train(net, train_data, valid_data, num_epochs, lr, wd, momentum, ctx):
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr, 'wd': wd})
    #trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr, 'momentum': momentum})
    metric = mx.metric.Accuracy()#用来记录训练过程中的参数

    #自己画出训练曲线
    train_loss = []
    if valid_data is not None:
        test_loss = []

    # collect parameter names for logging the gradients of parameters in each epoch
    params = net.collect_params()
    param_names = params.keys()
    # define a summary writer that logs data and flushes to the file every 5 seconds
    sw = SummaryWriter(logdir='./logs', flush_secs=2)
    global_step = 0
    prev_time = datetime.datetime.now()#记录每一个epoch的时间
    for epoch in range(num_epochs):
        trainer = update_learning_rate(lr, trainer, epoch, opt.lr_factor, lr_steps) #学习率衰减策略
        _loss = 0.
        metric.reset()
        for i, (data, label) in enumerate(train_data):
            label = label.as_in_context(ctx) #标签和数据，放在gpu上
            data = data.as_in_context(ctx)
            #开始记录计算图
            with autograd.record():
                output = net(data) #预测值
                loss = softmax_cross_entropy(output, label) #和真实label对比，计算loss
            sw.add_scalar(tag='cross_entropy', value=loss.mean().asscalar(), global_step=global_step)
            global_step += 1
            loss.backward() #反向传播梯度
            trainer.step(opt.batch_size)
            metric.update([label], [output])
            if i % 100 == 0 and i > 0:
                name, acc = metric.get()
                print('[Epoch %d Batch %d] Training: %s=%f' % (epoch, i, name, acc))
            if i == 0:
                pass
                #sw.add_image('kaggleDog_first_minibatch', data.reshape((opt.batch_size, 2048, 1, 1)), epoch)
            _loss += nd.mean(loss).asscalar()

        ####################使用MXboard画出训练曲线###################
        if epoch == 0:
            sw.add_graph(net)
        grads = [i.grad() for i in net.collect_params().values()]
        assert len(grads) == len(param_names)
        # logging the gradients of parameters for checking convergence
        for i, name in enumerate(param_names):
            sw.add_histogram(tag=name, values=grads[i], global_step=epoch, bins=1000)

        #训练精度
        name, acc = metric.get()
        print('[Epoch %d] Training: %s=%f' % (epoch, name, acc))
        # logging training accuracy
        sw.add_scalar(tag='train_acc', value=acc, global_step=epoch)

        #得到测试精度
        name, val_acc = test(valid_data, ctx, net)
        # logging the validation accuracy
        print('[Epoch %d] Validation: %s=%f' % (epoch, name, val_acc))
        sw.add_scalar(tag='valid_acc', value=val_acc, global_step=epoch)
        ####################使用MXboard画出训练曲线###################


        cur_time = datetime.datetime.now()
        #转换为时分秒格式
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
    sw.close()

    #训练完成则画出loss曲线，保存到本地train.png
    plt.plot(train_loss, 'r')
    if valid_data is not None: 
        plt.plot(test_loss, 'g')
    plt.legend(['Train_Loss', 'Test_Loss'], loc=2)

    #保存训练参模型文件
    plt.savefig(pngname, dpi=1000)
    net.collect_params().save(modelparams)
    net.export('model')

if __name__ == '__main__':
    if opt.cuda:
        ctx = mx.gpu(0)
    else:
        ctx = mx.cpu()
    net = Net(ctx).output #迁移学习，冻结预训练模型的feature,只训练output部分的参数
    net.hybridize()  #net.hybridize()是gluon特有的，可以将动态图转换成静态图加快训练速度
    train(net, train_data, valid_data, opt.epochs, opt.lr, opt.weight_decay,opt.momentum, ctx)#开始训练
    print('train finished')
