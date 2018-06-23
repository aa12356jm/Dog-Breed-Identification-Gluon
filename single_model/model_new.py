'''
此模块的主要作用：构造单网络模型，对训练数据和测试数据进行扩增和转换
优点：便于c++部署
'''

from mxnet import init
from mxnet.gluon.model_zoo import vision
from mxnet.gluon import nn
from mxnet import image
import numpy as np
from mxnet import nd
import mxnet as mx

#根据输入的features和output网络，构造新网络
class  OneNet(nn.HybridBlock):
    def __init__(self, features, output, **kwargs):
        super(OneNet, self).__init__(**kwargs)
        self.features = features
        self.output = output

    #进行前向计算得到输出结果
    def hybrid_forward(self, F, x1):
        return self.output(self.features(x1))

#对训练数据进行转换，如数据增强等等，将每个数据都处理2次，结果返回对应处理后的结果和标签
def transform_train(data, label):

    #将图像调整为不同的大小，分别进行数据扩增,因为是使用两个模型进行融合，所以也使用两类数据分别给两个网络进行训练
    im = image.imresize(data.astype('float32') / 255, 299, 299)

    auglist = image.CreateAugmenter(data_shape=(3, 299, 299), resize=0,
                        rand_crop=False, rand_resize=False, rand_mirror=True,
                        mean=np.array([0.485, 0.456, 0.406]), std=np.array([0.229, 0.224, 0.225]),
                        brightness=0, contrast=0, saturation=0, hue=0,
                        pca_noise=0, rand_gray=0, inter_method=2)

    for aug in auglist:
        im = aug(im)
    # 将数据格式从"高*宽*通道"改为"通道*高*宽"。
    im = nd.transpose(im, (2, 0, 1))
    #返回给两个数据，分别给两个网络进行训练
    return (im, nd.array([label]).asscalar().astype('float32'))

#对测试数据进行数据扩增，同上
def transform_test(data, label):
    im = image.imresize(data.astype('float32') / 255, 299, 299)
    auglist = image.CreateAugmenter(data_shape=(3, 299, 299),
                        mean=np.array([0.485, 0.456, 0.406]),
                        std=np.array([0.229, 0.224, 0.225]))
    for aug in auglist:
        im = aug(im)
    # 将数据格式从"高*宽*通道"改为"通道*高*宽"。
    im = nd.transpose(im, (2, 0, 1))
    return (im, nd.array([label]).asscalar().astype('float32'))



#构建完整的网络模型
class Net():
    def __init__(self, ctx, nameparams=None):
        inception = vision.inception_v3(pretrained=True, ctx=ctx).features #提取预训练模型inception_v3的features
        self.features = inception
        self.output = self.__get_output(ctx, nameparams) #只有输出层的网络
        self.net = OneNet(self.features, self.output) #将含有features的网络和输出层的网络构造为新的网络

    #构造一个输出层的网络，然后返回这个网络
    def __get_output(self, ctx, ParamsName=None):
        net = nn.HybridSequential("output")
        with net.name_scope():
            net.add(nn.Dense(256, activation='relu'))
            net.add(nn.Dropout(.5))
            net.add(nn.Dense(120))
        if ParamsName is not None:
            net.collect_params().load(ParamsName,ctx)
        else:
            net.initialize(init=init.Xavier(), ctx=ctx)
        return net

#构建完整的网络模型，实验
class Net_updateAllFea():
    def __init__(self, ctx):
        inception = vision.inception_v3(pretrained=True, ctx=ctx)#提取预训练模型inception_v3的features
        self.finetune_net = vision.inception_v3(classes=120)
        self.finetune_net.features = inception.features
        self.finetune_net.output.initialize(init.Xavier())



#在pretest.py文件中使用，进行预测
class Pre():
    def __init__(self, nameparams, idx, ctx=0):
        self.idx = idx
        if ctx == 0:
            self.ctx = mx.cpu()
        if ctx == 1:
            self.ctx = mx.gpu()
        self.net = Net(self.ctx, nameparams=nameparams).net
        self.Timg = transform_test

    def PreImg(self, img):
        imgs = self.Timg(img, None)
        out = nd.softmax(self.net(nd.reshape(imgs[0], (1, 3, 224, 224)).as_in_context(self.ctx), nd.reshape(imgs[1], (1, 3, 299, 299)).as_in_context(self.ctx))).asnumpy()
        return self.idx[np.where(out == out.max())[1][0]]

    def PreName(self, Name):
        img = image.imread(Name)
        return self.PreImg(img)
