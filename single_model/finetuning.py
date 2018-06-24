import sys
sys.path.insert(0, '..')
import gluonbook as gb
from mxnet import nd, image, gluon, init
from mxnet.gluon.data.vision import transforms
import mxnet as mx
import numpy as np

data_dir = 'E:\\WorkSpace\\gluon-tutorials-zh_aa12356jm\\data\\kaggle_dog\\train_valid_test\\'

#对训练数据进行转换，如数据增强等等，将每个数据都处理2次，结果返回对应处理后的结果和标签
def transform_train(data, label):

    #将图像调整为不同的大小，分别进行数据扩增,因为是使用两个模型进行融合，所以也使用两类数据分别给两个网络进行训练
    im1 = image.imresize(data.astype('float32') / 255, 224, 224)#将图像调整为224x224，data的每个像素点除以255,保证在[0-1]
    #im2 = image.imresize(data.astype('float32') / 255, 299, 299)

    #数据增强参数1，给第一个网络
    auglist1 = image.CreateAugmenter(data_shape=(3, 224, 224), resize=0,
                        rand_crop=False, rand_resize=False, rand_mirror=True,
                        mean=np.array([0.485, 0.456, 0.406]), std=np.array([0.229, 0.224, 0.225]),
                        brightness=0, contrast=0, saturation=0, hue=0,
                        pca_noise=0, rand_gray=0, inter_method=2)

    # 数据增强参数2，给第二个网络
    # auglist2 = image.CreateAugmenter(data_shape=(3, 299, 299), resize=0,
    #                     rand_crop=False, rand_resize=False, rand_mirror=True,
    #                     mean=np.array([0.485, 0.456, 0.406]), std=np.array([0.229, 0.224, 0.225]),
    #                     brightness=0, contrast=0, saturation=0, hue=0,
    #                     pca_noise=0, rand_gray=0, inter_method=2)
    #分别进行增强
    for aug in auglist1:
        im1 = aug(im1)
    # for aug in auglist2:
    #     im2 = aug(im2)
    # 将数据格式从"高*宽*通道"改为"通道*高*宽"。
    im1 = nd.transpose(im1, (2, 0, 1))
    #im2 = nd.transpose(im2, (2, 0, 1))
    #返回给两个数据，分别给两个网络进行训练
    #return (im1, im2, nd.array([label]).asscalar().astype('float32'))
    return (im1, nd.array([label]).asscalar().astype('float32'))

#对测试数据进行数据扩增，同上
def transform_test(data, label):
    im1 = image.imresize(data.astype('float32') / 255, 224, 224)
    #im2 = image.imresize(data.astype('float32') / 255, 299, 299)
    auglist1 = image.CreateAugmenter(data_shape=(3, 224, 224),
                        mean=np.array([0.485, 0.456, 0.406]),
                        std=np.array([0.229, 0.224, 0.225]))
    # auglist2 = image.CreateAugmenter(data_shape=(3, 299, 299),
    #                     mean=np.array([0.485, 0.456, 0.406]),
    #                     std=np.array([0.229, 0.224, 0.225]))
    for aug in auglist1:
        im1 = aug(im1)
    # for aug in auglist2:
    #     im2 = aug(im2)
    # 将数据格式从"高*宽*通道"改为"通道*高*宽"。
    im1 = nd.transpose(im1, (2, 0, 1))
    #im2 = nd.transpose(im2, (2, 0, 1))
    #return (im1, im2, nd.array([label]).asscalar().astype('float32'))
    return (im1, nd.array([label]).asscalar().astype('float32'))

train_imgs = gluon.data.vision.ImageFolderDataset(data_dir + 'train', flag=1,
                                     transform=transform_train)
test_imgs = gluon.data.vision.ImageFolderDataset(data_dir + 'valid', flag=1,
                                     transform=transform_test)
batch_size = 32
learning_rate = 0.01
loader = gluon.data.DataLoader
train_data = loader(train_imgs, batch_size, shuffle=True, last_batch='keep')
valid_data = loader(test_imgs, batch_size, shuffle=True, last_batch='keep')


def train(net, epochs=5):
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': learning_rate, 'wd': 0.001})
    gb.train(train_data, valid_data, net, loss, trainer, ctx, epochs)
    net.export("model")

ctx = gb.try_all_gpus()
#组建网络
pretrained_net = gluon.model_zoo.vision.resnet18_v1(pretrained=True)
finetune_net = gluon.model_zoo.vision.resnet18_v1(classes=120)
finetune_net.features = pretrained_net.features
finetune_net.output.initialize(init.Xavier())
finetune_net.collect_params().reset_ctx(ctx)
finetune_net.hybridize()
train(finetune_net, 10)
#scratch_net = gluon.model_zoo.vision.resnet18_v2(classes=120)
#scratch_net.initialize(init=init.Xavier())
#train(scratch_net, 0.1)