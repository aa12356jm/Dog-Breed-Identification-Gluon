import mxnet as mx
from mxnet import gluon,nd
from mxboard import SummaryWriter
import time
'''
加载一个网络，进行前向计算，然后绘制网络图
'''
ctx = mx.gpu(0)
net = gluon.model_zoo.vision.AlexNet(classes=10)
net.hybridize()
net.initialize(ctx=ctx, init=mx.init.Xavier())
net.forward(nd.ones((1, 3, 227, 227)).as_in_context(ctx))#注意这里net是在gpu上，所以也需要将数据放在gpu上

sw = SummaryWriter('./log/%s' % (time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())))
sw.add_graph(net)

