#Author: Hao ZHANG PIERRE
#Date : 2015/03/18
#Function : visualize convolutional filters
import numpy as np
import matplotlib.pyplot as plt
import pylab
caffe_root = '/opt/caffe-master/'
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
plt.rcParams['figure.figsize'] = (10,10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
import os
caffe.set_mode_cpu()
#net = caffe.Net(caffe_root + 'data/MyData/deploy.prototxt', caffe_root + 'models/finetune_yifu_iter_10000.caffemodel',caffe.TEST)
net = caffe.Net(caffe_root + 'models/finetune_googlenet/deploy.prototxt', caffe_root + 'models/finetune_googlenet/bvlc_googlenet.caffemodel',caffe.TEST)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) 
transformer.set_raw_scale('data', 255)  
transformer.set_channel_swap('data', (2,1,0))  
#net.blobs['data'].reshape(1,3,227,227)
net.blobs['data'].reshape(1,3,224,224)
net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(caffe_root + 'matlab/caffe/9.jpg'))
out = net.forward()
#print("Predicted class is #{}.".format(out['prob'].argmax()))
def vis_square(data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    plt.imshow(data)
    pylab.show()
plt.imshow(transformer.deprocess('data', net.blobs['data'].data[0]))
pylab.show()
filters = net.params['conv1/7x7_s2'][0].data
print filters.ndim
print filters.shape
print filters.size
print len(filters)
vis_square(filters.transpose(0, 2, 3, 1))
feat = net.blobs['conv1/7x7_s2'].data[0, :36]
vis_square(feat, padval=1)
filters = net.params['conv2/3x3_reduce'][0].data
print filters.ndim
print filters.shape
print filters.size
print len(filters)
vis_square(filters[:64].reshape(64**2, 1, 1))#48 conv1 one part
#vis_square(filters[:12].reshape(12*32,3,3))
#filters = filters[:24].reshape(24**2,9,9)
#print filters.ndim
#print filters.shape
#print filters.size
#print len(filters)
feat = net.blobs['conv2/3x3_reduce'].data[0, :64]
vis_square(feat, padval=1)
feat = net.blobs['conv3'].data[0]
vis_square(feat, padval=0.5)
feat = net.blobs['conv4'].data[0]
vis_square(feat, padval=0.5)
feat = net.blobs['conv5'].data[0]
vis_square(feat, padval=0.5)
feat = net.blobs['pool5'].data[0]
vis_square(feat, padval=1)
feat = net.blobs['fc6'].data[0]
plt.subplot(2, 1, 1)
plt.plot(feat.flat)
plt.subplot(2, 1, 2)
_ = plt.hist(feat.flat[feat.flat > 0], bins=100)
feat = net.blobs['fc7'].data[0]
plt.subplot(2, 1, 1)
pylab.show()
plt.plot(feat.flat)
plt.subplot(2, 1, 2)
_ = plt.hist(feat.flat[feat.flat > 0], bins=100)
pylab.show()
feat = net.blobs['prob'].data[0]
plt.plot(feat.flat)
pylab.show()
