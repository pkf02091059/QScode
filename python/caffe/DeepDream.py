#!/usr/bin/env python
# coding=utf-8
############################################
# DeepDream
# Written by Pierre_Hao on 2015/08/24
############################################

from cStringIO import StringIO
import numpy as np
import scipy.ndimage as nd
import PIL.Image
from IPython.display import clear_output, Image, display
from google.protobuf import text_format
import sys
import caffe
import time


def imresize(img, w=800):
    """Make image small if necessary"""
    if img.size[0] > w:
        scale = 1.0*img.size[0]/w
        return img.resize((w, int(img.size[1]/scale)))
    else:
        return img


def blur(img, sigma):
    if sigma > 0:
        img[0] = nd.filters.gaussian_filter(img[0], sigma, order=0)
        img[1] = nd.filters.gaussian_filter(img[1], sigma, order=0)
        img[2] = nd.filters.gaussian_filter(img[2], sigma, order=0)
    return img


def showarray(a, fmt='jpeg'):
    """Display image in windows"""
    a = np.uint8(np.clip(a, 0, 255))
    f = StringIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))

    
# Class DD: DeepDream
class DD():
    """The main class for deep dream"""
    def __init__(self):
        caffe.set_mode_gpu()
        #caffe.set_device(0)
        model_path = '../models/bvlc_googlenet/' # substitute your path here
        net_fn   = model_path + 'deploy.prototxt'
        param_fn = model_path + 'bvlc_googlenet.caffemodel'
        model = caffe.io.caffe_pb2.NetParameter()
        text_format.Merge(open(net_fn).read(), model)
        model.force_backward = True #backward to input layer
        open('tmp.prototxt', 'w').write(str(model))
        self.net = caffe.Classifier('tmp.prototxt', param_fn,
                       mean = np.float32([104.0, 116.0, 122.0]), 
                       channel_swap = (2,1,0))
        # for the mode guide, if flag = 1               
        self.flag = 0
        self.epoch = 20
        self.end = 'inception_4c/output'
        #self.end = 'conv4'
    
    def GenerateInputImage(self):
        original_w = 2*self.net.blobs['data'].width
        original_h = 2*self.net.blobs['data'].height
        # the background color of the initial image
        background_color = np.float32([200.0, 200.0, 200.0])
        # generate initial random image
        gen_image = np.random.normal(background_color, 8, (original_w, original_h, 3))
        return gen_image
        
    def Preprocess(self, img):
        return np.float32(np.rollaxis(img, 2)[::-1]) - self.net.transformer.mean['data']

    def Deprocess(self, img):
        return np.dstack((img + self.net.transformer.mean['data'])[::-1])
    
    def Objective_L2(self,dst):
        dst.diff[:] = dst.data 

    def Make_step(self, step_size=1.5, jitter=32, clip=True):
        if self.flag == 0:
            objective = self.Objective_L2
        else:
            objective = self.Objective_guide    
        src = self.net.blobs['data']
        dst = self.net.blobs[self.end]
        ox, oy = np.random.randint(-jitter, jitter+1, 2)
        src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2) # apply jitter shift            
        self.net.forward(end=self.end)
        #one_hot = np.zeros_like(dst.data)
        #one_hot.flat[113] = 0.99
        #dst.diff[:] = one_hot
        objective(dst)  # specify the optimization objective
        self.net.backward(start=self.end)
        g = src.diff[0]
        # apply normalized ascent step to the input image
        src.data[:] += (step_size/np.abs(g).mean()) * g
        src.data[0] = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2) # unshift image            
        if clip:
            bias = self.net.transformer.mean['data']
            src.data[:] = np.clip(src.data, -bias, 255-bias)
        #src.data[0] = blur(src.data[0], 0.4)
        
    def Deepdream(self, base_img, iter_n=10, octave_n=4, octave_scale=1.4, clip=True):
        # prepare base images for all octaves
        octaves = [self.Preprocess(base_img)]
        for i in xrange(octave_n-1):
            # shrink the image octave[0] so that function always return image size as octave[0]
            octaves.append(nd.zoom(octaves[-1], (1, 1.0/octave_scale,1.0/octave_scale), order=1))

        src = self.net.blobs['data']
        detail = np.zeros_like(octaves[-1]) # allocate image for network-produced details
        for octave, octave_base in enumerate(octaves[::-1]):# from end to 0
            h, w = octave_base.shape[-2:]
            if octave > 0:
                # upscale details from the previous octave
                h1, w1 = detail.shape[-2:]
                detail = nd.zoom(detail, (1, 1.0*h/h1,1.0*w/w1), order=1)
            src.reshape(1,3,h,w) # resize the network's input image size
            src.data[0] = octave_base+detail
            for i in xrange(iter_n):
                self.Make_step()
                # visualization
            	'''
            	vis = self.deprocess(net, src.data[0])
            	if not clip: # adjust image contrast if clipping is disabled
                	vis = vis*(255.0/np.percentile(vis, 99.98))
            	showarray(vis)
            	print octave, i, end, vis.shape
            	clear_output(wait=True)
                '''
        	# extract details produced on the current octave
        	#print octave, self.end, src.data[0].shape
        	detail = src.data[0]-octave_base
    	# returning the resulting image
    	return self.Deprocess(src.data[0])
    	
    def Get_guide(self):
        """Generate guide image feature"""
    	guide = np.float32(imresize(PIL.Image.open(self.guide_path),224))
    	h,w = guide.shape[:2]
    	src, dst = self.net.blobs['data'], self.net.blobs[self.end]
    	src.reshape(1,3,h,w)
    	src.data[0] = self.Preprocess(guide)
    	self.net.forward(end=self.end)
    	self.guide_features = dst.data[0].copy()
    	self.flag = 1
    	
    def Objective_guide(self, dst):
        """Another type of objective, mainly for guide mode"""
    	x = dst.data[0].copy()
    	y = self.guide_features
    	ch = x.shape[0]
    	x = x.reshape(ch,-1)
    	y = y.reshape(ch,-1)
    	A = x.T.dot(y) # compute the matrix of dot-products with guide features
    	dst.diff[0].reshape(ch,-1)[:] = y[:,A.argmax(1)] # select ones that match best

    def Run(self, img_path, guide_image_path='', objective=0):
        """Run deep dream"""
        self.guide_path = guide_image_path
        if self.guide_path != '':
            self.Get_guide()
        self.net.blobs.keys()
        if img_path != '':
            frame = PIL.Image.open(img_path)
            frame = imresize(frame)
            frame = np.float32(frame)
        else:
            frame = self.GenerateInputImage()
        frame_i = 0
        h, w = frame.shape[:2]
        #s = 0.05 # scale coefficient
        for i in xrange(self.epoch):
            start = time.time()
            frame = self.Deepdream(frame)
            PIL.Image.fromarray(np.uint8(frame)).save("frames/%04d.jpg"%frame_i)
            #frame = nd.affine_transform(frame, [1-s,1-s,1], [h*s/2,w*s/2,0], order=1)
            frame_i += 1
            stop = time.time()
            print "Time cost for {:d}th image: {:.3f} s".format(i,stop-start)


if __name__ == '__main__':
    deepdream = DD()
    if len(sys.argv) < 2:
        img_path = ''
        deepdream.Run(img_path)
    elif len(sys.argv) == 2:
        img_path = sys.argv[1]
        deepdream.Run(img_path)
    else:
        img_path = sys.argv[1]
        img2_path = sys.argv[2]
        deepdream.Run(img_path, img2_path)    
