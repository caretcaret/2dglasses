#!/usr/bin/env python
"""
Command line tool and library to classify an image.
Output is a probability vector.

Usage:
  classify.py </path/to/image>
"""
from __future__ import print_function

import numpy as np
import sys
import os

# insert pycaffe into pythonpath
caffe_path = os.path.abspath(os.getenv('CAFFE'))
pycaffe_path = caffe_path + '/python'
if pycaffe_path not in sys.path:
  sys.path.insert(0, pycaffe_path)
# suppress wall of text logging
os.environ['GLOG_minloglevel'] = '2'

import caffe
from caffe.proto import caffe_pb2

IM_MEAN_PATH = './data/val_mean.binaryproto'
DEPLOY_PATH = './model/deploy.prototxt'
CAFFEMODEL_PATH = './snapshots/bootstrap/caffenet_train_iter_10000.caffemodel'

def build_classifier(im_mean_path, deploy_path, caffemodel_path):
  caffe.set_mode_cpu()
  net = caffe.Net(deploy_path, caffemodel_path, caffe.TEST)

  im_mean_blob = caffe_pb2.BlobProto()
  with open(im_mean_path, 'rb') as im_mean_file:
    data = im_mean_file.read()
  
  im_mean_blob.ParseFromString(data)
  im_mean = np.squeeze(caffe.io.blobproto_to_array(im_mean_blob), axis=(0,))

  transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
  transformer.set_transpose('data', (2, 0, 1))
  transformer.set_mean('data', im_mean.mean(1).mean(1))
  transformer.set_raw_scale('data', 255)
  transformer.set_channel_swap('data', (2, 1, 0))

  net.blobs['data'].reshape(1, 3, 227, 227)

  return (net, transformer)

def classify_image(net, transformer, im):
  net.blobs['data'].reshape(1, 3, 227, 227)
  net.blobs['data'].data[...] = transformer.preprocess('data', im)
  p = net.forward()['prob']
  return (p[0][0], p[0][1])

def command_line(images):
  net, transformer = build_classifier(IM_MEAN_PATH, DEPLOY_PATH, CAFFEMODEL_PATH)

  for image_path in sys.argv[1:]:
    im = caffe.io.load_image(image_path)
    p_0, p_1 = classify_image(net, transformer, im)
    print("{} [2d: {}, 3d: {}]".format(image_path, p_0, p_1))

if __name__ == '__main__':
  if len(sys.argv) < 2:
    print(__doc__)

  else:
    command_line(sys.argv[1:])
