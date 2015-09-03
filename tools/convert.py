#!/usr/bin/env python
"""
Converts an image to fit a style given by a texture.

Usage:
  convert.py convert <texture> <source> <destination>
  convert.py build <source> <destination>
"""

from __future__ import print_function

import numpy as np
import scipy.ndimage as nd
import sys
import skimage.io
from PIL import Image
import os
import sys
import glob
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# insert pycaffe into pythonpath
caffe_path = os.path.abspath(os.getenv('CAFFE'))
pycaffe_path = caffe_path + '/python'
if pycaffe_path not in sys.path:
  sys.path.insert(0, pycaffe_path)
# suppress wall of text logging
os.environ['GLOG_minloglevel'] = '2'

import caffe
from caffe.proto import caffe_pb2

DEPLOY_PATH = './models/vgg/deploy.prototxt'
CAFFEMODEL_PATH = './snapshots/VGG_ILSVRC_19_layers.caffemodel'
DEFAULT_CONTENT_LAYERS = ['conv4_2']
DEFAULT_STYLE_LAYERS = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

def build_net(deploy_path, caffemodel_path, mode=caffe.TRAIN):
  caffe.set_mode_cpu()

  return caffe.Classifier(deploy_path, caffemodel_path, mode,
      mean=np.array([103.939, 116.779, 123.680]), channel_swap=(2, 1, 0))

def preprocess(net, im):
  return np.float32(np.rollaxis(im, 2)[::-1]) - net.transformer.mean['data']

def deprocess(net, im):
  return np.dstack((im + net.transformer.mean['data'])[::-1])

def layer_texture(net, layer):
  # activations already exist in the net
  activations = net.blobs[layer].data[0]
  n_filters = activations.shape[0]
  activations = activations.reshape((n_filters, -1))
  return activations.dot(activations.T) #/ activations.shape[1]

def build_texture(net, im_files, layers=None):
  """
  Texture is G, where G[l][i, j] is the dot product of the activations of filters i and j of layer l.
  Each dot product is normalized by the dimensions of the activations.

  If multiple images are given, the textures for each image are averaged together.
  """
  layers = layers or DEFAULT_STYLE_LAYERS
  G = {}

  for layer in layers:
    n_filters = net.blobs[layer].data.shape[1]
    G[layer] = np.zeros((len(im_files), n_filters, n_filters))

  for n, im_file in enumerate(im_files):
    # feed forward
    im = np.float32(Image.open(im_file))
    im = preprocess(net, im)

    if im.shape[1] * im.shape[2] > 720 * 442:
      # XXX temporary resize
      print("Warning: image may be too big and cause caffe to segfault, cropping")
      im = im[:, :720, :442]
    blob = net.blobs['data']
    # resize net input
    blob.reshape(1, *im.shape)
    blob.data[0] = im
    net.forward(end=layers[-1])

    # compute dot products
    for layer in layers:
      activations = net.blobs[layer].data[0]
      n_filters = activations.shape[0]
      activations = activations.reshape((n_filters, -1))
      G[layer][n, :, :] = activations.dot(activations.T) #/ activations.shape[1]

  # average imagewise textures
  for layer in G:
    G[layer] = np.average(G[layer], axis=0)

  return G

def adadelta(data, diff, param):
  p = param.get('p', 0.95) # decay constant
  e = param.get('e', 1e-2) # epsilon
  
  if not 'g_t' in param:
    # initialize accumulation variables
    param['g_t'] = np.zeros_like(data)
    param['dx_t'] = np.zeros_like(data)
  
  g_t = param['g_t']
  dx_t = param['dx_t']
      
  # accumulate gradient
  g_t[:] = p * g_t + (1.0 - p) * diff * diff
  
  # compute update
  rms_g = np.sqrt(g_t + e)
  rms_dx = np.sqrt(dx_t + e)
  dx = -rms_dx / rms_g * diff
  
  # accumulate update
  dx_t[:] = p * dx_t + (1.0 - p) * dx * dx
  
  # apply update
  data += dx
  return dx

def irpropm(data, diff, param):
  n_p = param.get('n_p', 1.2) # increase factor
  n_m = param.get('n_m', 0.5) # decrease factor
  step_max = param.get('step_max', 5.0)
  step_min = param.get('step_min', 1e-6)
  
  if not 'g_t' in param:
    param['g_t'] = np.zeros_like(data)
    param['dx_t'] = np.ones_like(data) / 2.0
      
  g_t = param['g_t']   # prev gradient
  dx_t = param['dx_t'] # prev step
  
  d_p = (diff * g_t) >= 0
  d_m = (diff * g_t) < 0
  
  dx_t[:] = np.maximum(dx_t, step_min)
  d = (d_p * np.minimum(dx_t * n_p, step_max)) + (d_m * np.maximum(dx_t * n_m, step_min))
  diff[d_m] = 0.0
  
  step = -np.sign(diff) * d
  data += step
  
  g_t[:] = diff
  dx_t[:] = d
  return step

def train_content_layer(net, layer, reference_data, alpha, param):
  net.forward(end=layer)

  activation = net.blobs[layer].data[0]
  ref_activation = reference_data[layer]
  net.blobs[layer].diff[0] = alpha * (activation - ref_activation) * (activation > 0)

  net.backward(start=layer)

  blob = net.blobs['data']
  irpropm(blob.data[0], blob.diff[0], param)

def train_style_layer(net, layer, texture, beta, layer_weight, param):
  net.forward(end=layer)

  G_l = layer_texture(net, layer)

  activation = net.blobs[layer].data[0]
  n_filters = activation.shape[0]
  n_size = activation.shape[1] * activation.shape[2]
  net.blobs[layer].diff[0] = beta * layer_weight / ((n_filters * n_size) ** 2) * \
    (G_l - texture[layer]).T.dot(activation.transpose(1, 0, 2)) * (activation > 0)

  net.backward(start=layer)

  blob = net.blobs['data']
  irpropm(blob.data[0], blob.diff[0], param)

def convert(net, im, texture, n_iter=250, content_layers=None, style_layers=None, alpha=1.0, beta=1000.0):
  content_layers = content_layers or DEFAULT_CONTENT_LAYERS
  style_layers = style_layers or DEFAULT_STYLE_LAYERS
  layer_weight = 1.0 / len(style_layers)
  im = preprocess(net, im)

  # compute reference activations
  reference_data = {}
  blob = net.blobs['data']
  blob.reshape(1, *im.shape)
  blob.data[0] = im
  for layer in content_layers:
    net.forward(end=layer)
    reference_data[layer] = np.copy(net.blobs[layer].data[0])

  param = {}

  for i in range(n_iter):
    # backprop onto reference image according to loss from http://arxiv.org/abs/1508.06576
    for layer in content_layers:
      train_content_layer(net, layer, reference_data, alpha, param)
    for layer in style_layers:
      train_style_layer(net, layer, texture, beta, layer_weight, param)

    print(i, end=" ")
    sys.stdout.flush()

  im = blob.data[0]
  return deprocess(net, im)

def convert_cmd(target, source, destination):
  texture = np.load(target)
  im = np.float32(Image.open(source))
  net = build_net(DEPLOY_PATH, CAFFEMODEL_PATH)
  im_new = convert(net, im, texture)

  im_new = np.uint8(np.clip(im_new, 0, 255))
  skimage.io.imsave(destination, im_new)

def build_cmd(source, destination):
  if os.path.isfile(source):
    im_files = [source]
  elif os.path.isdir(source):
    im_files = glob.iglob("{}/*.*".format(source))
  else:
    raise ValueError("Source does not exist.")

  net = build_net(DEPLOY_PATH, CAFFEMODEL_PATH)
  G = build_texture(net, im_files)
  with open(destination, 'wb') as f:
    np.savez(f, **G)


if __name__ == '__main__':
  if len(sys.argv) == 5 and sys.argv[1] == 'convert':
    convert_cmd(sys.argv[2], sys.argv[3], sys.argv[4])

  elif len(sys.argv) == 4 and sys.argv[1] == 'build':
    build_cmd(sys.argv[2], sys.argv[3])

  else:
    print(__doc__)
