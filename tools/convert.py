#!/usr/bin/env python
"""
Convert an image to 2d or 3d.

Usage:
  convert.py 2d <source> <destination>
  convert.py 3d <source> <destination>
"""

from __future__ import print_function
from classify import IM_MEAN_PATH, DEPLOY_PATH, CAFFEMODEL_PATH

import numpy as np
import scipy.ndimage as nd
import sys
import skimage.io
from PIL import Image
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import caffe
from caffe.proto import caffe_pb2

def build_net(im_mean_path, deploy_path, caffemodel_path, mode=caffe.TRAIN):
  caffe.set_mode_cpu()

  im_mean_blob = caffe_pb2.BlobProto()
  with open(im_mean_path, 'rb') as im_mean_file:
    data = im_mean_file.read()

  im_mean_blob.ParseFromString(data)
  im_mean = np.squeeze(caffe.io.blobproto_to_array(im_mean_blob), axis=(0,))

  net = caffe.Classifier(deploy_path, caffemodel_path, mode,
      mean=im_mean.mean(1).mean(1), channel_swap=(2, 1, 0))

  return net

def preprocess(net, im):
  return np.float32(np.rollaxis(im, 2)[::-1]) - net.transformer.mean['data']

def deprocess(net, im):
  return np.dstack((im + net.transformer.mean['data'])[::-1])

def step_scale(net, base_img, scale, objective, n_iter=1, step_size=2, tile_size=227):
  # downscale image
  scaled = nd.zoom(base_img, (1, scale, scale), order=1)
  margin = (227 - tile_size) / 2

  src = net.blobs['data']
  dst = net.blobs['fc8_binary']

  detail = np.zeros_like(scaled)

  for i in range(n_iter):
    # collect diffs for each tile separately
    iter_detail = np.zeros_like(scaled)
    # iterate through tiles
    for r in range(0, scaled.shape[1], tile_size):
      for c in range(0, scaled.shape[2], tile_size):
        # offset window to center the actual tile size within 227x227 window
        window = scaled.take(range(r - margin, r - margin + 227), axis=1, mode='wrap') \
                       .take(range(c - margin, c - margin + 227), axis=2, mode='wrap')
        detail_window = detail.take(range(r - margin, r - margin + 227), axis=1, mode='wrap') \
                              .take(range(c - margin, c - margin + 227), axis=2, mode='wrap')
        src.data[0] = window + detail_window
        # run the network with objective
        net.forward(end='fc8_binary')
        objective(dst)
        net.backward(start='fc8_binary')
        g = src.diff[0]
        # normalize the diff
        g = g * step_size / (np.abs(g).mean() + np.finfo(np.float32).eps)
        # copy diff out (this shouldn't be that hard -_-)
        r_begin, c_begin = r + margin, c + margin
        r_end = min(r_begin + tile_size, iter_detail.shape[1])
        c_end = min(c_begin + tile_size, iter_detail.shape[2])
        r_len, c_len = r_end - r_begin, c_end - c_begin
        iter_detail[:, r_begin:r_end, c_begin:c_end] = g[:, :r_len, :c_len]

    # add diff this iteration to overall diff to affect next iteration
    detail += iter_detail

    # clip detail
    bias = net.transformer.mean['data']
    detail = np.clip(scaled + detail, -bias, 255 - bias) - scaled

  # upscale detail; inverse scale calculated with sizes explicitly to get exact upscaled size
  inv_scale_x = float(base_img.shape[1]) / scaled.shape[1]
  inv_scale_y = float(base_img.shape[2]) / scaled.shape[2]

  return nd.zoom(detail, (1, inv_scale_x, inv_scale_y), order=1)

def get_objective(target):
  def objective(dst):
    # constant error for the target, adjust other classes to 0
    # https://www.reddit.com/r/MachineLearning/comments/3df6ps//ct4u7id
    dst.diff[:] = np.array([[1 if i == target else -dst.data[0][i] for i in range(dst.data.shape[1])]])
  return objective

def convert(net, im, target, n_iter=50, n_scale=1, scale_factor=2**0.5, **step_params):
  objective = get_objective(target)

  im = preprocess(net, im)

  # start at smallest scale
  scale_base = max(227.0 / im.shape[1], 227.0 / im.shape[2])
  scales = [scale_base * (scale_factor ** i) for i in range(n_scale)]

  detail = np.zeros_like(im)
  for scale_ratio in scales:
    detail += step_scale(net, im + detail, scale_ratio, objective, n_iter=n_iter, **step_params)

  return deprocess(net, im + detail)

def command_line(target, source, destination):
  if target == '2d':
    class_ = 0
  elif target == '3d':
    class_ = 1
  else:
    raise ValueError("Not 2d or 3d")
  im = np.float32(Image.open(source))
  net = build_net(IM_MEAN_PATH, DEPLOY_PATH, CAFFEMODEL_PATH)
  im_new = convert(net, im, class_)

  im_new = np.uint8(np.clip(im_new, 0, 255))
  skimage.io.imsave(destination, im_new)


if __name__ == '__main__':
  if len(sys.argv) < 4:
    print(__doc__)

  else:
    command_line(sys.argv[1], sys.argv[2], sys.argv[3])
