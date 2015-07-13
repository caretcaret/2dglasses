#!/usr/bin/env python
"""
Prepares data for caffe usage. Output directory is ./data.

Usage:
  scripts/prepare_data.py <class_directory>...
"""

from __future__ import print_function
import subprocess
import sys
import os
import errno
import glob
import random
from collections import Counter
import numpy as np

# insert pycaffe into pythonpath
caffe_path = os.path.abspath(os.getenv('CAFFE'))
pycaffe_path = caffe_path + '/python'
if pycaffe_path not in sys.path:
  sys.path.insert(0, pycaffe_path)
# suppress wall of text logging
os.environ['GLOG_minloglevel'] = '2'

import caffe
from caffe.proto import caffe_pb2

def mkdir_p(path):
  try:
    os.makedirs(path)
  except OSError as exc:
    if exc.errno == errno.EEXIST and os.path.isdir(path):
      pass
    else:
      raise

  return path

def perform_rescale(source, target, width, height):
  subprocess.call([
    'convert', source,
    '-resize', '{}x{}^'.format(width, height),
    '-gravity', 'center',
    '-flatten',
    '-crop', '{}x{}+0+0'.format(width, height),
    target
  ])

def main(output_directory, class_directories):
  mkdir_p(output_directory)
  mkdir_p("{}/train".format(output_directory))
  mkdir_p("{}/val".format(output_directory))
  train_class_path = "{}/train.txt".format(output_directory)
  val_class_path = "{}/val.txt".format(output_directory)

  classes = {}
  counts = Counter()

  for i, class_directory in enumerate(class_directories):
    for image_path in glob.iglob("{}/*.*".format(class_directory)):
      image_name, extension = os.path.splitext(os.path.basename(image_path))
      dataset = 'train' if random.random() < 0.8 else 'val'

      new_image_filename = "{}.jpg".format(image_name)
      new_image_path = os.path.abspath("{}/{}/{}".format(output_directory, dataset, new_image_filename))

      # dimensions used by caffenet
      perform_rescale(image_path, new_image_path, 256, 256)
      classes[dataset, new_image_filename] = i
      counts[i] += 1

  # write images
  with open(train_class_path, 'w') as train_class_file:
    with open(val_class_path, 'w') as val_class_file:
      for (dataset, path), val in classes.items():
        if dataset == 'train':
          train_class_file.write("{} {}\n".format(path, val))
        else:
          val_class_file.write("{} {}\n".format(path, val))

  # infogain matrix
  total = float(sum(counts.values()))
  n_labels = len(class_directories)
  matrix = np.array([[total / counts[j] if i == j else 0.0
              for j in range(n_labels)]
              for i in range(n_labels)])
  # normalize matrix so one lr step is similar in size
  H = matrix / (np.linalg.det(matrix) ** (1.0 / n_labels))

  # write H to prototxt
  H_blob = caffe.io.array_to_blobproto(H[np.newaxis, np.newaxis, :, :])
  H_bytes = H_blob.SerializeToString()
  with open("{}/infogain.binaryproto".format(output_directory), 'wb') as H_f:
    H_f.write(H_bytes)

if __name__ == '__main__':
  if len(sys.argv) < 3:
    print(__doc__)
  else:
    output_directory = './data'
    class_directories = sys.argv[1:]
    main(output_directory, class_directories)
