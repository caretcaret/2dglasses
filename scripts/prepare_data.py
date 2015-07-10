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

  # TODO: correct for class imbalance
  for i, class_directory in enumerate(class_directories):
    for image_path in glob.iglob("{}/*.*".format(class_directory)):
      image_name, extension = os.path.splitext(os.path.basename(image_path))
      dataset = 'train' if random.random() < 0.8 else 'val'

      new_image_filename = "{}.jpg".format(image_name)
      new_image_path = os.path.abspath("{}/{}/{}".format(output_directory, dataset, new_image_filename))

      # dimensions used by caffenet
      perform_rescale(image_path, new_image_path, 256, 256)
      classes[dataset, new_image_filename] = i

  with open(train_class_path, 'w') as train_class_file:
    with open(val_class_path, 'w') as val_class_file:
      for (dataset, path), val in classes.items():
        if dataset == 'train':
          train_class_file.write("{} {}\n".format(path, val))
        else:
          val_class_file.write("{} {}\n".format(path, val))


if __name__ == '__main__':
  if len(sys.argv) < 3:
    print(__doc__)
  else:
    output_directory = './data'
    class_directories = sys.argv[1:]
    main(output_directory, class_directories)
