# -*- coding: utf-8 -*-
#
# Copyright © 2019 Attila Ulbert
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation 
# files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, 
# modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software 
# is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES 
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE 
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR 
# IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from shutil import move
from shutil import SameFileError

from sklearn.utils import shuffle

import os
import sys
import codecs
import argparse

import dataset_utils

_VALIDATION_SUBDIR = 'validation'
_TEST_SUBDIR = 'test'


def move_files(dry_run, split_source_dir, target_dir, file_list):
  for file in file_list:
    source_file = os.path.join(split_source_dir, file)
    destination_file = os.path.join(target_dir, file)
    print(source_file + "->" + destination_file)
    if not dry_run:
      try:
        move(source_file, destination_file)
      except SameFileError:
        print("Already exists: " + destination_file)


def move_split_and_test_files(dry_run, split_source_dir, validation_category_dir, test_category_dir, validation_files, test_files):
  print("dry run: " + str(dry_run))
  print("split source dir: " + split_source_dir)
  print("validation category dir: " + validation_category_dir)
  print("test category dir: " + test_category_dir)
  print("validation: " + str(validation_files))
  print("test:" + str(test_files))

  move_files(dry_run, split_source_dir, validation_category_dir, validation_files)
  move_files(dry_run, split_source_dir, test_category_dir, test_files)


def run(dry_run, dataset_dir, source_dir, validation_percent, test_percent):
  validation_dir = os.path.join(dataset_dir, _VALIDATION_SUBDIR)
  test_dir = os.path.join(dataset_dir, _TEST_SUBDIR)
  if not dry_run:
    if not os.path.exists(validation_dir) and not os.path.exists(test_dir):
      os.mkdir(validation_dir)
      os.mkdir(test_dir)
    else:
      print("Datasets already exist!!!")
      return

  for category in os.listdir(source_dir):
    category_dir = os.path.join(source_dir, category)
    if os.path.isdir(category_dir):

      image_list = []
      for image_file in os.listdir(category_dir):
        image_list.append(image_file)

      validation_category_dir = os.path.join(validation_dir, category)
      if not os.path.exists(validation_category_dir):
        os.mkdir(validation_category_dir)
      test_category_dir = os.path.join(test_dir, category)
      if not os.path.exists(test_category_dir):
        os.mkdir(test_category_dir)

      shuffle(image_list)
      validation_num = int(len(image_list)*validation_percent/100)
      test_num = int(len(image_list)*test_percent/100)

      if len(image_list) > 2:
        if validation_num == 0:
          validation_num = 1
        if test_num == 0:
          test_num = 1

      move_split_and_test_files(dry_run, category_dir, validation_category_dir, test_category_dir, image_list[:validation_num], image_list[validation_num+1:validation_num+test_num+1])

  print('\nFinished creating datasets')


def main():
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
    "--dry_run",
    type=bool,
    default=False,#True,
    help="Don't copy files.")
  parser.add_argument(
    "--dataset_dir",
    type=str,
    default="clinic",#"dataset/clinic",
    help="Root directory of the training, validation, and test datasets.")
  parser.add_argument(
    "--source_dir",
    type=str,
    default="clinic/train",
    help="Root directory of dataset to be split.")
  parser.add_argument(
    "--validation_percent",
    type=int,
    default=10,
    help="Percent of the validation set.")
  parser.add_argument(
    "--test_percent",
    type=int,
    default=10,
    help="Percent of the validation set.")
      
  flags, unparsed = parser.parse_known_args()

  print("dry_run: '" + str(flags.dry_run) + "'")
  print("dataset_dir: '" + flags.dataset_dir + "'")
  print("source_dir: '" + flags.source_dir + "'")
  print("validation_percent: '" + str(flags.validation_percent) + "'")
  print("test_percent: '" + str(flags.test_percent) + "'")
  
  run(flags.dry_run, flags.dataset_dir, flags.source_dir, flags.validation_percent, flags.test_percent)


if __name__ == '__main__':
  if sys.stdout.encoding != 'cp850':
    sys.stdout = codecs.getwriter('cp850')(sys.stdout.buffer, 'strict')
  if sys.stderr.encoding != 'cp850':
    sys.stderr = codecs.getwriter('cp850')(sys.stderr.buffer, 'strict')
    
  main()
