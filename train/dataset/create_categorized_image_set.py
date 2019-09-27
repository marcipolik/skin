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

from shutil import copyfile
from shutil import copy2
from shutil import SameFileError

from PIL import Image
from PIL.ExifTags import TAGS

import numpy as np

import math
import os
import random
import sys
import codecs
import argparse
import sqlite3
import cv2

import dataset_utils

from derma_db_to_dirs import normalize_dg

def get_exif(fn):
    ret = {}
    i = Image.open(fn)
    info = i._getexif()
    for tag, value in info.items():
        decoded = TAGS.get(tag, tag)
        ret[decoded] = value

    a = np.array(i)
    val = a[int(a.shape[0] / 2)][int(a.shape[1] / 2)]
    ret['val'] = val

    # print(fn)
    # print(val)
    # print(np.var(val))
    # print(np.max(val) - np.min(val))


    #print(i.histogram())
    # print(np.argmax((i.histogram())))
    # print(np.max((i.histogram())))
    # print(np.median(np.array(i)))
    #print(sum(i.histogram()))
    # print(sum(i.histogram()[-50:]))
    # print(i.histogram()[:50])
    #print(sum(i.histogram()[:50]))
    #print(sum(i.histogram()[-50:]))
    #ret['startsum'] = sum(i.histogram()[:50])
    #ret['endsum'] = sum(i.histogram()[-50:])

    return ret


def copy_files(src, dest):
    try:
        if not dry_run:
            copy2(src, dest)
    except SameFileError:
        pass


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    if sys.stdout.encoding != 'cp850':
        sys.stdout = codecs.getwriter('cp850')(sys.stdout.buffer, 'strict')
    if sys.stderr.encoding != 'cp850':
        sys.stderr = codecs.getwriter('cp850')(sys.stderr.buffer, 'strict')

    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--dry_run",
        type=str2bool,
        default=True,
        help="Don't copy files, just create the csv.")
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Path to the input images.")
    parser.add_argument(
        "--target_dir",
        type=str,
        help="Path to the target dir.")

    flags, unparsed = parser.parse_known_args()

    input_dir = flags.input_dir
    target_dir = flags.target_dir
    dry_run = flags.dry_run

    for arg in vars(flags):
        print(arg + "='" + str(getattr(flags, arg)) + "'")

    if target_dir is not None:
        csv_file = open(os.path.join(target_dir, "results.csv"), "w", encoding='utf-8')

    i = 0
    for letter_dir in os.listdir(input_dir):
        dir = os.path.join(input_dir, letter_dir)

        if os.path.isdir(dir):
            for patient_dir in os.listdir(dir):
                letter_dir_path = os.path.join(dir, patient_dir)

                if os.path.isdir(letter_dir_path):
                    for image_name in os.listdir(letter_dir_path):
                        image_file_path = os.path.join(letter_dir_path, image_name)

                        i += 1
                        try:
                            target_dir_path = None
                            if not dry_run:
                                target_dir_path = os.path.join(target_dir, os.path.join(letter_dir, patient_dir))
                                if not os.path.exists(target_dir_path):
                                    os.makedirs(target_dir_path)

                            exif = get_exif(image_file_path)

                            if target_dir is None:
                                continue

                            val = exif['val']
                            if val[0] > 100 and (np.max(val) - np.min(val) < 20 or np.var(val) < 90):
                                csv_file.write(image_file_path + ";0;0;1\n")
                                target_file_path = os.path.join(target_dir, "paper_" + str(i) + ".jpg")
                            elif exif['FocalLength'][0] == 0 or exif['FNumber'][0] == 0:
                                csv_file.write(image_file_path + ";1;0;0\n")
                                target_file_path = os.path.join(target_dir_path, "micro_" + str(i) + ".jpg")
                            else:
                                csv_file.write(image_file_path + ";0;1;0\n")
                                target_file_path = os.path.join(target_dir_path, "macro_" + str(i) + ".jpg")

                            copy_files(image_file_path, target_file_path)

                        except IOError:
                            #print("IOError", image_file_path)
                            continue


