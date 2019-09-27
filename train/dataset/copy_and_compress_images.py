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
from scipy import misc

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


def main():
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--dir_list_path",
        type=str,
        help="Path to the file of directory list.")
    parser.add_argument(
        "--source_root_path",
        type=str,
        help="Path to the root of the directories in dir_list_path.")

    parser.add_argument(
        "--target_path",
        type=str,
        help="Path to the target dir.")

    flags, unparsed = parser.parse_known_args()

    dir_list_path = flags.dir_list_path
    target_path = flags.target_path
    source_root_path = flags.source_root_path

    print("dir_list_path: '" + str(dir_list_path) + "'")
    print("target_path: '" + str(target_path) + "'")
    print("source_root_path: '" + str(source_root_path) + "'")

    f = open(dir_list_path, "r", encoding='utf-8')
    lines = f.readlines()

    aaa = 0
    for line in lines:
        dir = line.strip()
        #print(dir + "-->")

        destination_dir = os.path.join(target_path, dir)
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)

        source_dir = os.path.join(source_root_path, dir)
        for filename in os.listdir(source_dir):
            input_file = os.path.join(source_dir, filename)
            destination_file = os.path.join(destination_dir, filename)
            #print(input_file + " --> " + destination_file)
            aaa += 1
            print(str(aaa))

            if os.path.exists(destination_file):
                print("Exists!")
                continue

            try:
                img = Image.open(input_file)
                exif = img.info['exif']
                img = img.resize((int(img.size[0] / 3), int(img.size[1] / 3)), Image.LANCZOS)
                img.save(destination_file, exif=exif)
            except IOError:
                print("IOError")#, filename)
                continue


if __name__ == '__main__':
    if sys.stdout.encoding != 'cp850':
        sys.stdout = codecs.getwriter('cp850')(sys.stdout.buffer, 'strict')
    if sys.stderr.encoding != 'cp850':
        sys.stderr = codecs.getwriter('cp850')(sys.stderr.buffer, 'strict')

    main()
