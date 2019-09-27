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

from imutils import paths

import numpy as np
import argparse
import cv2
import os
import sys
import codecs
import argparse



def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()


def is_blurry(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)

    return fm < threshold


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--dry_run",
        type=str2bool,
        default=True,
        help="Don't copy files.")
    parser.add_argument(
        "--src_dir",
        type=str,
        help="Source directory.")
    parser.add_argument(
        "--target_dir",
        type=str,
        help="Target directory.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=100.0,
        help="Threshold.")

    flags, unparsed = parser.parse_known_args()

    for arg in vars(flags):
        print(arg + "='" + str(getattr(flags, arg)) + "'")

    dry_run = flags.dry_run
    src_dir = flags.src_dir
    target_dir = flags.terget_dir
    threshold = flags.threshold

    blurry_count = 0
    sharp_count = 0
    for class_dir in os.listdir(src_dir):
        class_dir_path = os.path.join(src_dir, class_dir)
        if os.path.isdir(class_dir_path):
            for image_file_name in os.listdir(class_dir_path):
                image_file_path = os.path.join(class_dir_path, image_file_name)

                image = cv2.imread(image_file_path)

                # sharpen
                # kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                # image = cv2.filter2D(image, -1, kernel)

                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                fm = variance_of_laplacian(gray)
                text = "Not Blurry"

                # if the focus measure is less than the supplied threshold,
                # then the image should be considered "blurry"
                if fm < threshold:
                    text = "Blurry"
                    blurry_count += 1
                else:
                    sharp_count += 1
                #print(text)
                print(fm)

                # show the image
                cv2.putText(image, "{}: {:.2f}".format(text, fm), (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                cv2.imshow("Image", image)
                key = cv2.waitKey(0)

    print(blurry_count)
    print(sharp_count)