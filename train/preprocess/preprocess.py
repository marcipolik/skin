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

import os
import cv2
import numpy as np

#import matplotlib.pyplot as plt

from PIL import Image


def pp_white_patch_retinex(img):
    img = img.astype(np.float)
    out = np.zeros(img.shape, dtype=float)
    L = [0, 0, 0]
    n_p = 0.1 * img.shape[0] * img.shape[1]
    for i in range(3):
        H, bins = np.histogram(img[:, :, i].flatten(), 256)
        sums = 0
        for j in range(255, -1, -1):
            if sums < n_p:
                sums += H[j]
            else:
                L[i] = j
                out[:, :, i] = img[:, :, i] * 255.0 / L[i]
                break

    return out


def preprocess_image_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    img = img.astype(np.float32)
    img /= 255.

    return img


def test_preprocess():
    for file in os.listdir("test"):
        file = os.path.join("test", file)

        img = Image.open(file)
        img = pp_white_patch_retinex(np.asarray(img))

        plt.imshow(img)
        plt.show()


if __name__ == "__main__":
    test_preprocess()
