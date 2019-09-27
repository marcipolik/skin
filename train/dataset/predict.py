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

import itertools
import glob
import os
import numpy as np
import keras
import pickle
import csv
import sys
import codecs
import argparse

import matplotlib.pyplot as plt

import predict as pred

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.metrics import classification_report

from scipy import interp
from itertools import cycle

from PIL import Image

import keras.preprocessing.image as image
from keras.applications.nasnet import NASNetMobile, NASNetLarge, preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras import backend as K


num_classes = 2
#class_names = [AKIEC  BCC  ,"BKL","DF","MEL","NV","VASC"]
class_names = ["m", "n"]
not_diagnosed_count = 0

misclassified = []
unconfident = []

img_height, img_width = 450, 450  # 224, 224

def load_models(model_dir):
    models = []

    for file in os.listdir(model_dir):
        if file.endswith(".model"):
            file = os.path.join(model_dir, file)

            print("Loading model and weights: " + file)
            model = load_model(file)
            models.append(model)
            print("Model loaded.")

    return models


def get_prediction_array(models, filepath, number_of_classes):
    print(filepath)

    loaded_img = Image.open(filepath)

    prediction_array = [0] * number_of_classes
    for model_idx in range(len(models)):
        img = loaded_img

        img = img.resize((256, 256))
        img = image.img_to_array(img).astype('float32')
        #img /= 255.
        img = np.expand_dims(img, axis=0)

        model = models[model_idx]
        pa = model.predict(img)[0]

        for i in range(number_of_classes):
            prediction_array[i] += pa[i]

    return prediction_array


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the model file or directory containing several models files.")
    parser.add_argument(
        "--mel_bias",
        type=float,
        help="Diagnose melanoma if its weight reaches this value.")
    parser.add_argument(
        "--mel_idx",
        type=int,
        help="Index of melanoma in the prediction array.")
    parser.add_argument(
        "--image_dir",
        type=str,
        help="Directory to the images to diagnose.")
    parser.add_argument(
        "--preds_path",
        type=str,
        help="Path to the predictions csv.")

    flags, unparsed = parser.parse_known_args()

    model_path = flags.model_path
    mel_bias = flags.mel_bias
    mel_idx = flags.mel_idx
    image_dir = flags.image_dir
    preds_path = flags.preds_path

    for arg in vars(flags):
        print(arg + "='" + str(getattr(flags, arg)) + "'")

    models = []
    if os.path.isdir(model_path):
        models = load_models(model_path)
    else:
        models.append(load_model(model_path))

    f = None
    if preds_path is not None:
        f = open(preds_path, "w")

    i = 1
    for file in os.listdir(image_dir):
        image_path = os.path.join(image_dir, file)

        prediction_array = get_prediction_array(models, image_path, len(class_names))

        pred = 0
        if mel_idx is not None:
            if prediction_array[mel_idx]/num_classes > mel_bias:
                pred = mel_idx
            else:
                pred = np.argmax(prediction_array)
        else:
            pred = np.argmax(prediction_array)

        print(str(i) + ". " + file + "-->" + class_names[pred] + " - " + str(prediction_array))

        if preds_path is not None:
            f.write(str(i) + ", " + file + ", " + class_names[pred] + "," + str(prediction_array[0]) + "," + str(prediction_array[1]) + "\n")

        i += 1