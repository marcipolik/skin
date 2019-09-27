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

## Models instantiated dyanamically, don't delete these imports!!!
from keras.applications import VGG16, VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201
from keras.applications.mobilenet import MobileNet
from keras.applications.nasnet import NASNetLarge, NASNetMobile
## ---------------------------------------------------------------

import argparse
import os
import time
import pickle
import keras
import cv2

import numpy as np

import preprocess.preprocess as pp
import cc

from sklearn.utils import shuffle
from PIL import Image, ImageEnhance

from keras.optimizers import Adam
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K


_IMAGE_NET_PRETRAINED_VALUE = "ImageNet"

_DEFAULT_DATASETS_DIR = "dataset/clinic"
_DEFAULT_IMAGE_SIZE = 256
_DEFAULT_MODEL_NAME = "VGG16"
_DEFAULT_LEARNING_RATE = 1e-5
_DEFAULT_PATIENCE = 10
_DEFAULT_WEIGHTS = _IMAGE_NET_PRETRAINED_VALUE
_DEFAULT_WEIGHTED_CLASSES = False
_DEFAULT_BATCH_SIZE = 10

_TRAIN_SUBDIR = "train"

_SPLIT_TRAIN = 'train'
_SPLIT_VALIDATION = 'validation'
_SPLIT_TEST = 'test'

_SAVED_WEIGHTS_NUM_CLASSES = 7 #ISIC2018

_MODELS_DIR = "models"

_NEW_MODEL_NAME_PREFIX = "new"
_CONV_MODEL_NAME_PREFIX = "con"

_TEST_PRED_FILE_NAME = "test_pred.pckl"
_VALID_PRED_FILE_NAME = "valid_pred.pckl"

num_fc_neurons = 4096
#dropout = 0.5


def add_new_last_layer(base_model, nb_classes):
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(num_fc_neurons, activation='relu')(x)
  x = Dropout(dropout)(x)
  x = Dense(num_fc_neurons, activation='relu')(x)
  x = Dropout(dropout)(x)
  predictions = Dense(nb_classes, activation='softmax')(x)
  model = Model(inputs=base_model.input, outputs=predictions)

  return model


def replace_top_layer(model, new_num_classes):
    #global new_model

    model.layers.pop()
    model.summary()

    new_layer = Dense(new_num_classes, activation='softmax')

    inp = model.input
    out = new_layer(model.layers[-1].output)

    model = Model(inp, out)
    model.summary()

    return model


def load_weight_file(model, weights):
    #global new_model

    if weights is not "" and weights != _IMAGE_NET_PRETRAINED_VALUE:
        print("load_weight_file: " + weights)
        replace_top_layer(model, _SAVED_WEIGHTS_NUM_CLASSES)

        #new_
        model.load_weights(weights)

        replace_top_layer(model, num_classes)


def create_model():
    modelClass = globals()[model_name]
    print("ModelClass: " + str(modelClass))

    if weights is not "":
        if weights == _IMAGE_NET_PRETRAINED_VALUE:
            model = modelClass(
                input_shape=(img_width, img_height, 3),
                include_top=False,
                weights='imagenet')
        else:
            model = modelClass(
                input_shape=(img_width, img_height, 3),
                include_top=False,
                weights=None)
    print("Model instance: " + str(model))

    model = add_new_last_layer(model, num_classes)
    #replace_top_layer(model, num_classes)
    load_weight_file(model, weights)
    print(model.summary())
    print(model.layers[-1].output_shape)

    return model


def load_split_images(split_name):
    assert split_name in [_SPLIT_TRAIN, _SPLIT_VALIDATION, _SPLIT_TEST]

    categories = class_names
    dataset_split_dir = os.path.join(dataset_dir, split_name)

    X, y = [], []

    if os.path.isdir(dataset_split_dir):
        i = 0
        for category in categories:
            category_dir = os.path.join(dataset_split_dir, category)
            for myfile in os.listdir(category_dir):
                img = Image.open(os.path.join(category_dir, myfile))
                img = img.resize((img_width, img_height))

                X.append(np.asarray(img))
                y.append(i)
            i += 1

    return X, y


def load_images():
    X_train, y_train = load_split_images(_SPLIT_TRAIN)
    X_valid, y_valid = load_split_images(_SPLIT_VALIDATION)
    X_test, y_test = load_split_images(_SPLIT_TEST)

    X_train, y_train = shuffle(X_train, y_train)

    if K.image_data_format() == 'channels_first':
        X_train = np.array(X_train).reshape(np.array(X_train).shape[0], 3, img_width, img_height)
        X_valid = np.array(X_valid).reshape(np.array(X_valid).shape[0], 3, img_width, img_height)
        X_test = np.array(X_test).reshape(np.array(X_test).shape[0], 3, img_width, img_height)
        input_shape = (3, img_width, img_height)
    else:
        X_train = np.array(X_train).reshape(np.array(X_train).shape[0], img_width, img_height, 3)
        X_valid = np.array(X_valid).reshape(np.array(X_valid).shape[0], img_width, img_height, 3)
        X_test = np.array(X_test).reshape(np.array(X_test).shape[0], img_width, img_height, 3)
        input_shape = (img_width, img_height, 3)

    X_train = X_train.astype('float32')
    X_valid = X_valid.astype('float32')
    X_test = X_test.astype('float32')

    print('x_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_valid.shape[0], 'valid samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_valid = keras.utils.to_categorical(y_valid, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def get_preprocessor():
    print("pp: '" + img_preprocess + "'")
    assert(img_preprocess == "clahe" or img_preprocess == "d255" or img_preprocess == "no" or img_preprocess == "stretch" or img_preprocess == "sharpen" or img_preprocess == "retinex" or img_preprocess == "contrast2")

    if img_preprocess == "clahe":
        print("clahe!!!!")
        def preprocess(img):
            img = img.astype(np.uint8)
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            lab_planes = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab_planes[0] = clahe.apply(lab_planes[0])
            lab = cv2.merge(lab_planes)
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            img = img.astype('float32')

            return img
        return preprocess
    elif img_preprocess == "d255":
        print("d255!!!!")
        def preprocess(img):
            img = img.astype('float32')
            img /= 255.
            
            return img
        return preprocess
    elif img_preprocess == "sharpen":
        print("sharpen!!!!")
        def preprocess(img):
            img = img.astype(np.uint8)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            fm = cv2.Laplacian(img, cv2.CV_64F).var()

            if fm < 100:
                kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                img = cv2.filter2D(img, -1, kernel)

            img = img.astype('float32')
            return img
        return preprocess
    elif img_preprocess == "stretch":
        print("stretch!!!")
        def preprocess(img):
            img = cc.stretch(img)

            return img
        return preprocess
    elif img_preprocess == "retinex":
        print("retinex!!!!") 
        def preprocess(img):
            img = cc.retinex(img)
        
            img = img.astype('float32')
            return img
        return preprocess
    elif img_preprocess == "contrast2":
        print("contrast2!!!!") 
        def preprocess(img):
            img = cc.from_pil(ImageEnhance.Contrast(cc.to_pil(img)).enhance(2.0))

            img = img.astype('float32')
            return img
        return preprocess
    elif img_preprocess == "no":
        print("no!!!!")
        def preprocess(img):
            return img 
        return preprocess


def train_model(data, model, save_weights_path):
    X_train, y_train, X_valid, y_valid, X_test, y_test = data

    # was val_loss
    checkpointer = keras.callbacks.ModelCheckpoint(filepath=save_weights_path, monitor='val_categorical_accuracy', verbose=1, save_best_only=True)
    # was val_loss
    earlyStopping = keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=patience, verbose=1, mode='auto') 

    train_prep = get_preprocessor()
    valid_prep = get_preprocessor()
    test_prep = get_preprocessor()

    train_datagen = ImageDataGenerator(rotation_range=180, horizontal_flip=True, vertical_flip=True,
                                       fill_mode='constant', preprocessing_function=train_prep)
    valid_datagen = ImageDataGenerator(preprocessing_function=valid_prep)
    test_datagen = ImageDataGenerator(preprocessing_function=test_prep)

    train_datagen.fit(X_train)
    valid_datagen.fit(X_valid)
    test_datagen.fit(X_test)

    class_weight = None
    if weighted_classes:
        class_weight = 'auto'

    history = model.fit_generator(train_datagen.flow(X_train, y_train, batch_size=batch_size),
                                      epochs=max_epochs,
                                      verbose=1,
                                      callbacks=[earlyStopping, checkpointer],
                                      validation_data=valid_datagen.flow(X_valid, y_valid, batch_size=batch_size),
                                      class_weight=class_weight)

    model = load_model(save_weights_path)

    if len(X_test) > 0:
        score = model.evaluate(X_test, y_test, verbose=1)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
    else:
        print("No test set was provided --> Test loss and Test accuracy are not calculated!")

    return history


def test_model(X, model, file_name):
    if len(X) > 0:
        y = model.predict(X)

        f = open(os.path.join(target_dir, file_name), 'wb')
        pickle.dump(y, f)
        f.close()


def do_transfer_learning(model, num_frozen_layers, optimizer):
    i = 0
    for layer in model.layers: #conv_model.layers:
        i += 1
        if i <= num_frozen_layers:
            layer.trainable = False
        else:
            layer.trainable = True

        print('{0}:\t{1}'.format(layer.trainable, layer.name))

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    X_train, y_train, X_valid, y_valid, X_test, y_test = load_images()

    save_weights_path = os.path.join(target_dir, train_result_dir_name + ".model")
    history = train_model(data=(X_train, y_train, X_valid, y_valid, X_test, y_test), model=model, #new_model=new_model,
                          save_weights_path=save_weights_path)

    model.load_weights(save_weights_path)

    test_model(X_valid, model, _VALID_PRED_FILE_NAME)
    test_model(X_test, model, _TEST_PRED_FILE_NAME)



def learn(model):
    num_frozen_layers = 0 #1
    optimizer = Adam(lr=learning_rate)

    do_transfer_learning(model, num_frozen_layers=num_frozen_layers, optimizer=optimizer)


def run():
    os.makedirs(target_dir)

    model = create_model()
    learn(model)


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
        "--dataset_dir",
        type=str,
        default=_DEFAULT_DATASETS_DIR,
        help="Root directory of the training, validation, and test datasets.")
    parser.add_argument(
        "--image_size",
        type=int,
        default=_DEFAULT_IMAGE_SIZE,
        help="Image size.")
    parser.add_argument(
        "--model_name",
        type=str,
        default=_DEFAULT_MODEL_NAME,
        help="Model name.")
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=_DEFAULT_LEARNING_RATE,
        help="Learning rate.")
    parser.add_argument(
        "--patience",
        type=int,
        default=_DEFAULT_PATIENCE,
        help="Patience.")
    parser.add_argument(
        "--weights",
        type=str,
        default=_DEFAULT_WEIGHTS,
        help="'ImageNet' or path to the weights file.")
    parser.add_argument(
        "--weighted_classes",
        type=str2bool,
        default=_DEFAULT_WEIGHTED_CLASSES,
        help="Balance classes with weighting.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=_DEFAULT_BATCH_SIZE,
        help="Batch size.")
    parser.add_argument(
        "--preprocess",
        type=str,
        default="no",
        help="Image preprocessing (training, validation, test).")
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
        help="Dropout rate.")
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=100,
        help="Maximum number of epochs.")


    flags, unparsed = parser.parse_known_args()

    dataset_dir = flags.dataset_dir
    img_height = img_width = flags.image_size
    model_name = flags.model_name
    learning_rate = flags.learning_rate
    patience = flags.patience
    weights = flags.weights
    weighted_classes = flags.weighted_classes
    batch_size = flags.batch_size
    img_preprocess = flags.preprocess
    dropout = flags.dropout
    max_epochs = flags.max_epochs

    print("--------------------------------------------------")
    arg_values = ""
    model_file_suffix = ""
    for arg in vars(flags):
        if arg_values is not "":
            arg_values += ","
        if model_file_suffix is not "" and model_file_suffix[len(model_file_suffix) - 1] != ",":
            model_file_suffix += ","

        if arg is not "dataset_dir":
            arg_values += arg + "=" + str(getattr(flags, arg)).replace("/", "_")
            if arg is "image_size" or arg is "model_name":
                model_file_suffix += arg + "=" + str(getattr(flags, arg))
        print(arg + "='" + str(getattr(flags, arg)) + "'")
    print("--------------------------------------------------")
    train_result_dir_name = time.strftime("%Y%m%d%H%M%S", time.gmtime()) + "," + arg_values
    target_dir = os.path.join(_MODELS_DIR, train_result_dir_name)
    print("target_dir --> " + target_dir)
    print("--------------------------------------------------")
    train_dir = os.path.join(dataset_dir, _TRAIN_SUBDIR)
    class_names = []
    for file in os.listdir(train_dir):
        if os.path.isdir(os.path.join(train_dir, file)):
            class_names.append(file)
    class_names.sort()
    num_classes = len(class_names)
    print("train_dir --> " + train_dir)
    print("num_classes --> " + str(num_classes))
    print("classes_names --> " + str(class_names))
    print("--------------------------------------------------")
    model_file_suffix += "num_classes=" + str(num_classes)
    print("model_file_suffix: " + model_file_suffix)
    print("--------------------------------------------------")

    num_fc_layers = 0
    if model_name is "NASNetLarge":
        num_fc_layers = 1
    if model_name is "VGG16" or model_name is "VGG19":
        num_fc_layers = 2
  


    print("num_fc_layers: " + str(num_fc_layers))
 
    loss = 'categorical_crossentropy'
    metrics = ['categorical_accuracy']

    run()
