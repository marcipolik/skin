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

import base64
import io
import json
import keras.preprocessing.image as image
import numpy as np
import os
from PIL import Image
from PIL.ExifTags import TAGS
from keras.models import load_model as load_keras_model

models = dict()
labels = dict()


def _model_predict(model_name, pil_img):
    model = models[model_name]
    model_labels = labels[model_name]
    
    image_size = int(model.input.shape[1])                 

    img = pil_img.resize((image_size, image_size))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    prediction_array = model.predict(x)[0]    
    
    pred = np.argmax(prediction_array)
    prediction = model_labels[str(pred)]    

    confidence = prediction_array[pred]    

    return prediction_array, prediction, confidence
    
    
def predict(pil_img):
    model_labels = labels['']
        
    normal_prediction_array, _, _ = _model_predict('', pil_img)

    prediction_array = normal_prediction_array

    normal_pred = np.argmax(normal_prediction_array)       
                        
    pred = np.argmax(prediction_array)
    prediction = model_labels[str(pred)]
    confidence = prediction_array[pred]
                        
    print('Predicted:', prediction + "; " + str(confidence) + "; " + str(prediction_array))

    return model_labels, prediction_array, prediction, confidence


def load_models(models_root):    
    global models
    global labels
    
    model_dir = models_root
    print("\nDIR: " + model_dir)

    for file in os.listdir(model_dir):
        if file.endswith(".model"):
            model_name = file[:-6]
            print("\nModel name: " + model_name)
                    
            file = os.path.join(model_dir, file)

            print("Loading model and weights: " + file)
            model = load_keras_model(file)
            print("Model loaded.")

            models[model_name] = model
        
            file = file[:-6] + ".labels.json"

            print("Loading labels: " + file)
            with open(file) as fp:
                labels[model_name] = json.load(fp)
            print("Labels loaded.")
            

if __name__ == "__main__":
    load_models('models')

    test_prediction()


