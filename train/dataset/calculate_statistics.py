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

batch_size = 16

num_classes = 7
class_names = []
not_diagnosed_count = 0

misclassified = []
unconfident = []

img_height, img_width = 450, 450  # 224, 224


def predict(model, validation_dir):
    sample_counts = [0, 0, 0, 0, 0, 0, 0]
    hit_counts = [0, 0, 0, 0, 0, 0, 0]

    truths = []
    predictions = []
    prediction_arrays = []

    index = 0
    for r, dirs, files in os.walk(validation_dir):
        for dr in dirs:
            print(index, dr)
            files_in_dir = glob.glob(os.path.join(r, dr + "/*"))
            for fid in files_in_dir:
                sample_counts[index] += 1
                print(fid)

                img = image.load_img(path=fid, target_size=(img_width, img_height))
                img = image.img_to_array(img).astype('float32')
                img = preprocess_input(img)
                img -= np.mean(img, keepdims=True)
                img /= (np.std(img, keepdims=True) + K.epsilon())

                img = np.expand_dims(img, axis=0)

                prediction_array = model.predict(img)[0]
                prediction_arrays.append(prediction_array)

                pred = np.argmax(prediction_array)

                if index == pred:
                    hit_counts[index] += 1

                print('Accuracy:', sample_counts, hit_counts, np.sum(hit_counts) / np.sum(sample_counts))

                truths.append(index)
                predictions.append(pred)

                # cnt += len(glob.glob(os.path.join(r, dr + "/*")))

            index = index + 1

    return sample_counts, hit_counts, truths, predictions, prediction_arrays


def calculate_statistics_from_models(pckl_path, image_dir):
    global class_names
    global not_diagnosed_count

    truths = []
    predictions = []

    prediction_indexes = []

    sample_counts = []
    hit_counts = []

    models = []
    if os.path.isdir(model_path):
        models = pred.load_models(model_path)
    else:
        models.append(load_model(model_path))

    for file in os.listdir(image_dir):
        image_path = os.path.join(image_dir, file)

        prediction_array = pred.get_prediction_array(models, image_path, len(class_names))
        prediction_array /= len(class_names)




    return sample_counts, hit_counts, truths, predictions


def calculate_statistics_from_pckl(pckl_path, image_dir):
    global class_names
    global not_diagnosed_count

    truths = []
    predictions = []

    prediction_indexes = []

    sample_counts = []
    hit_counts = []

    pckl_count = 0
    pckls = []
    if os.path.isdir(pckl_path):
        for file in os.listdir(pckl_path):
            if file.endswith(".pckl"):
                pckl_file = os.path.join(pckl_path, file)
                print(str(pckl_file))
                with open(pckl_file, 'rb') as f:
                    prediction_pckl = pickle.load(f)

                    if pckl_count == 0:
                        pckls = prediction_pckl
                    else:
                        pckls += prediction_pckl

                    pckl_count += 1

                    print(str(prediction_pckl))

        pckls /= pckl_count
    else:
        with open(pckl_path, 'rb') as f:
            pckls = pickle.load(f)

    print(str(pckls))

    category_names = []
    prediction_pckl = pckls
    class_idx = 0
    prediction_pckl_idx = 0
    for class_name in os.listdir(image_dir):
        if os.path.isdir(os.path.join(image_dir, class_name)):
            class_names.append(class_name)

            truth = class_name
            sample_count = 0
            hit_count = 0
            category_dir = os.path.join(image_dir, class_name)
            for img_file in os.listdir(category_dir):
                category_names.append(class_name)


                # pred = 0
                # pred_array = prediction_pckl[prediction_pckl_idx]
                #
                # m = False
                # for m_idx in range(7, 12):
                #     m = m or pred_array[m_idx] > 0.0
                #
                # if m:
                #     pred = np.argmax(prediction_pckl[prediction_pckl_idx][7:12])+7
                # else:
                #     pred = np.argmax(prediction_pckl[prediction_pckl_idx])

                prediction_array = prediction_pckl[prediction_pckl_idx]
                prediction_pckl_idx += 1

                pred = 0
                # if prediction_array[7] > 0.1:
                #     pred = 7
                # else:
                pred = np.argmax(prediction_array)
                confidence = prediction_array[pred]

                if min_confidence is not None:
                    if 100*confidence < min_confidence:
                        if discard_none:
                            unconfident.append((img_file, class_idx, pred, confidence))
                        else:
                            unconfident.append((img_file, class_idx + 1, pred + 1, confidence))

                        not_diagnosed_count += 1
                        pred = 0

                    elif not discard_none:
                        pred += 1 # 0th element of the array == None, classes are indexed from 1
                elif not discard_none:
                    # 0th element of the array == None, classes are indexed from 1
                    #
                    # this user setting makes no sense here, since there won't be None diagnosis,
                    # but we set it anyway to have correct calculations
                    pred += 1

                print(str(class_idx) + " - " + str(pred))

                if discard_none and pred == 0:
                    continue

                sample_count += 1
                truths.append(truth)
                prediction_indexes.append(pred)

                hit = False
                expected_pred = class_idx
                if discard_none:
                    if class_idx == pred:
                        hit = True
                else:
                    expected_pred = class_idx + 1
                    if class_idx + 1 == pred:
                        hit = True

                if hit:
                    hit_count += 1
                else:
                    misclassified.append((img_file, expected_pred, pred, confidence))

            sample_counts.append(sample_count)
            hit_counts.append(hit_count)

            class_idx += 1

    for idx in prediction_indexes:
        predictions.append(class_names[idx])

    return sample_counts, hit_counts, truths, predictions


def calculate_statistics_from_diagnosis_file(diagnosis_file):
    global class_names
    global not_diagnosed_count

    sample_count_dict = {}
    hit_count_dict = {}

    truths = []
    predictions = []

    with open(diagnosis_file) as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')

        for row in reader:
            supposed = row[2]
            histology = row[3]

            print("supposed="+supposed+", histology="+histology)
            if histology == "None":
                print("Skip... (missing histology)")
                continue

            print(supposed != "None")
            if supposed == "None":
                not_diagnosed_count += 1
                print("Not diagnosed")

                if discard_none:
                    print("Skip... (missing diagnosis)")
                    continue

            if histology not in sample_count_dict:
                sample_count_dict[histology] = 0
            sample_count_dict[histology] += 1

            # if supposed is "None":
            #     supposed = "-"

            predictions.append(supposed)
            truths.append(histology)

            if histology == supposed:
                if histology not in hit_count_dict:
                    hit_count_dict[histology] = 0
                hit_count_dict[histology] += 1


    print("-----------")

    sample_counts = []
    hit_counts = []
    tn_counts = []

    for key, value in sample_count_dict.items():
        class_names.append(key)

        sample_counts.append(sample_count_dict[key])
        if key in hit_count_dict:
            hit_counts.append(hit_count_dict[key])
        else:
            hit_counts.append(0)

    print("scd:"+str(sample_count_dict))
    print("sc:"+str(sample_counts))

    print("hcd:"+str(hit_count_dict))
    print("hc:"+str(hit_counts))

    print("class_names:" + str(class_names))

    print("-----------")

    return sample_counts, hit_counts, truths, predictions


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Histology')
    plt.xlabel('Supposed diagnosis')


def plot_roc(truths, prediction_arrays):
    truths = keras.utils.to_categorical(truths, num_classes)
    prediction_arrays = np.array(prediction_arrays)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(truths[:, i], prediction_arrays[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(truths.ravel(), prediction_arrays.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= num_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='blue', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    lw = 2

    colors = cycle(['red', 'yellow', 'green', 'blue', 'magenta', 'black', 'gray'])
    class_names = cycle(['AKIEC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC'])
    for i, color, class_name in zip(range(num_classes), colors, class_names):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve for {0}'''.format(class_name))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating characteristic curve for skin lesion prediction')
    plt.legend(loc="lower right")
    # plt.show()


def get_roc_auc_score(truths, prediction_arrays):
    truths = keras.utils.to_categorical(truths, num_classes)
    prediction_arrays = np.array(prediction_arrays)

    for i in range(num_classes):
        two_class_y_valid = truths[:, i]  # [value[0] for value in truths]
        two_class_y_valid_pred = prediction_arrays[:, i]  # [value[0] for value in y_valid_pred]

        two_class_y_valid = np.array(two_class_y_valid)
        two_class_y_valid_pred = np.array(two_class_y_valid_pred)

        mel_vs_rest_score = roc_auc_score(two_class_y_valid, two_class_y_valid_pred)

        print("Valid vs rest AUC: ", i, str(mel_vs_rest_score))



def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def write_suspicious(csvfile, kind, items):
    for item in items:
        csvfile.write(kind + ", " + item[0] + ", " +
                      class_names[item[1]] + ", " + class_names[item[2]] + ", " + str(item[3]) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--suspicious_file_path",
        type=str,
        help="Path the CSV file of suspicious cases.")
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the model file or directory containing several models files.")
    parser.add_argument(
        "--discard_none",
        type=str2bool,
        default=False,
        help="Discard from stats those cases when there is no supposed diagnosis.")
    parser.add_argument(
        "--min_confidence",
        type=int,
        help="Min confidence in percent.")
    parser.add_argument(
        "--pckl_path",
        type=str,
        help="Path to a pckl file or directory of pckl files with predictions.")
    parser.add_argument(
        "--image_dir",
        type=str,
        help="Directory to the images per category.")

    flags, unparsed = parser.parse_known_args()

    suspicious_file_path = flags.suspicious_file_path
    model_path = flags.model_path
    min_confidence = flags.min_confidence
    pckl_path = flags.pckl_path
    image_dir = flags.image_dir
    discard_none = flags.discard_none

    if not discard_none:
        class_names.append(str(None))

    print("discard_none:" + str(discard_none))
    if min_confidence is not None:
        print("min_cofidence:" + str(min_confidence))

    if model_path is not None:
        print("model_path:" + model_path)

        sample_counts, hit_counts, truths, preds = calculate_statistics_from_models(model_path, image_dir)
    elif pckl_path is not None:
        print("pckl_path:" + pckl_path)
        print("image_dir:" + image_dir)

        sample_counts, hit_counts, truths, preds = calculate_statistics_from_pckl(pckl_path, image_dir)
    else:
        sample_counts, hit_counts, truths, preds = calculate_statistics_from_diagnosis_file("diags.csv")

    # write misclassified and unconfident
    if suspicious_file_path is not None:
        with open(suspicious_file_path, 'w') as csvfile:
            write_suspicious(csvfile, 'Misclassified', misclassified)
            write_suspicious(csvfile, 'Unconfident', unconfident)


    print("------------------------------")
    print("Truths:", truths)
    print("Predictions:", preds)
    print('Accuracy:', sample_counts, hit_counts, np.sum(hit_counts) / np.sum(sample_counts))

    print("------------------------------")
    print(classification_report(truths, preds, target_names=class_names))

    # Calculate score
    #get_roc_auc_score(truths, prediction_arrays)

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(truths, preds, labels=class_names)
    np.set_printoptions(precision=2)

    cm = np.array(cnf_matrix)
    last_col_idx = len(cnf_matrix) - 1

    print("------------------------------")
    total_sum = np.sum(cnf_matrix)
    print("not diagnosed=" + str(not_diagnosed_count) + " ("+str(100*(not_diagnosed_count/total_sum))+"%)")

    for i in range(len(cnf_matrix)):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        # if discard_none:
        #     fn -= cm[i, 0]
        tn = total_sum - tp - fp - fn

        print(class_names[i] + ":")
        print(tp)
        print(fp)
        print(fn)
        print(tn)
        print("  accuracy    = " + str((tp + tn) / (tp + tn + fp + fn)))

        print("  sensitivity = "+str(tp/(tp+fn)))
        print("  specificity = "+str(tn/(tn+fp)))
        print("  precision   = "+str(tp/(tp+fp)))

        print("  f1 score    = "+str((2*tp)/(2*tp+fp+fn)))

    print("------------------------------")


    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')

    # Plot ROC
    #plot_roc(truths, prediction_arrays)

    plt.show()
