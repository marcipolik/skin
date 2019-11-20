# Skin  

Skin is a neural network classifier for skin lesions.

## Requirements

To use Skin, you will need the following libraries installed:

* pickle
* cv2
* sklearn
* numpy
* gevent
* gunicorn
* keras == 2.1.4
* h5py
* pillow
* tensorflow == 1.5
* scikit-image

Skin uses Python3.

## Installation

You can download Skin from GitHub.
```git
git clone https://github.com/atomrom/skin.git
```
## Preprocess data

To use ISIC dataset you should download input data, metadata and groundtruth from the ISIC website (2019 version: https://challenge2019.isic-archive.com/data.html)
Files:

* ISIC_2019_Training_Input.zip
* ISIC_2019_Training_Metadata.csv
* ISIC_2019_Training_GroundTruth.csv

First edit the /train/dataset/isic_to_dirs.py and add the isic files path and the desired target directory.

After the script unpacked the images to separate folders you should run create_validation_and_test_set.py.

The parameters are:
* --dry_run : If True it does not copy the files. Default value is False.
* --dataset_dir : Root directory of the training, validation, and test datasets. Default value is 'clinic'
* --source_dir : Root directory of dataset to be split. See the step above. 
* --validation_percent : Percent of the validation set. Default value is 10 percent.
* --test_percent : Percent of the validation set. Default value is 10 percent.

It creates the training, test and validation set from the data. 

## Usage

### Train:
Parameters:
* --dataset_dir : Root directory of the training, validation, and test datasets. (See steps above)
* --image_size : Image size.
* --model_name : Model name
* --learning_rate : Learning rate
* --patience : Patience
* --weights : 'ImageNet' or path to the weights file.
* --weighted_classes : Balance classes with weighting.
* --batch_size : Batch size.
* --preprocess : Image preprocessing (training, validation, test).
* --dropout : Dropout rate.
* --max_epochs : Maximum number of epochs.

All of the parameters have default value, so all of them are optional. You can check the default parameters in train/train.py

```bash
python train/train.py [parameters]
```

### Predict

```bash
python predictor/predict.py 
```

## Summary


## License

Copyright (c) 2019 Attila Ulbert

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
