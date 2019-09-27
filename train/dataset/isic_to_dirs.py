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
import csv
import zipfile
import shutil
import numpy as np

from PIL import Image
from PIL import ImageOps

from sklearn.utils import shuffle

_TARGET_DIR = 'isic'

header_columns = []

with open('isic/ISIC2018_Task3_Training_GroundTruth.csv', 'r') as csvfile:
  csvreader = csv.reader(csvfile)
  for row in csvreader:
    if len(header_columns) == 0:
      header_columns = row
      
      for i in range(1, len(header_columns)):
        category = header_columns[i]
        category_dir = os.path.join(_TARGET_DIR, category)
        if not os.path.exists(category_dir):
          print("Mkdir: ", category)
          
          os.mkdir(category_dir)
    else:
      file_name = row[0] + ".jpg"
      category = header_columns[row.index('1.0')]
          
      with zipfile.ZipFile('isic/ISIC2018_Task3_Training_Input.zip') as zip:
        with zip.open("ISIC2018_Task3_Training_Input/" + file_name) as file_in_zip, open(os.path.join(_TARGET_DIR, category, category + "_" + file_name), 'wb') as target_file:
          print(category, "->", _TARGET_DIR, category, "/" + category + "_" + file_name)
          
          shutil.copyfileobj(file_in_zip, target_file)

          