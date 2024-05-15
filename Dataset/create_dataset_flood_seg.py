import numpy as np
import cv2 as cv
import time
from google.colab.patches import cv2_imshow
import os
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras import models, layers, callbacks
import tensorflow as tf

from tensorflow.keras import backend as K
import keras



# Function to return the list of input_arr and output dataset for training the model
# 'path' - path to the folder containing the data

def get_dataset(path):

  p = path
  l = os.listdir(p)

  inp = []
  out = []

  for i in l:
    if i[-4:]=='.png':
      continue

    ip = p + i
    op = p + i[:-4] + '.png'

    input_arr = cv.imread(ip)
    output = cv.imread(op)

    output = cv.cvtColor(output, cv.COLOR_BGR2GRAY)
    output = np.where(output>0, 1, 0)
    output = np.array(output, dtype=np.uint8)

    size = 256
    sh = output.shape
    h, w = int(sh[0]/size), int(sh[1]/size)

    in_arr = []
    out_arr = []

    for k in range(h):
      for l in range(w):

        k1 = k * size
        l1 = l * size

        if not 1 in output[k1:k1+size, l1:l1+size]:
          continue

        out_arr.append(output[k1:k1+size, l1:l1+size])
        in_arr.append(input_arr[k1:k1+size, l1:l1+size])

    inp.extend(in_arr)
    out.extend(out_arr)

  return inp, out
