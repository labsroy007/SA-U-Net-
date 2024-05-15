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





# Performing various operations on some of the input_arr images

def preprocess(input_arr):

  for i in range(len(input_arr)):
    x1 = np.random.randint(0,7)

    # Adding random noise
    if x1==0:
      x2 = np.random.randint(0,2)
      x3 = np.random.randint(0, 30, input_arr[i].shape, dtype = np.uint8)

      if x2==0:
        input_arr[i] = cv.subtract(input_arr[i], x3)
      else:
        input_arr[i] = cv.add(input_arr[i], x3)

    # Adding blur
    elif x1==1:
      input_arr[i] = cv.GaussianBlur(input_arr[i], (3, 3), 0)

    # Changing brightness randomly
    elif x1==2:
      y1 = np.random.randint(30, 50)
      y2 = np.array(input_arr[i], dtype=np.int16)
      y3 = np.random.randint(0,2)

      if y3==0:
        y2 = y2 + y1
        y2 = y2*(255/max(np.reshape(y2, (-1))))

      else:
        y4 = min(np.reshape(y2, (-1))) - y1
        if y4<0:
          y1 = y1 + y4
        y2 = y2-y1

      input_arr[i] = np.array(y2, dtype=np.uint8)

  input_arr = np.array(input_arr, dtype=np.float16)
  input_arr = input_arr/128

  return input_arr

