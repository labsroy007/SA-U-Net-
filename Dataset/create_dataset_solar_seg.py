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





# Function to divide a large 1024x0124 image into 'count' 256x256 images, and return the same
# Return those partitions which have solar panels in them, rather than any random partition
# f=0 makes the function return partitions, irrespective of them having a solar panel or not
# otherwise, return only those partitions, which have solar panels

def get_partitions(input_arr, output, count, f):
  maxpool = layers.MaxPooling2D(pool_size=(4,4), strides=(4,4), padding='valid')
  small_out = cv.resize(output, (16,16), interpolation=cv.INTER_AREA)

  out_arr = []
  in_arr = []

  # Resize the output image to 16x16, then perform maxpool operation with filter size and stride 4x4
  # The output of this maxpool operation will be a 4x4 matrix, each element of which will correspond
  # to a 256x256 non-overlapping partition of the original image
  # Thus each element of this output will have 0 if partitions have (approx) no solar panel,
  # otherwise will have a non-zero value

  small_out = np.reshape(small_out, (1,16,16,1))
  max_out = np.reshape(maxpool(small_out).numpy(), (4,4))
  c = 0

  for k in range(4):
    for l in range(4):
      if (not f) or max_out[k, l]==1:
        if c==count:
          break

        k1 = k*256
        l1 = l*256

        print(k1, k1+256)
        out_arr.append(output[k1:k1+256, l1:l1+256])
        in_arr.append(input_arr[k1:k1+256, l1:l1+256])
        c += 1

  return in_arr, out_arr


# Function to return the list of input_arr and output dataset for training the model
# 'path' - path to the folder containing the data
# 'prob' - for determining 'f', to pass to the get_partitions function
# 'count' - passing it to the get_partitions function
# the last two parameters are required if the input_arr image needs partition

def get_dataset(path, prob=-1, count=-1):

  p = path
  l = os.listdir(p)

  inp = []
  out = []
  sparse = []

  for i in l:
    sp = p+i+'/'
    sl = os.listdir(sp)
    print(len(sl))

    for j in sl:
      if j[-10:]=='_label.bmp':
        continue

      output = cv.imread(sp+j[:-4]+'_label.bmp')


      output = cv.cvtColor(output, cv.COLOR_BGR2GRAY)
      output = np.where(output>0, 1, 0)
      output = np.array(output, dtype=np.uint8)



      if not count==-1:
        a = np.count_nonzero(output == 1)
        ab = a/(1024*1024)
        sparse.append(ab)


      else:
        a = np.count_nonzero(output == 1)
        ab = a/(256*256)
        sparse.append(ab)

  return sparse


