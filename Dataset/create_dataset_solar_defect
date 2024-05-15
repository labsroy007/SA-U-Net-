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


# Pass the location of the entire dataset, and this function will return the dataset as an array in the required format
def get_dataset(p):
  z = ['broken/', 'foreign_body/', 'miss/']
  
  input_arr = []
  output = []
  sparse = []
  
  for i in range(3):
    l = os.listdir(p + 'test/' + z[i])
  
    for j in l:
      img = cv.imread(p + 'test/' + z[i] + j) 
  
      mask = cv.imread(p + 'ground_truth/' + z[i] + j[:-4] + '_mask.png')
      mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
      mask = np.where(mask==255, i+1, 0)
      mask = np.array(mask, dtype=np.uint8)
  
      mask = cv.resize(mask, (256,256), interpolation=cv.INTER_NEAREST)
      img = cv.resize(img, (256,256), interpolation=cv.INTER_AREA)
  
      input_arr.append(img)
      output.append(mask)  


  p1 = p+'/test/good/'
  l = os.listdir(p1)
  mask = np.zeros((256, 256), dtype=np.uint8)
  
  for i in range(500):
    img = cv.imread(p1 + l[5*i])   # len of l = 2500
    img = cv.resize(img, (256,256), interpolation=cv.INTER_AREA)
  
    input_arr.append(img)
    output.append(mask)

  return input_arr, output

  
