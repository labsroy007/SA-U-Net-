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
# Enter the dataset path of input images in 'inp_path' and their corresponding segmentation masks path in 'out_path'

def get_dataset(inp_path, out_path):
  l = os.listdir(inp_path)

  input_arr = []
  output = []
  sparse = []

  for i in l:
    ip = inp_path + i
    op = out_path + i[:-4] + '_lab.png'

    inp = cv.imread(ip)
    inp = cv.resize(inp, (512, 512), interpolation=cv.INTER_AREA)

    out = cv.imread(op)
    out = cv.cvtColor(out, cv.COLOR_BGR2GRAY)
    out = cv.resize(out, (512, 512), interpolation=cv.INTER_NEAREST)

    inp1, inp2, inp3, inp4 = inp[0:256, 0:256], inp[256:512, 0:256], inp[0:256, 256:512], inp[256:512, 256:512]
    out1, out2, out3, out4 = out[0:256, 0:256], out[256:512, 0:256], out[0:256, 256:512], out[256:512, 256:512]

    inp = []
    out = []

    inp.append(inp1)
    inp.append(inp2)
    inp.append(inp3)
    inp.append(inp4)

    out.append(out1)
    out.append(out2)
    out.append(out3)
    out.append(out4)

    input_arr.extend(inp)
    output.extend(out)

  return input_arr, output

