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




def self_attention(inp_layer, num_filters):

  layer_1 = layers.Conv2D(num_filters, (1, 1), padding="same")(inp_layer)
  layer_2 = layers.Conv2D(num_filters, (1, 1), padding="same")(inp_layer)
  layer_3 = layers.Conv2D(num_filters, (1, 1), padding="same")(inp_layer)

  mul = layers.add([layer_1, layer_2])

  sig = layers.Activation('sigmoid')(mul)
  mult = layers.multiply([layer_3, sig])

  out_layer = layers.Conv2D(num_filters, (1, 1), padding="same")(mult)

  return out_layer


def unet_pp_mid(num_filters, *layer_names):
  layer_names = list(layer_names)
  new_layer = layers.UpSampling2D(size=(2, 2))(layer_names[-1])
  layer_names.pop()
  layer_names.append(new_layer)

  conc = layers.concatenate(layer_names)
  outp = layers.Conv2D(num_filters, (3, 3), activation="relu", padding="same")(conc)

  return outp


# num_out_neuron specifies the number of output layer neurons required, and is dependent on the task at hand
def SA_UNetpp(num_out_neuron):
  start_neurons = 32
  input_arr_layer = layers.Input((256, 256, 3))
  
  
  # Downsampling
  
  conv1 = layers.Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(input_arr_layer)
  pool1 = layers.MaxPooling2D((2, 2))(conv1)
  # pool1 = layers.Dropout(0.25)(pool1)
  
  conv2 = layers.Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(pool1)
  pool2 = layers.MaxPooling2D((2, 2))(conv2)
  # pool2 = layers.Dropout(0.2)(pool2)
  
  conv3 = layers.Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(pool2)
  pool3 = layers.MaxPooling2D((2, 2))(conv3)
  pool3 = layers.Dropout(0.2)(pool3)
  
  conv4 = layers.Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(pool3)
  pool4 = layers.MaxPooling2D((2, 2))(conv4)
  pool4 = layers.Dropout(0.2)(pool4)
  
  
  # Middle
  
  convm = layers.Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(pool4)
  
  mid_01 = unet_pp_mid(start_neurons * 1, conv1, conv2)
  mid_11 = unet_pp_mid(start_neurons * 2, conv2, conv3)
  mid_21 = unet_pp_mid(start_neurons * 4, conv3, conv4)
  
  mid_02 = unet_pp_mid(start_neurons * 1, conv1, mid_01, mid_11)
  mid_12 = unet_pp_mid(start_neurons * 2, conv2, mid_11, mid_21)
  
  mid_03 = unet_pp_mid(start_neurons * 1, conv1, mid_01, mid_02, mid_12)
  
  
  # Upsampling
  
  uconv4 = unet_pp_mid(start_neurons * 8, conv4, convm)
  uconv4 = layers.Dropout(0.2)(uconv4)
  uconv4 = self_attention(uconv4, start_neurons * 8)
  
  uconv3 = unet_pp_mid(start_neurons * 4, conv3, mid_21, uconv4)
  uconv3 = layers.Dropout(0.2)(uconv3)
  uconv3 = self_attention(uconv3, start_neurons * 4)
  
  uconv2 = unet_pp_mid(start_neurons * 2, conv2, mid_11, mid_12, uconv3)
  # uconv2 = layers.Dropout(0.2)(uconv2)
  uconv2 = self_attention(uconv2, start_neurons * 2)
  
  uconv1 = unet_pp_mid(start_neurons * 1, conv1, mid_01, mid_02, mid_03, uconv2)
  # uconv1 = layers.Dropout(0.5)(uconv1)
  uconv1 = self_attention(uconv1, start_neurons * 1)
  
  output_layer = layers.Conv2D(num_out_neuron, (1, 1), padding="same", activation="sigmoid")(uconv1)
  
  model = models.Model(inputs=input_arr_layer, outputs=output_layer)
