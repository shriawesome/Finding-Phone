import sys
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras import layers

def build_model(input_shape):
    """
    Creates and compiles a model
    ------------------------------------
    Parameters:
    input_shape  : input shape of the image

    Returns:
    model : compiled inception v3 model
    ------------------------------------
    """
    # Loading the inception model and setting parameter values
    inception = tf.keras.applications.InceptionV3(weights='imagenet',include_top=False, 
                                              input_shape=input_shape)
    inception.trainable = True
    last_layer = inception.get_layer('mixed10')
    layer_output = last_layer.output
    
    # Dense model architecture
    x1 = layers.Flatten()(layer_output)
    x1 = layers.Dropout(0.3)(x1)
    x1 = layers.Dense(4112, activation="relu")(x1)
    x1 = layers.Dense(1028, activation="relu")(x1)
    x1 = layers.Dense(1028, activation="relu")(x1)
    x1 = layers.Dropout(0.5)(x1)
    x1 = layers.Dense(256, activation="relu")(x1)
    coordinates = layers.Dense(2, name="coordinates")(x1)

    model = tf.keras.Model(inputs=inception.inputs, outputs=coordinates)
    
    # Compiling the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
             loss = {'coordinates':'mse'},
             metrics={'coordinates':'mse'})
    
    return model

def scheduler(epoch, lr):
    """
    Applies learning rate decay
    ------------------------------------
    Parameters:
    epoch : training epoch number
    lr    : Learning rate 

    Returns:
    lr : modified lr rate
    ------------------------------------
    """
    if epoch<40:
        return lr
    else:
        return lr*tf.math.exp(-0.12)

def split_data(x,y,test_frac=0.3):
    """
    Split the data into training and testing
    ------------------------------------
    Parameters:
    x         : Input images data
    y         : Labels
    test_frac : fractional split (default = 0.3)

    Returns:
    x_train, y_train, x_test, y_test
    ------------------------------------
    """
    n = x.shape[0]
    shuffle_idx = np.random.permutation(range(n))
    idx = int(n*(1-test_frac))
    train_idx, test_idx = shuffle_idx[:idx], shuffle_idx[idx:]
    
    return x[train_idx],y[train_idx],x[test_idx],y[test_idx]