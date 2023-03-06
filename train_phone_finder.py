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

from utils import build_model, split_data, scheduler
    
def get_imgs(folder):
    """
    Opens the folder and extracts images and labels.
    ------------------------------------
    Parameters:
    folder  : Input folder for extracting images

    Returns:
    x : Numpy array with images
    y : Respective x, y coordinates 
    ------------------------------------
    """
    metadata = []
    
    try:
        with open(folder+os.sep+"labels.txt" ,"r") as f:
            data = f.readline()
            while data:
                metadata.append(data.strip().split(" "))
                data = f.readline()
                
    except:
        print("train_phone_finder.py -> get_imgs() : labels.txt not found or incorrect format")
        sys.exit(0)
        
    if len(metadata) == 0:
        print("train_phone_finder.py -> get_imgs() : No data found in labels.txt")
        sys.exit(0)
        
    # Loading the images in numpy array
    x, y = np.zeros((len(metadata),326,490,3)), np.empty((len(metadata),2))
    for i,data in enumerate(metadata):
        img = mpimg.imread(folder+os.sep+data[0])
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img/255.
        img_gray = img_gray/255.
        x[i,:,:,:] = img
        y[i,0], y[i,1] = np.float(data[1]), np.float(data[2])
    
    
    return x, y

def scale_y(y):
    """
    Get the absolute values for x, y cordinate.
    ------------------------------------
    Parameters:
    y  : Input coordinates

    Returns:
    y_scaled : scaled x,y coordinates
    ------------------------------------
    """
    y_scaled = y.copy()
    y_scaled[:,0] = 490*y_scaled[:,0]
    y_scaled[:,1] = 326*y_scaled[:,1]
    return y_scaled

def train_model(x, y, save_plot=True):
    """
    Performs transfer learning using InceptionV3.
    ------------------------------------
    Parameters:
    x         : Images to be used for training
    y         : Scaled x,y coordinates
    save_plot : Saves the MSE plots after training
    

    Returns:
    ------------------------------------
    """
    # Build the Model
    print("Building the InceptionV3 Model")
    input_shape = x.shape[1:]
    model = build_model(input_shape)
    
    # Split the data into training and testing(70% - 30%)
    x_train, y_train, x_test, y_test = split_data(x,y)
    print(f'x_train : {x_train.shape}, y_train: {y_train.shape}, x_test: {x_test.shape}, y_test: {y_test.shape}')
    
    # Saving the best model
    if not os.path.isdir("model"):
        os.mkdir("model/")
        
    save_path = "model/{epoch:02d}-{val_mse:.2f}.h5"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath = save_path,
                                                 save_weights_only=True,
                                                 monitor='val_mse',
                                                 mode='min',
                                                 save_best_only=True)
    
    # Applying learnig rate decay after 40 epochs
    lr_decay_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    
    # Training the model
    n_epochs = 70
    hist = model.fit(x_train, y_train,
                    epochs=n_epochs,
                    batch_size=8,
                    validation_data=(x_test,y_test),
                    callbacks=[lr_decay_callback, cp_callback])
    
    if save_plot:
        train_mse, val_mse = hist.history['mse'], hist.history['val_mse']
        plt.xlabel('Iteration')
        plt.ylabel('MSE')
        plt.plot(range(n_epochs), train_mse, label="train_MSE")
        plt.plot(range(n_epochs), val_mse, label="val_MSE")
        plt.title("Train and Val MSE")
        plt.savefig('MSE_plot.png')
        
        
    

if __name__=="__main__":
    # Extracing folder name, expecting folder name as 1st argument
    if len(sys.argv)!=2:
        print("Invalid Arguments\n Try: python train_phone_finder.py <folder_name>")
        sys.exit(0)
    else:
        folder_name = sys.argv[1]
        
    # Sets the first GPU visible for processing
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
      # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.set_visible_devices(gpus[0], 'GPU')
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)
            
    
    # Get the data
    print("Loading Data from labels.txt")
    x, y = get_imgs(folder_name)
    print(f'X Shape: {x.shape}, Y Shape: {y.shape}')

    # Scaling Y vals
    y_scaled = scale_y(y)
   
    # Training the model
    train_model(x, y_scaled)