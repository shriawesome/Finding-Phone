import sys
import os
import matplotlib.image as mpimg
import numpy as np

from utils import build_model 
import warnings
warnings.filterwarnings('ignore')

def get_img(img_path):
    """
    Opens the image and returns the numpy array
    ------------------------------------
    Parameters:
    img_path : Images path

    Returns:
    img : numpy array
    ------------------------------------
    """
    
    try:
        return mpimg.imread(img_path)/255.
    except:
        print(f"find_phone -> get_img() : Error opening the {img_path} image, check the path")
        sys.exit(0)
        
def get_model(img, path=None):
    """
    Returns the model with trained weights
    ------------------------------------
    Parameters:
    img  : Input Image
    path : Path to load the model

    Returns:
    model : trained model
    ------------------------------------
    """    
    # Load the untrained model
    input_shape = img.shape
    model = build_model(input_shape)
    
    if not path:
        # Change the path to load the best model from desired folder name
        model_folder = "models"
        if not os.path.isdir(model_folder):
            print(f"find_phone -> get_model() : 'models' directory not found for loading the models")
            sys.exit(0)
        else:
            best_model_path = get_path(model_folder)
            model.load_weights(best_model_path)
    else:
        model.load_weights(path)
        
    return model

def get_path(folder):
    """
    Returns the path for the model with lest val_mse
    ------------------------------------
    Parameters:
    folder  : Folder with all the saved models

    Returns:
    path : saved model path
    ------------------------------------
    """  
    files = os.listdir(folder)
    file_names = []
    minval = float('INF')
    print(files)
    for file in files:
        try:
            val = float(file.split("-")[1].split('.h5')[0])
        except:
            continue
        if val < minval:
            path = file
            minval = val
            
    path = folder + os.sep + path
    print("Best Model Path: ",path)
    
        
    return path

def get_predictions(img, model):
    """
    Makes predictions and returns the normalized x, y coordinates
    ------------------------------------
    Parameters:
    folder  : Folder with all the saved models

    Returns:
    path : saved model path
    ------------------------------------
    """  
    # Make predictions
    img_ = np.expand_dims(img, axis=0)
    y_pred = model.predict(img_)
    
    # Rescaling the outputs
    x, y = y_pred[0][0]/490, y_pred[0][0]/326
    
    return x, y

if __name__ == "__main__":
    # get the image path
    if len(sys.argv)!=2:
        print("Invalid Arguments\n Try: python find_phone.py <folder_name/image_name.jpg>")
        sys.exit(0)
    else:
        img_path = sys.argv[1]
        
    
    # Open the image
    img = get_img(img_path)
    print(img.shape)
    
    # Load the model, 
    # mention path = 'trained model path', incase of loading a specific model
    model = get_model(img)
    
    # Make predictions
    x, y = get_predictions(img, model)
    print(f'{x:.4f} {y:.4f}')