# Finding-Phone

## Problem Statement:
Implement a prototype of a visual object detection system for a customer. The task is to find a location of a phone dropped on the floor from a single RGB camera image.

## 1. Files Descriptions
    * train_phone_finder.py : Python script to train the inceptionV3 Net.
    * find_phone.py         : Python script to make predictions on single image.
    * utils.py              : Script containing all the supporting common functions.
    * Phone_Finder.ipynb    : Detailed description of different approaches taken to solve the problem.
    * MSE_plot.png          : MSE plot for training the model.
    * models/               : Folder that contains all the models (generated after training)
    * find_phone/           : Folder with all the images.

## 2. Future Work/ Improvements:
Because of lack of time I did not disect individual approaches, but following steps could be taken to get better results:
* **Data Augmentation:**
    * Since the dataset size is very small to train any model from scratch, an alternative approach to the problem could have been to either gather more data or use different data augmentatoion techniques like flipping, rotating, changing color etc to generate more data.
* **Tranfer Learning:**
    * Other well trained models could been used in place of inceptionV3 like YOLOv5, Fast RCNN etc. But the objective was mostly `object localization`, I felt inceptionV3 also performs well.
    * Other popular neural networks could be ViT(Vision Transformers) and performing few-shot training to use ViTs for specific task at hand. This solution comes at a computational cost.

