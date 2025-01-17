import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import cv2
from glob import glob
from itertools import chain
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import streamlit as st
from PIL import Image

im_width = 256
im_height = 256

train_files = []
mask_files = glob('kaggle_3m/*/*_mask*')

for i in mask_files:
    train_files.append(i.replace('_mask',''))
    
    
df = pd.DataFrame(data={"filename": train_files, 'mask' : mask_files})
df_train, df_test = train_test_split(df,test_size = 0.1)
df_train, df_val = train_test_split(df_train,test_size = 0.2)

smooth=100

def dice_coef(y_true, y_pred):
    y_truef=K.flatten(y_true)
    y_predf=K.flatten(y_pred)
    And=K.sum(y_truef* y_predf)
    return((2* And + smooth) / (K.sum(y_truef) + K.sum(y_predf) + smooth))

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def iou(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    sum_ = K.sum(y_true + y_pred)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac

def jac_distance(y_true, y_pred):
    y_truef=K.flatten(y_true)
    y_predf=K.flatten(y_pred)

    return - iou(y_true, y_pred)
    
    
model = load_model('unet_brain_mri_seg.hdf5', custom_objects={'dice_coef_loss': dice_coef_loss, 'iou': iou, 'dice_coef': dice_coef})

im_height, im_width = 256, 256

def predict_and_display(image_path):
    img = Image.open(image_path)
    img_array = np.array(img)

    slice_index = img_array.shape[2] // 2
    img_slice = img_array[:, :, slice_index]

    img_slice = cv2.resize(img_slice, (im_height, im_width))
    img_slice = img_slice / np.max(img_slice)

    img_slice_rgb = np.stack((img_slice,) * 3, axis=-1)

    img_slice_rgb = img_slice_rgb[np.newaxis, :, :, :]

    pred = model.predict(img_slice_rgb)

    # Assuming img_slice_rgb and pred are defined earlier

    # st.subheader('Original Image')
    # st.image(np.squeeze(img_slice_rgb), caption='Original Image.')

    # Create three columns to display images side by side
    col1, col2, col3 = st.columns(3)

    # Display prediction in the first column
    with col1:
        st.subheader('Original Image')
        st.image(np.squeeze(img_slice_rgb), caption='Original Image.')

    # Display overlay in the second column
    with col2:
        st.subheader('Prediction')
        overlay_pred = np.zeros_like(np.squeeze(img_slice_rgb))
        overlay_pred[np.squeeze(pred) > 0.5] = [1, 0, 0] 
        st.image(overlay_pred, caption='Prediction')

    # Leave the third column empty for spacing
    with col3:
        st.subheader('Overlay')
        overlay_img = np.squeeze(img_slice_rgb.copy())
        overlay_img[np.squeeze(pred) > 0.5] = [1, 0, 0]
        st.image(overlay_img, caption='Overlay')

st.title('MRI Image Segmentation App')
st.write('Upload a TIFF image to perform segmentation.')

uploaded_file = st.file_uploader("Choose a TIFF file", type="tiff")

if uploaded_file is not None:
    predict_and_display(uploaded_file)
