import streamlit as st
from skin import segment_image
from skin import predict
from PIL import Image
import brain
import os
import numpy as np
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = ""
uploaded_file = st.file_uploader("Choose a file", type=["tiff", "jpg", "jpeg", "png"])

def main():
    if uploaded_file is not None:
        if uploaded_file.name.endswith((".jpg", ".jpeg", ".png")):
            st.write("You uploaded a skin Image")

            if uploaded_file is not None:
                img = Image.open(uploaded_file)
                original_img, pred_img, cropped_img, segmented_image = segment_image(img)
                
                classification_result, confidence = predict(cropped_img)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.image(original_img, caption='Original Image', use_column_width=True)
                
                with col2:
                    st.image(pred_img.reshape(224, 224), caption='Predicted Output', use_column_width=True, channels='GRAY')
                
                with col3:
                    st.image(segmented_image, caption='segmented Image', use_column_width=True)
                    
                st.write("Classification Result:", classification_result)
                st.write("Confidence:", confidence)

        elif uploaded_file.name.endswith(".tif"):
            st.write("You uploaded a TIFF file.")

            im_width = 256
            im_height = 256
            img = Image.open(uploaded_file)
            img_array = np.array(img)

            slice_index = img_array.shape[2] // 2
            img_slice = img_array[:, :, slice_index]

            img_slice = cv2.resize(img_slice, (im_height, im_width))
            img_slice = img_slice / np.max(img_slice)

            img_slice_rgb = np.stack((img_slice,) * 3, axis=-1)

            img_slice_rgb = img_slice_rgb[np.newaxis, :, :, :]

            pred = brain.model.predict(img_slice_rgb)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader('Original Image')
                st.image(np.squeeze(img_slice_rgb), caption='Original Image.')

            with col2:
                st.subheader('Prediction')
                overlay_pred = np.zeros_like(np.squeeze(img_slice_rgb))
                overlay_pred[np.squeeze(pred) > 0.5] = [1, 0, 0] 
                st.image(overlay_pred, caption='Prediction')

            with col3:
                st.subheader('Overlay')
                overlay_img = np.squeeze(img_slice_rgb.copy())
                overlay_img[np.squeeze(pred) > 0.5] = [1, 0, 0]
                st.image(overlay_img, caption='Overlay')

        else:
            st.write("Unsupported file format. Please upload a JPG, JPEG, PNG, or TIFF file.")

if __name__ == "__main__":
    main()
