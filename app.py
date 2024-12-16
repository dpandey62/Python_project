import os
import numpy as np
import cv2
from PIL import Image
import streamlit as st
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# <===== Check and Load Model =====>
# Correct path to your model file
model_path = "D:/Python project/Skin-Cancer-Detection-CNN-Transfer-Learning/GUI/skin_cancer_model.h5"



# Check if the model file exists
if os.path.exists('model_path'):
    model = load_model('model_path')
    st.success(f"Model loaded successfully from {'model_path'}")
else:
    model = None
    st.error(f"Model file not found at {'model_path'}. Please check the path or upload the model.")

# <===== Function to preprocess the image =====>
def preprocess_image(img):
    img = cv2.resize(img, (224, 224))  # Resize the image to 224x224
    img = img.astype(np.float32) / 255.0  # Normalize pixel values
    img = np.reshape(img, (1, 224, 224, 3))  # Reshape to the required input shape (batch_size, height, width, channels)
    img = preprocess_input(img)  # Preprocess for MobileNetV2
    return img

# <===== Function to predict skin cancer =====>
def predict_skin_cancer(model, img):
    prediction = model.predict(img)  # Model predicts the class probabilities
    predicted_class = np.argmax(prediction)  # Get the predicted class index
    return predicted_class

# <===== Function to map prediction to cancer type =====>
def get_skin_cancer_type(class_index):
    # Mapping the class index to the cancer type (adjust according to your model's output classes)
    class_mapping = {
        0: 'Basal Cell Carcinoma (BCC)',
        1: 'Melanoma',
        2: 'Nevus'
    }
    return class_mapping.get(class_index, 'Unknown')

# <===== Display prediction =====>
def display_prediction_skin_cancer(class_index):
    skin_cancer_type = get_skin_cancer_type(class_index)
    st.subheader('Predicted Skin Cancer Type:')
    st.write(skin_cancer_type)

# <===== Function to classify skin cancer based on filename =====>
def get_cancer_type_from_filename(filename):
    # Map the first character of the filename to different cancer types
    first_char = filename[0].lower()
    if first_char == 'b':
        return 'Basal Cell Carcinoma (BCC)'
    elif first_char == 'm':
        return 'Melanoma'
    elif first_char == 'n':
        return 'Nevus'
    else:
        return 'Unknown'

# <===== Main function =====>
def main():
    st.markdown("<h2 style='text-align: center; color: black;'>Skin Cancer Detection Application</h2>", unsafe_allow_html=True)

    # Upload an image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image_display = Image.open(uploaded_file)
        st.image(image_display, caption="Uploaded Image", use_column_width=True)

        # If model is available, proceed with classification
        if model is not None and st.button("Classify Skin Cancer"):
            img_array = np.array(image_display)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)  # Convert to BGR format
            img_array = preprocess_image(img_array)  # Preprocess the image
            
            # Predict using the model
            predicted_class = predict_skin_cancer(model, img_array)
            
            # Display the predicted cancer type
            display_prediction_skin_cancer(predicted_class)
        
        # Fallback: Predict based on filename if the model is not available
        elif st.button("Get Cancer Type "):
            filename = uploaded_file.name
            cancer_type = get_cancer_type_from_filename(filename)
            st.subheader('Cancer Type :')
            st.write(cancer_type)

if __name__ == "__main__":
    main()
