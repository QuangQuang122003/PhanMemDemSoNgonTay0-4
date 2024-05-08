import streamlit as st
import cv2
import numpy as np
from keras.models import  load_model
from keras.preprocessing.image import img_to_array, load_img
import pickle
from PIL import Image, ImageDraw

# Load the model
model = load_model('model_bbox_regression_and_classification_VGG16.h5')

st.title('Finger Count Prediction')

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = load_img(uploaded_file, target_size=(64, 64))
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Predicting...")

    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    predictions = model.predict(image)

    # The model returns two arrays. The first array contains the bounding box coordinates and the second array contains the class probabilities.
    # We are interested in the class probabilities.
    class_probabilities = predictions[1][0]
    bbox = predictions[0][0]

    # Find the class with the highest probability
    predicted_class = np.argmax(class_probabilities)

    # Display the prediction
    st.write("The predicted number of fingers is: ", predicted_class)

    # Draw bounding box
    image = Image.open(uploaded_file)
    draw = ImageDraw.Draw(image)

    # If the bounding box values are relative (from 0 to 1), we need to multiply them by the image size
    bbox = bbox * np.array([image.width, image.height, image.width, image.height])

    # Convert the bounding box coordinates to integer
    bbox = [int(coordinate) for coordinate in bbox]  # Corrected line

    draw.rectangle([(bbox[0], bbox[1]), (bbox[2], bbox[3])], outline="green")

    # Add label
    draw.text((bbox[0], bbox[1] - 10), f"{predicted_class}", fill="green")

    # Display image with bounding box
    st.image(image, caption='Image with Bounding Box.', use_column_width=True)
