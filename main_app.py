import numpy as np
import streamlit as st
from cv2 import cv2
from keras.models import load_model

# load the model:
model = load_model('Dog_Breed_Prediction.h5')

# Classes:
class_ = ['scottish_deerhound', 'maltese_dog', 'bernese_mountain_dog']

# Title of the app:
st.title("Dog Breed Prediction")
st.markdown("Upload an image of a dog")

# Uploading image:
image = st.file_uploader("Choose any image.", type = "png")
submit = st.button("Predict its breed")

if submit:
    if image is not None:
        # Convert it into opencv image:

        file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        # Display the image:
        st.image(opencv_image, channels='BGR')

        # Resize:
        opencv_image = cv2.resize(opencv_image, (224,224))
        opencv_image.shape = (1,224,224,3)

        # Prediction:
        Y_pred = model.predict(opencv_image)

        st.title(str("The dog breed is "+class_[np.argmax(Y_pred)]))

