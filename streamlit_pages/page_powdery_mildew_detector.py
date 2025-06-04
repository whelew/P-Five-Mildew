from huggingface_hub import hf_hub_download
import os 
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from keras.preprocessing.image import img_to_array
from keras.models import load_model


@st.cache_resource
def load_final_model():
    model_path = hf_hub_download(
        repo_id="whelew/cherry-leaf-mildew-detector",
        filename="cherry_leaf_classifier_final_model.keras"
    )
    return load_model(model_path)

final_model = load_final_model()

def page_powdery_mildew_detector():
    st.info("""The client would like to be able to predict whether 
            a cherry leaf contains powdery mildew or is healthy based 
            on an image. The model was designed to predict to a metric 
            accuracy score of at least 97%.
            """)
    
    st.success('Here is a link to a live data set containing 2104 healthy cherry leaf images and 2104 powdery mildew cherry leaf images.')
    st.markdown("[Cherry Leaf Dataset](https://www.kaggle.com/datasets/codeinstitute/cherry-leaves)", unsafe_allow_html=True)
    
    def predict_live_data(uploaded_image, model, target_size=(100, 100)):
        # based off notebook prediction function
        img = Image.open(uploaded_image).convert('RGB')
        img = img.resize(target_size)
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prob = model.predict(img_array)[0][0]
        label = 'powdery mildew' if prob > 0.5 else 'healthy'

        return label, prob
    
    uploaded_images = st.file_uploader('Upload Your Cherry Leaf Image', type=['jpeg', 'png', 'jpg'], accept_multiple_files=True)

    if uploaded_images is not None:
        st.subheader('Prediction Results:')
        for image in uploaded_images:
            st.image(image, caption='Uploaded Cherry Leaf', use_column_width=True)
            label, prob = predict_live_data(image, final_model)

            if label == 'healthy':
                st.success(f'**Prediction:** {label}')
            else:
                st.warning(f'**Prediction:** {label}')
            st.write(f'**Prediction Probability:** {prob:.2%}')
            st.write('---')
    
