import streamlit as st
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

import itertools
import random


def page_leaf_visualiser_body():
    st.write('### Leaf Visualisation')
    st.info(
        f'The client has asked for a page to visualise:'
        f'Average images and variability images for each class which is healthy or powdery mildew.'
        f'The difference between average healthy and average powdery mildew cherry leaves.'
        f'They have also asked for an image montage for each class.'
        )

    option = st.radio(
    'Choose the visualisation you would like to see:',
    ('Average & Variability - Healthy:',
     'Average & Variability - Powdery Mildew:',
     'Standard Deviation - Class Comparison,'
     'Image Montage - Healthy',
     'Image Montage - Powdery Mildew')
    )

    if option == 'Image Montage - Healthy':
        st.write('### Image Montage')
        image_montage_healthy = Image.open('outputs/images/healthy_img_montage.png')
        st.image(image_montage_healthy)   

    if option == 'Image Montage - Powdery Mildew':
        st.write('### Image Montage - Powdery Mildew') 
        image_montage_powdery_mildew = Image.open('outputs/images/powdery_mildew_img_montage.png')
        st.image(image_montage_powdery_mildew)   
