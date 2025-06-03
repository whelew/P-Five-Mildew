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
    st.info("""
        The client has asked for a page to visualise:
        - Average images and variability images for each class which is healthy or powdery mildew.
        - The difference between average healthy and average powdery mildew cherry leaves.
        - They have also asked for an image montage for each class.
        """)

    option = st.radio(
    'Choose the visualisation you would like to see',
    ('Average & Variability - Healthy',
     'Average & Variability - Powdery Mildew',
     'Standard Deviation - Class Comparison',
     'Image Montage - Healthy',
     'Image Montage - Powdery Mildew')
    )

    if option == 'Image Montage - Healthy':
        st.write('### Image Montage')
        image_montage_healthy = Image.open('outputs/images/healthy_img_montage.png')
        st.image(image_montage_healthy, use_column_width=True)   

    elif option == 'Image Montage - Powdery Mildew':
        st.write('### Image Montage - Powdery Mildew') 
        image_montage_powdery_mildew = Image.open('outputs/images/powdery_mildew_img_montage.png')
        st.image(image_montage_powdery_mildew, use_column_width=True)   

    elif option == 'Average & Variability - Healthy':
        st.write('### Average & Variability - Healthy') 
        avg_healthy = Image.open('outputs/images/variability_within_healthy_images.png')
        st.image(avg_healthy, use_column_width=True) 
    
    elif option == 'Average & Variability - Powdery Mildew':
        st.write('### Average & Variability - Powdery Mildew') 
        avg_powdery_mildew = Image.open('outputs/images/variability_within_powdery_mildew_images.png')
        st.image(avg_powdery_mildew, use_column_width=True) 
    
    elif option == 'Standard Deviation - Class Comparison':
        st.write('### Standard Deviation - Class Comparison') 
        std_class_comp = Image.open('outputs/images/std_between_img_classes.png')
        st.image(std_class_comp, use_column_width=True) 
