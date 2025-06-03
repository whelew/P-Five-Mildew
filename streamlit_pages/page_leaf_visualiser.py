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
    
    def check_image_path_exists(image_path, caption=None):
        if os.path.exists(image_path):
            image = Image.open(image_path)
            st.image(image, caption=caption, use_container_width=True)
        else:
            st.error(f'Image path not found {image_path}')

    option = st.radio(
    'Choose the visualisation you would like to see',
    ('Average & Variability - Healthy',
     'Average & Variability - Powdery Mildew',
     'Standard Deviation - Class Comparison',
     'Image Montage - Healthy',
     'Image Montage - Powdery Mildew')
    )

    if option == 'Image Montage - Healthy':
        st.write('### Image Montage - Healthy')
        image_montage_healthy = 'outputs/images/healthy_img_montage.png'
        check_image_path_exists(image_montage_healthy) 

    elif option == 'Image Montage - Powdery Mildew':
        st.write('### Image Montage - Powdery Mildew') 
        image_montage_powdery_mildew = 'outputs/images/powdery_mildew_img_montage.png'
        check_image_path_exists(image_montage_powdery_mildew) 

    elif option == 'Average & Variability - Healthy':
        st.write('### Average & Variability - Healthy') 
        avg_healthy = 'outputs/images/variability_within_healthy_images.png'
        check_image_path_exists(avg_healthy) 
    
    elif option == 'Average & Variability - Powdery Mildew':
        st.write('### Average & Variability - Powdery Mildew') 
        avg_powdery_mildew = 'outputs/images/variability_within_powdery_mildew_images.png'
        check_image_path_exists(avg_powdery_mildew) 
    
    elif option == 'Standard Deviation - Class Comparison':
        st.write('### Standard Deviation - Class Comparison') 
        std_class_comp = 'outputs/images/std_between_img_classes.png'
        check_image_path_exists(std_class_comp) 
