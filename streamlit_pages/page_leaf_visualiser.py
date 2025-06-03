import streamlit as st
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image imread

import itertools
import random

OUTPUT_DIR = 'outputs/images'

def page_leaf_visualiser_body():
    st.write('### Leaf Visualisation')
    st.info(
        f'The client has asked for a page to visualise:'
        f'Average images and variability images for each class which is healthy or powdery mildew.'
        f'The difference between average healthy and average powdery mildew cherry leaves.'
        f'They have also asked for an image montage for each class.'
        )