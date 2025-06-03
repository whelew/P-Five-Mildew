import streamlit as st
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

import itertools
import random

def page_ml_performance_metrics():

    st.header('### Train, Validation and Test Set: Label Frequencies')

    st.info("""
            These bar plots show the distribution of labels/classes (Healthy and Powdery Mildew) 
            for the train, validation and test sets. As you can see there is a balanced
            distribution between all sets.
        """)

    col1, col2, col3 = st.columns(3)
    with col1:
        train_labels = Image.open('outputs/images/train_set_bar_plot.png')
        st.image(train_labels, caption='Train Label Distribution')
    with col2:
        val_labels = Image.open('outputs/images/val_set_bar_plot.png')
        st.image(val_labels, caption='Validation Label Distribution')
    with col3:
        test_labels = Image.open('outputs/images/test_set_bar_plot.png')
        st.image(test_labels, caption='Test Label Distribution')
    st.write('---')

    st.subheader('Best Model Summary from Hyperparameter Tuning')
    st.info(""" Before creating the final model, I used keras tuner to find the 
            best hyperparameters for model optimisation. The results are shown here. 
        """)
    summary_path = 'outputs/logs/tuner_results_summary.txt'
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            tuner_sumamry = f.read()
        st.text(tuner_sumamry)
    else:
        st.error('Tuner results summary not found')